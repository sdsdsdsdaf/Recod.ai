import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import math

class FeatureHook:
    def __init__(self, name):
        self.name = name
        self.features = None
    
    # 훅 함수는 (module, input, output) 3개의 인자를 받습니다.
    def __call__(self, module, input, output):
        # output (토큰)을 저장합니다.
        self.features = output

class SMPUnetWithNorm(smp.Unet):
    def __init__(self, **kwargs):
        # 기존 smp.Unet 초기화 그대로 수행
        super().__init__(**kwargs)

        # classification_head가 있을 때만 교체
        if self.classification_head is not None:
            # dropout 값 추출 (기존 aux_params 유지)
            dropout_p = getattr(self.classification_head, 'p', 0.3)
            in_ch = self.encoder.out_channels[-1]

            # ✅ LayerNorm 추가로 안정화된 head 구성
            self.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.LayerNorm(in_ch),
                nn.Dropout(dropout_p),
                nn.Linear(in_ch, 1)
            )

        self.classification_head.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class SMPDeepLabV3PlusWithNorm(smp.DeepLabV3Plus):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.classification_head is not None:
            dropout_p = getattr(self.classification_head, 'p', 0.3)
            in_ch = self.encoder.out_channels[-1]

            self.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.LayerNorm(in_ch),
                nn.Dropout(dropout_p),
                nn.Linear(in_ch, 1),
            )

        self.classification_head.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        if isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from Utils.Decoder import SegDecoder

import torch
import torch.nn as nn
# DINOv2 모델을 로드했다고 가정
# dino_v2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

class DINOv2Extractor(nn.Module):
    def __init__(self, dino_model, low_layer=3, mid_layer=7, final_layer=11, patch_size=14, use_antialias=True, pos_embed_interpolate_offset=0.0):
        super().__init__()
        # DINOv2의 blocks (Transformer Blocks)만 가져옵니다.
        self.dino = dino_model
        self.patch_embed:nn.Module = dino_model.patch_embed
        self.cls_token:torch.Tensor = dino_model.cls_token
        self.pos_embed:nn.Module = dino_model.pos_embed
        self.norm:nn.Module = dino_model.norm # 최종 LayerNorm도 포함
        self.low_layer = low_layer
        self.mid_layer = mid_layer
        self.final_layer = final_layer
        self.patch_size = patch_size
        self.interpolate_antialias = use_antialias
        self.interpolate_offset = 0
    
    def forward(self, x):
        return self.dino(x)

    def interpolate_pos_encoding(self, x:torch.Tensor, w=224, h=224):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)


    def forward_features(self, x:torch.Tensor):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        pos_embed = self.interpolate_pos_encoding(x, H, W)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        x = x + pos_embed 
        
        features = {}
        # 3. Transformer Blocks 순회 (Hook 없이 직접 특징 추출)
        for i, blk in enumerate(self.dino.blocks):
            x = blk(x)
            
            # ViT-B/14 (12개 블록) 기준: 
            # Block 3 (low), Block 7 (mid), Block 11 (final)
            if i == self.low_layer: 
                features['low'] = x.clone()
            if i == self.mid_layer:
                features['mid'] = x.clone()
            if i == self.final_layer:
                features['final'] = x.clone()

        return features

class DINOv2SegmentationModel(nn.Module):
    def __init__(
        self, backbone_name='dinov2_vits14', freeze_backbone=True, use_skip_connections=True, 
        decoder_type='simple_mlp', vit_dim=384, num_classes=1, img_size=224,
        low_layer=3, mid_layer=7, final_layer=11, **kwargs
    ):
        super().__init__()
        
        # 1. Backbone (DINOv2)
        print(f"🚀 Loading Backbone: {backbone_name} ...")
        self.backbone = DINOv2Extractor(torch.hub.load('facebookresearch/dinov2', backbone_name))
        self.use_skip_connections = use_skip_connections
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # ---------------------------------------------------------
        # 2. Decoder (Segmentation Head)
        # ---------------------------------------------------------
        
        # [A] Low-Res Processing (뇌: 16x16에서 정보 압축 및 판단)
        # DINO 출력(384) -> 128로 압축
        self.decoder = SegDecoder(decoder_type, vit_dim, num_classes, img_size)
        self.apply(self._init_weights)
        print("Weight initalization complete.")

    def forward(self, x):
        # 1. DINO 특징 추출
        # cls_feat = self.backbone(x)
        feat = self.backbone.forward_features(x)
        final_f, mid_f, low_f = feat['final'], feat['mid'], feat['low']
        
        if self.decoder.decoder_type == 'simple_mlp' or self.use_skip_connections is False:
             return self.decoder(final_f)
        else:
             return self.decoder(final_f, mid_f, low_f)

    # 모든 모듈에 대해 초기화를 적용하는 헬퍼 함수
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            # Kaiming Normal (He) 초기화 적용 (ReLU 사용 시 표준)
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            # BatchNorm 초기화 (weight=1, bias=0)
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
