import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

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

class SimpleDINOv2(nn.Module):
    def __init__(self, backbone_name='dinov2_vits14', freeze_backbone=True, **kwargs):
        super().__init__()
        
        # 1. Backbone (DINOv2)
        print(f"🚀 Loading Backbone: {backbone_name} ...")
        self.backbone = torch.hub.load('facebookresearch/dinov2', backbone_name)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # ---------------------------------------------------------
        # 2. Decoder (Segmentation Head)
        # ---------------------------------------------------------
        
        # [A] Low-Res Processing (뇌: 16x16에서 정보 압축 및 판단)
        # DINO 출력(384) -> 128로 압축
        self.decoder_low = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # [B] High-Res Refinement (눈: 224x224에서 디테일 다듬기)
        # 사용자님이 원하시던 "Upsample 후 처리" 부분입니다.
        # Conv2d를 쓰면 픽셀 단위 MLP와 똑같은 효과 + 주변 정보까지 봅니다.
        self.decoder_high = nn.Sequential(
            # 128채널을 받아서 64로 줄이면서 경계면 부드럽게
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 최종 마스크 (1채널)
            nn.Conv2d(64, 1, kernel_size=1)
        )
        self.apply(self._init_weights)
        print("Weight initalization complete.")

    def forward(self, x):
        # 1. DINO 특징 추출
        self.backbone.eval()
        with torch.no_grad():
            features_dict = self.backbone.forward_features(x)
            patch_tokens = features_dict["x_norm_patchtokens"] # [B, 256, 384]
        
        # 2. 형태 변환 [B, N, C] -> [B, C, H, W]
        B, N, C = patch_tokens.shape
        H_grid = W_grid = int(N ** 0.5) # 16
        x = patch_tokens.permute(0, 2, 1).reshape(B, C, H_grid, W_grid)
        
        # 3. [Low-Res] 핵심 특징 추출 (16x16)
        x = self.decoder_low(x) # -> [B, 128, 16, 16]
        
        # 4. [Upsample] 뻥튀기 (16 -> 224)
        # Bilinear로 늘리면 흐릿해집니다.
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # 5. [High-Res] 디테일 보정 (224x224)
        # 여기서 흐릿해진 경계선을 다시 선명하게 잡아줍니다.
        x = self.decoder_high(x) # -> [B, 1, 224, 224]
        
        return x

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
