import torch
import torch.nn as nn
import torch.nn.functional as F

# DINOv2 패치 토큰을 2D 그리드로 변환
def token_to_grid(tokens, H_grid, W_grid):
    # cls 토큰 제외
    x = tokens[:, 1:, :].permute(0, 2, 1)
    return x.reshape(x.shape[0], x.shape[1], H_grid, W_grid)

# U-Net 스타일 Conv Block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

# ----------------- Attention 모듈 -----------------
# SE-Block (채널 Attention)
class SE_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# SE-Block이 추가된 Conv Block (Arch 2)
class SE_ConvBlock(ConvBlock):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        self.block.add_module("se_block", SE_Block(out_channels))

# Spatial Attention (CBAM용)
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_out = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv(x_out)
        return self.sigmoid(x_out)

# CBAM (복합 Attention)
class CBAM_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.ca = SE_Block(channel, reduction) # 채널 Attention은 SE-Block 재활용
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x # Channel Attention
        x = self.sa(x) * x # Spatial Attention
        return x

# CBAM이 추가된 Conv Block (Arch 3)
class CBAM_ConvBlock(ConvBlock):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        self.block.add_module("cbam_block", CBAM_Block(out_channels))
        

class SegDecoder(nn.Module):
    def __init__(self, decoder_type, vit_dim=384, num_classes=1, img_size=224):
        super().__init__()
        self.img_size = img_size
        self.grid_size = img_size // 14
        self.decoder_type = decoder_type
        
        # Conv Block 선택 (Architecture에 따라 교체)
        if decoder_type == 'simple_mlp':
            self.ConvBlock = ConvBlock # Architecture 0
        elif decoder_type == 'unet_style':
            self.ConvBlock = ConvBlock # Architecture 1
        elif decoder_type == 'cnn_se':
            self.ConvBlock = SE_ConvBlock # Architecture 2 (Channel Attn)
        elif decoder_type == 'cnn_cbam':
            self.ConvBlock = CBAM_ConvBlock # Architecture 3 (Composite Attn)
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

        if decoder_type == 'simple_mlp':
            self._build_simple_mlp(vit_dim, num_classes)
        else:
            self._build_unet_style(vit_dim, num_classes)

    # ------------------ Architecture 0: Simple MLP Head ------------------
    def _build_simple_mlp(self, vit_dim, num_classes):
        self.initial_conv = nn.Conv2d(vit_dim, 128, kernel_size=1)
        self.refine_conv = ConvBlock(128, 64) 
        self.output_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def _forward_simple_mlp(self, final_features):
        x = token_to_grid(final_features, self.grid_size, self.grid_size)
        x = self.initial_conv(x)
        x = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=False) 
        x = self.refine_conv(x)
        return self.output_conv(x)

    # ------------------ Architecture 1, 2, 3: U-Net Style Heads ------------------
    def _build_unet_style(self, vit_dim, num_classes):
        # Decoder 층에 선택된 Conv Block 사용
        DecoderConvBlock = self.ConvBlock
        
        # 0. 초기 특징 압축
        self.initial_conv = DecoderConvBlock(vit_dim, 256)
        
        # 1. Stage 1 (16x16 -> 28x28) - Skip: Block 8 특징 (vit_dim)
        self.up_conv1 = DecoderConvBlock(256 + vit_dim, 128)
        self.target_size1 = self.img_size // 8 # 28
        
        # 2. Stage 2 (28x28 -> 56x56) - Skip: Block 4 특징 (vit_dim)
        self.up_conv2 = DecoderConvBlock(128 + vit_dim, 64)
        self.target_size2 = self.img_size // 4 # 56
        
        # 3. Final Stage (56x56 -> 224x224)
        self.final_refine = DecoderConvBlock(64, 32)
        self.output_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def _forward_unet_style(self, final_features, mid_features, low_features):
        B = final_features.shape[0]
        H_grid = W_grid = self.grid_size
        
        # 1. 초기 그리드 변환 및 압축 (Final Feature)
        x = token_to_grid(final_features, H_grid, W_grid)
        x = self.initial_conv(x) # B x 256 x 16 x 16

        # 2. Stage 1: Mid-Level 특징 통합 (Block 8)
        mid_skip = token_to_grid(mid_features, H_grid, W_grid)
        x_up = F.interpolate(x, size=self.target_size1, mode='bilinear', align_corners=False) 
        mid_skip_up = F.interpolate(mid_skip, size=self.target_size1, mode='bilinear', align_corners=False) 
        x = torch.cat([x_up, mid_skip_up], dim=1)
        x = self.up_conv1(x) # B x 128 x 28 x 28

        # 3. Stage 2: Low-Level 특징 통합 (Block 4)
        low_skip = token_to_grid(low_features, H_grid, W_grid)
        x_up = F.interpolate(x, size=self.target_size2, mode='bilinear', align_corners=False) 
        low_skip_up = F.interpolate(low_skip, size=self.target_size2, mode='bilinear', align_corners=False) 
        x = torch.cat([x_up, low_skip_up], dim=1)
        x = self.up_conv2(x) # B x 64 x 56 x 56

        # 4. Final Stage: 56x56 -> 224x224
        x = self.final_refine(x)
        output = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=False)
        return self.output_conv(output)


    def forward(self, final_features, mid_features=None, low_features=None):
        if self.decoder_type == 'simple_mlp':
            return self._forward_simple_mlp(final_features)
        else:
            if mid_features is None or low_features is None:
                 raise ValueError("U-Net style decoders require mid_features and low_features.")
            return self._forward_unet_style(final_features, mid_features, low_features)


# ==============================================================================
#   Swin Transformer Decoder
# ==============================================================================

class SegFormerHead(nn.Module):
    def __init__(self, in_chs, num_cls, emb_ch=256, dropout_ratio=0.1, *args, **kwargs) -> None:
        """
        SegFormer decoder head.

        Args:
            in_chs (list[int]): Input channel sizes for each stage.
            out_ch (int): Number of output channels/classes.
            emb_ch (int, optional): Embedding channel size.

        Returns:
            None
        """
        super().__init__(*args, **kwargs)

        # Since each stage has a different channel depth, MLP (1x1 Conv) to fit all into embedded_dim (256)
        self.proj = self.linear_layers = nn.ModuleList([
            nn.Conv2d(c, emb_ch, kernel_size=1) for c in in_chs
        ])

        # Fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(emb_ch * 4, emb_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(emb_ch),
            nn.ReLU(inplace=True)
        )

        self.dropout = nn.Dropout(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.classifier = nn.Conv2d(emb_ch, num_cls, kernel_size=1)

    def forward(self, features:list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features (list[torch.Tensor]): List of feature maps from different stages.
        Returns:
            return (torch.Tensor): Segmentation Logit map.
        """

        upsampled_features = []
        target_size = features[0].shape[2:]  
        
        
        for i, x in enumerate(features):
            x:torch.Tensor = self.proj[i](x) 
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            upsampled_features.append(x)

        x = torch.cat(upsampled_features, dim=1)
        x = self.linear_fuse(x)  
        x = self.dropout(x)
        x = self.classifier(x)  

        return x

# TODO UNET Decoder 구현 예정
class UNetDecoder(nn.Module):
    def __init__(self, in_chs, out_channels=256, num_classes=1, *args, **kwargs):
        super().__init__()

        c1, c2, c3, c4 = in_chs  # Swin 4-stage channels

        # 1×1 Conv to unify channel dimension
        self.l1 = nn.Conv2d(c4, out_channels, 1)
        self.l2 = nn.Conv2d(c3, out_channels, 1)
        self.l3 = nn.Conv2d(c2, out_channels, 1)
        self.l4 = nn.Conv2d(c1, out_channels, 1)

        # UNet upsampling blocks
        self.up1 = self._up_block(out_channels, out_channels)
        self.up2 = self._up_block(out_channels, out_channels)
        self.up3 = self._up_block(out_channels, out_channels)
        self.up4 = self._up_block(out_channels, out_channels)

        # Final output conv
        self.out_conv = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def _up_block(self, in_ch, out_ch):
        # Simple conv block (Conv → BN → ReLU → Conv → BN → ReLU)
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats):
        """
        feats: List of 4 scales:
            feats[0] : Stage1  (B, C1, H/4, H/4)
            feats[1] : Stage2  (B, C2, H/8, H/8)
            feats[2] : Stage3  (B, C3, H/16, H/16)
            feats[3] : Stage4 fused (B, C4, H/32, H/32)
        """
        f1, f2, f3, f4 = feats

        # Project all into same channel dim
        f1 = self.l4(f1)
        f2 = self.l3(f2)
        f3 = self.l2(f3)
        f4 = self.l1(f4)

        # Decoder steps
        x = F.interpolate(f4, scale_factor=2, mode='bilinear', align_corners=False)
        x = x + f3
        x = self.up1(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = x + f2
        x = self.up2(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = x + f1
        x = self.up3(x)

        # Final upsample to original
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up4(x)

        out = self.out_conv(x)
        return out


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, dilation_rates=(1, 6, 12, 18)):
        super().__init__()

        self.branches = nn.ModuleList()

        # 1x1 conv
        self.branches.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        )

        # dilated convs
        for rate in dilation_rates:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

        # global pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilation_rates) + 2), out_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        h, w = x.shape[2:]

        out = []

        # atrous branches
        for branch in self.branches:
            out.append(branch(x))

        # global branch
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=(h, w), mode="bilinear", align_corners=False)
        out.append(gp)

        out = torch.cat(out, dim=1)
        return self.project(out)


class DeepLabV3Decoder(nn.Module):
    def __init__(self, encoder_channels, emb_ch, out_channels=1):
        """
        encoder_channels: 마지막 encoder stage의 output channel 수
        out_channels: segmentation output channel (binary=1)
        """
        super().__init__()

        self.aspp = ASPP(encoder_channels, out_channels=emb_ch)
        
        self.final = nn.Sequential(
            nn.Conv2d(emb_ch, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, out_channels, 1)
        )

    def forward(self, x):
        x = self.aspp(x)
        x = self.final(x)
        return x