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