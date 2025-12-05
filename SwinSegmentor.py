# ==============================================================================
#   Swin Transformer Decoder
# ==============================================================================

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F



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
    

class SwinTransformerSegmentationModel(nn.Module):
    def __init__(self, backbone_name='swin_base_patch4_window7_224', decoder_type='segformer',  num_classes=1, emb_ch=256, dropout_ratio=0.1, *args,**kwargs):

        """
        Swin Transformer based Segmentaion Model.

        Args:
            backbone_name (str): Swin Transformer model name.
            decoder_type (str): decoder type ['segformer'].
            num_classes (int): Number of output class.
            emb_ch (int): Decoder`s embedding channel size.
            dropout_ratio (float): Decoder`s dropout ratio.
        Returns:
            torch.Tensor: Segmentation logits of shape (B, num_classes, H, W).
        """

        super().__init__(*args,**kwargs)
        
        # 1. Backbone (Swin Transformer)
        print(f"🚀 Loading Backbone: {backbone_name} ...")
        self.backbone = timm.create_model(backbone_name, pretrained=False, pretrained_cfg=None, features_only=True)
        print("Backbone loaded.")
        print(f"Backbone output channels: {self.backbone.feature_info.channels()}")

        # ---------------------------------------------------------
        # 2. Decoder (Segmentation Head)
        # ---------------------------------------------------------
        in_chs = self.backbone.feature_info.channels()
        if decoder_type.lower() == 'segformer':
            self.decoder = SegFormerHead(in_chs=in_chs, num_cls=num_classes, emb_ch=emb_ch, dropout_ratio=dropout_ratio)
        elif decoder_type.lower() == 'unet':
            self.decoder = UNetDecoder(in_chs=in_chs, num_classes=num_classes, out_channels=emb_ch, dropout_ratio=dropout_ratio)
        else:
            raise ValueError(f"Unsupported decoder type: {decoder_type}")
        
        # self.decoder.apply(init_weights)
        print("Weight initalization complete.")
        
    def forward(self, x:torch.Tensor):
        """
        Forward pass through the Swin Transformer backbone and segmentation head.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Segmentation logits of shape (B, num_classes, H, W).
        """
        # x: [Batch, 3, H, W] (ex: 224x224)

        input_size = x.shape[-2:]
        features:list[torch.Tensor] = self.backbone(x)
        if features[0].ndim == 4 and features[0].shape[1] not in self.backbone.feature_info.channels():
            features = [f.permute(0, 3, 1, 2).contiguous() for f in features]

        logit_map = self.decoder(features)

        # features[0]: [B, 128, H/4, W/4]  (Detail, Forge Detect Low level Context) 
        # features[1]: [B, 256, H/8, W/8]
        # features[2]: [B, 512, H/16, W/16]
        # features[3]: [B, 1024, H/32, W/32] (Global, High level Context)

        logit_map = F.interpolate(logit_map, size=input_size, mode='bilinear', align_corners=False)

        return logit_map
    

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