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

    
