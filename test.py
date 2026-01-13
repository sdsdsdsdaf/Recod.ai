from pathlib import Path
import os
import timm
import segmentation_models_pytorch as smp
import torch
from Utils.utils import freeze_encoder_after_epoch

print(timm.list_models("*swin*", pretrained=True))

model = smp.FPN(
    encoder_name="tu-swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
        encoder_weights="imagenet",   # 또는 None
        in_channels=3,
        classes=1,
)

from torchinfo import summary
summary(
    model,
    input_size=(1, 3, 384, 384),
    depth=4  # 깊이 늘리면 더 많이 보임
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(model.encoder.model)

if model.encoder.model.layers_0.__class__.__name__ == "SwinTransformerV2Stage":
    print("yes")



optimizer = freeze_encoder_after_epoch(model, 0, 0, optimizer, n_layers=1,)

for name, param in model.named_parameters():
    print(name, param.requires_grad)

