import segmentation_models_pytorch as smp
import torch

model = smp.Unet(
    encoder_name="efficientnet-b3",     # backbone
    encoder_weights="imagenet",         # pretrained 가중치
    in_channels=3,                      # 입력 채널 수 (RGB 이미지)
    classes=1,                          # 출력 채널 수 (binary mask)
    activation=None,                    # ⚠️ sigmoid는 loss 함수 쪽에서 처리
)

print(model.state_dict().keys())
model.load_state_dict(torch.load("fast_model.pth"))