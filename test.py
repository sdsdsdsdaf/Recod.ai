import segmentation_models_pytorch as smp
import torch
import pandas as pd
import os
import transformers as test
import segmentation_models_pytorch as smp
dir = r'C:\Users\user\.cache\kagglehub\competitions\recodai-luc-scientific-image-forgery-detection'


model = smp.Unet(encoder_name="efficientnet-b3", encoder_weights="imagenet", #모델 파라미터
        in_channels=3, classes=1, activation=None,  aux_params={
        "classes": 1,           # 출력 클래스 개수
        "pooling": "avg",       # global avg pooling
        "dropout": 0.3,
        "activation": "sigmoid" # optional
    })

print(model.classification_head)

