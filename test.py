import segmentation_models_pytorch as smp
import torch
import pandas as pd
import os
import cv2 as test
dir = r'C:\Users\user\.cache\kagglehub\competitions\recodai-luc-scientific-image-forgery-detection'


df = pd.read_csv(os.path.join(dir,'sample_submission.csv'))
print(test.__version__)