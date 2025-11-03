import torch
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader, Subset
from Utils.Dataset import HybridDataset
from Utils.utils import train, evaluate, set_seed
import torch.nn as nn
import segmentation_models_pytorch as smp
import cv2
from datetime import timedelta


def cross_val_score(
        model_cls,
        k:int=5,
        dataset:HybridDataset=None,
        random_state:int=42,
        ues_pin_memory:bool=False,
        num_workers:int=0,
        device=None,
        batch_size=16,
        cls_loss=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(26.33)),
        dice_loss=smp.losses.DiceLoss(mode='binary', from_logits=True),
        alpha=0.5,
        beta=0.5,
        epoch=10,
        interpolation: int = cv2.INTER_NEAREST, 
        threshold: float = 0.5,
        min_area: int = 100, 
        low_conf_max_prob: float = 0.06, 
        low_viz_thr: float = 0.04, 
        low_conf_min_pixel: int = 128,
        optimizer_cls=torch.optim.Adam,
        scheduler_cls=None,
        use_amp=False,
        **kwargs,
    ):

    if optimizer_cls is None:
        raise ValueError("optimizer must be provided")

    kfold = KFold(n_splits=k, shuffle=True, random_state=random_state)
    result = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\nFOLD [{fold + 1}/{k}]...")
        fold_start_time = time()

        model = model_cls(**kwargs)
        optimizer = optimizer_cls(model.parameters())
        scheduler = scheduler_cls(optimizer) if scheduler_cls else None
        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)

        train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=ues_pin_memory)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=ues_pin_memory)

        model = model.to(device)
        log = train(
            model, train_loader, val_loader, optimizer, epoch,
            device, cls_loss, dice_loss, alpha, beta, scheduler,
            interpolation=interpolation, threshold=threshold,
            min_area=min_area, low_conf_max_prob=low_conf_max_prob, 
            low_viz_thr=low_viz_thr, low_conf_min_pixel=low_conf_min_pixel,
            fold=fold+1, use_amp=use_amp,
        )
        result.append(log)

        
        score = evaluate(
            model, val_loader, device, cls_loss, dice_loss, alpha, beta,
            interpolation=interpolation, threshold=threshold,
            min_area=min_area, low_conf_max_prob=low_conf_max_prob, 
            low_viz_thr=low_viz_thr, low_conf_min_pixel=low_conf_min_pixel
        )
        fold_end_time = time()
        elapsed_time = fold_end_time - fold_start_time
        print(f"Final Score F1: {score} TOTAL LOSS: {score['loss_total']:.4f} CLS LOSS: {score['loss_cls']:.4f} DICE LOSS: {score['loss_dice']:.4f} FOLD{fold + 1} TIME: {timedelta(seconds=elapsed_time)}")



    return result



      

if __name__ == "__main__":

    import os
    from time import time

    set_seed()

    from pathlib import Path
    import os
    import platform

    COMP_DIR = r"C:\Users\user\.cache\kagglehub\competitions\recodai-luc-scientific-image-forgery-detection"
    TEST_DIR = os.path.join(COMP_DIR, "test_images")
    TRAIN_DIR = os.path.join(COMP_DIR, "train_images")
    MODEL_PATH = os.path.join(r"C:\Users\user\.cache\kagglehub\models\aikim12345689\smp-unet\PyTorch\smp2\4", "SMP_UNet.pth")

    # Output
    OUT_DIR = "/kaggle/working"
    os.makedirs(OUT_DIR, exist_ok=True)
    SUB_PATH = os.path.join(OUT_DIR, "submission.csv")

    #Interpolation Method
    INTERPOLATION = cv2.INTER_NEAREST

    #Learning HyperParams
    IMG_SIZE = 256
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LR = 1e-4
    bce_loss_cls = nn.BCEWithLogitsLoss
    dice_loss_cls = smp.losses.DiceLoss
    alpha = 0.5  # Weight for combining BCE and Dice losses
    beta = 1 - alpha
    optimizer_cls = torch.optim.Adam

    #Post Processing HyperParams
    THRESHOLD = 0.5
    LOW_CONF_MAX_PROB = 0.06
    LOW_VIZ_THR = 0.04
    LOW_CONF_MIN_PIXEL = 128
    MIN_AREA = 64
    THRESHOLD = 0.5
    MIN_AREA = 16
    TRAIN_SAMPLE_NUM = None
    TEST_SAMPLE_NUM = None


    #Aceleration Params -> Default: cpu settings
    USE_PIN_MEM = torch.cuda.is_available
    NUM_WORKERS = 4 if torch.cuda.is_available() and "Windows" not in platform.platform() else 0
    USE_AMP = True

    print("COMP_DIR: ", COMP_DIR)
    print("TEST_DIR: ", TEST_DIR)
    print("TRAIN_DIR: ", TRAIN_DIR)
    print("MODEL_PATH: ", MODEL_PATH)
    print("SUB_PATH: ", SUB_PATH)
    print("MODEL_PATH is exists: ", os.path.exists(MODEL_PATH))
    print("USE AMP: ", USE_AMP)
    print()
    paths = {
        'train_authentic': os.path.join(TRAIN_DIR, "authentic"),
        'train_forged': os.path.join(TRAIN_DIR, "forged"),
        'train_masks': os.path.join(COMP_DIR, "train_masks"),
        'test_images': TEST_DIR
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_ds = HybridDataset(
        "train_data.h5",
        paths['train_authentic'],
        paths['train_forged'],
        paths['train_masks'],
        img_size=IMG_SIZE,
        is_train=True,
        preload=True,
        verbose=True
    )
    
    cv_start_time = time()
    train_log = cross_val_score(
        smp.Unet, k=5, dataset=full_ds, device=device, batch_size=BATCH_SIZE,
        ues_pin_memory=USE_PIN_MEM, num_workers=NUM_WORKERS,
        cls_loss=bce_loss_cls(pos_weight=torch.tensor(26.33)),
        dice_loss=dice_loss_cls(mode='binary', from_logits=True),
        alpha=alpha, beta=beta, epoch=NUM_EPOCHS, interpolation=INTERPOLATION,
        threshold=THRESHOLD, min_area=MIN_AREA, low_conf_max_prob=LOW_CONF_MAX_PROB,
        low_viz_thr=LOW_VIZ_THR, low_conf_min_pixel=LOW_CONF_MIN_PIXEL,
        optimizer_cls=optimizer_cls, scheduler_cls=None, use_amp=USE_AMP,

        encoder_name="efficientnet-b3", encoder_weights="imagenet", #모델 파라미터
        in_channels=3, classes=1, activation=None,                    
    )

    print(f"CV is complete! Total Time: {timedelta(seconds=time() - cv_start_time)}")

    import pickle as pkl

    with open("train_log.pkl", "wb") as f:
        pkl.dump(train_log, f)
        print("\nLog save complete! \n")