import torch
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader, Subset
from Utils.Dataset import HybridDataset
from Utils.utils import compute_pos_weight, train, evaluate, set_seed
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import FocalLoss
import cv2
from datetime import timedelta
from time import time
from transformers import get_cosine_schedule_with_warmup
from Utils.Model import SMPUnetWithNorm
from Utils.Loss import FocalTverskyLoss
from torch.utils.tensorboard import SummaryWriter
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from datetime import datetime


# IMPORTANT: SOME KAGGLE DATA SOURCES ARE PRIVATE
# RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES.
# import kagglehub
# kagglehub.login()


# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

# recodai_luc_scientific_image_forgery_detection_path = kagglehub.competition_download('recodai-luc-scientific-image-forgery-detection')


def print_hyperparams():
    """Print all major hyperparameter and directory settings in a clean table format."""
    import torch, platform, cv2, os

    print("=" * 60)
    print("🔧 TRAINING CONFIGURATION SUMMARY")
    print("=" * 60)

    # === PATHS ===
    print("\n📁 PATH SETTINGS")
    print(f"COMP_DIR:           {COMP_DIR}")
    print(f"TRAIN_DIR:          {TRAIN_DIR}")
    print(f"TEST_DIR:           {TEST_DIR}")
    print(f"MODEL_PATH:         {MODEL_PATH}")
    print(f"OUT_DIR:            {OUT_DIR}")
    print(f"SUB_PATH:           {SUB_PATH}")

    # === INTERPOLATION ===
    inter_map = {
        cv2.INTER_NEAREST: "INTER_NEAREST",
        cv2.INTER_LINEAR: "INTER_LINEAR",
        cv2.INTER_CUBIC: "INTER_CUBIC",
        cv2.INTER_AREA: "INTER_AREA",
        cv2.INTER_LANCZOS4: "INTER_LANCZOS4"
    }
    print("\n🧩 INTERPOLATION METHOD")
    print(f"INTERPOLATION:      {inter_map.get(INTERPOLATION, INTERPOLATION)}")

    # === TRAINING HYPERPARAMETERS ===
    print("\n⚙️ LEARNING HYPERPARAMETERS")
    print(f"IMG_SIZE:           {IMG_SIZE}")
    print(f"BATCH_SIZE:         {BATCH_SIZE}")
    print(f"NUM_EPOCHS:         {NUM_EPOCHS}")
    print(f"LEARNING RATE:      {LR}")
    print(f"LOSS SCALER:        {loss_scaler}")
    print(f"α (cls loss):       {alpha}")
    print(f"β (dice loss):      {beta}")
    print(f"γ (img loss):       {gamma}")
    print(f"POS_WEIGHT:         {POS_W}")
    print(f"Optimizer:          {optimizer_cls.__name__}")
    print(f"Scheduler:          {scheduler_cls.__name__}")
    print(f"Scheduler Params:   {scheduler_params}")
    print(f"Freeze Epoch:       {FREEZE_EPOCH}")
    print(f"Freeze Layer:       {FREEZE_LAYER}")
    print(f"New LR Ratio:       {NEW_LR_RATIO}")


    # === LOSS FUNCTIONS ===
    print("\n📉 LOSS SETTINGS")
    print(f"CLS LOSS:           {cls_loss.__class__.__name__}")
    print(f"DICE LOSS:          {dice_loss.__class__.__name__}")

    # === POSTPROCESSING ===
    print("\n🧠 POST-PROCESSING SETTINGS")
    print(f"THRESHOLD:          {THRESHOLD}")
    print(f"LOW_CONF_MAX_PROB:  {LOW_CONF_MAX_PROB}")
    print(f"LOW_VIZ_THR:        {LOW_VIZ_THR}")
    print(f"LOW_CONF_MIN_PIXEL: {LOW_CONF_MIN_PIXEL}")
    print(f"MIN_AREA_RATIO:     {MIN_AREA_RATIO}")

    # === HARDWARE SETTINGS ===
    print("\n💻 ACCELERATION SETTINGS")
    print(f"USE_PIN_MEM:        {USE_PIN_MEM}")
    print(f"NUM_WORKERS:        {NUM_WORKERS}")
    print(f"USE_AMP:            {USE_AMP}")
    print(f"CUDA AVAILABLE:     {torch.cuda.is_available()}")
    print(f"PLATFORM:           {platform.platform()}")

    print("=" * 60)
    print("✅ Hyperparameter summary complete.")
    print("=" * 60)
    print()


def cross_val_score(
        model_cls,
        k:int=5,
        dataset:HybridDataset=None,
        loss_scaler:int=8,
        random_state:int=42,
        ues_pin_memory:bool=False,
        num_workers:int=0,
        device=None,
        batch_size=16,
        lr=1e-3,
        cls_loss=nn.BCEWithLogitsLoss,
        dice_loss=smp.losses.DiceLoss,
        alpha=0.5,
        beta=0.5,
        gamma=0.3,
        epoch=10,
        freeze_epoch = 15,
        freeze_layer = None,
        after_freeze_lr = None,
        interpolation: int = cv2.INTER_NEAREST, 
        threshold: float = 0.5,
        min_area_ratio: float = 0.02, 
        low_conf_max_prob: float = 0.06, 
        low_viz_thr: float = 0.04, 
        low_conf_min_pixel: int = 128,
        optimizer_cls=torch.optim.Adam,
        scheduler_cls=None,
        scheduler_params=None,
        use_amp=False,
        use_log=False,
        run_name=None,
        **kwargs,
    ):

    if optimizer_cls is None:
        raise ValueError("optimizer must be provided")
    kfold = KFold(n_splits=k, shuffle=True, random_state=random_state)
    log = {}
    writer = None

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\nFOLD [{fold + 1}/{k}]...")
        if use_log:
            log_dir = os.path.join(f"runs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}", f"FOLD_{fold+1}") if run_name is None else os.path.join(f"runs/{run_name}", f"FOLD_{fold+1}")
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=log_dir)
        fold_start_time = time()

        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)

        train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=ues_pin_memory)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=ues_pin_memory)

        #후에 제거
        torch.autograd.set_detect_anomaly(True)

        total_steps = len(train_loader) * epoch
        scheduler_params["num_warmup_steps"] = int(total_steps * 0.1)
        scheduler_params["num_training_steps"] = total_steps

        model = model_cls(**kwargs)
        
        #TODO 후에 하드코딩아닌 함수인자로 받기
        try:
            optimizer = optimizer_cls([
                {"params": model.encoder.parameters(), "lr": lr*0.5},   # 천천히 미세조정
                {"params": model.decoder.parameters(), "lr": lr},   # 크게 학습
                {"params": model.segmentation_head.parameters(), "lr": lr},  # 크게 학습
                {"params": model.classification_head.parameters(), "lr": lr}
            ], weight_decay=1e-3)
        except:
            optimizer = optimizer_cls(model.parameters(), lr=lr, weight_decay=1e-3)

        scheduler = scheduler_cls(optimizer, **scheduler_params) if scheduler_cls else None
        if scheduler_cls is not None: scheduler.last_epoch = -1; scheduler.step()

        model = model.to(device)
        log[f'fold{fold + 1}'] = train(
            model, train_loader, val_loader, optimizer, epoch,
            device, cls_loss, dice_loss, alpha, beta, gamma, loss_scaler, scheduler,
            interpolation=interpolation, threshold=threshold,
            min_area_ratio=min_area_ratio, low_conf_max_prob=low_conf_max_prob, 
            low_viz_thr=low_viz_thr, low_conf_min_pixel=low_conf_min_pixel,
            fold=fold+1, use_amp=use_amp, use_log=use_log, writer=writer,
            freeze_epoch=freeze_epoch, freeze_layer=freeze_layer, after_freeze_lr=after_freeze_lr
        )
        
        score = evaluate(
            model, val_loader, device, cls_loss, dice_loss, alpha, beta, gamma, loss_scaler,
            interpolation=interpolation, threshold=threshold,
            min_area_ratio=min_area_ratio, low_conf_max_prob=low_conf_max_prob,     
            low_viz_thr=low_viz_thr, low_conf_min_pixel=low_conf_min_pixel
        )
        fold_end_time = time()
        elapsed_time = fold_end_time - fold_start_time
        print(f"Final Score F1: {score} TOTAL LOSS: {score['loss_total']:.4f} CLS LOSS: {score['loss_cls']:.4f} DICE LOSS: {score['loss_dice']:.4f} IS FORGED IMG LOSS: {score['loss_img']:.4f}")
        print(f"FOLD{fold + 1} TIME: {timedelta(seconds=elapsed_time)}")



    return log

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
    NUM_EPOCHS = 40
    NEW_LR_RATIO = None
    FREEZE_EPOCH = 10
    FREEZE_LAYER = None
    LR = 1e-3
    # POS_W = torch.tensor(1) # Resized
    MODEL_CLS = SMPUnetWithNorm
    POS_W = torch.tensor(32) # Original
    cls_loss = nn.BCEWithLogitsLoss(weight=POS_W)

    """
    cls_loss = FocalLoss(
        mode='binary',
        alpha=0.8,        # 0.9 → 0.7 : forged 픽셀(양성) 가중치 완화 → FN penalty 약화
        gamma=2.0,        # 2.0 → 1.5 : overly hard focusing 완화 → 확률 분포 부드럽게
        ignore_index=None,
        normalized=False,
        reduction='mean'
    )
    """
    
    

    # 2️⃣ FN penalty 완화된 FocalTverskyLoss (mask overlap 중심)
    dice_loss = FocalTverskyLoss(
        mode="binary",
        alpha=0.35,        # 0.3 → 0.4 : FP penalty 강화 → forged mask를 더 타이트하게
        beta=0.8,         # 0.8 → 0.7 : FN penalty 완화 → forged 영역 덜 탐색
        gamma=0.85,       # 0.9 → 0.85 : focusing 완화 → 안정성 증가
        log_loss=False,
        from_logits=True,
        smooth=1e-7,
    )
    """
    dice_loss = smp.losses.TverskyLoss(
        mode='binary',       # binary / multiclass / multilabel
        log_loss=False,      # 로그적용 여부 (보통 False 유지)
        alpha=0.3,           # FP penalty 
        beta=0.8,            # FN penalty 
        from_logits=True,
        smooth=1e-7,
    )
    """

    alpha = 0.4  # Weight for combining BCE and Dice losses
    beta = 1 - alpha
    gamma = 0.5
    loss_scaler = 0.1
    optimizer_cls = torch.optim.AdamW

    scheduler_cls = get_cosine_schedule_with_warmup
    scheduler_params = {
        "num_warmup_steps": None,   # 아래에서 계산됨
        "num_training_steps": None, # 아래에서 계산됨
    }



    #Post Processing HyperParams
    THRESHOLD = 0.3
    LOW_CONF_MAX_PROB = 0.06
    LOW_VIZ_THR = 0.04
    LOW_CONF_MIN_PIXEL = 128
    MIN_AREA_RATIO = 0.001
    THRESHOLD = 0.5
    TRAIN_SAMPLE_NUM = None
    TEST_SAMPLE_NUM = None


    #Aceleration Params -> Default: cpu settings
    USE_PIN_MEM = torch.cuda.is_available() and "Windows" not in platform.platform()
    NUM_WORKERS = 4 if torch.cuda.is_available() and "Windows" not in platform.platform() else 0
    USE_AMP = True

    #ACK
    USE_LOG = True


    print_hyperparams()

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
        MODEL_CLS, k=5, dataset=full_ds, device=device, batch_size=BATCH_SIZE,
        ues_pin_memory=USE_PIN_MEM, num_workers=NUM_WORKERS, pos_w=POS_W,
        cls_loss=cls_loss, dice_loss=dice_loss, loss_scaler=loss_scaler,
        alpha=alpha, beta=beta, gamma=gamma, epoch=NUM_EPOCHS, interpolation=INTERPOLATION,
        threshold=THRESHOLD, min_area_ratio=MIN_AREA_RATIO, low_conf_max_prob=LOW_CONF_MAX_PROB,
        low_viz_thr=LOW_VIZ_THR, low_conf_min_pixel=LOW_CONF_MIN_PIXEL, lr=LR, use_log=USE_LOG,
        optimizer_cls=optimizer_cls, scheduler_cls=get_cosine_schedule_with_warmup, scheduler_params=scheduler_params, use_amp=USE_AMP,
        freeze_epoch=FREEZE_EPOCH, freeze_layer=FREEZE_LAYER, after_freeze_lr=NEW_LR_RATIO, 

        encoder_name="efficientnet-b3", encoder_weights="imagenet", #모델 파라미터
        in_channels=3, classes=1, activation=None, aux_params={
            "classes": 1,           # 출력 클래스 개수
            "pooling": "avg",       # global avg pooling
            "dropout": 0.3,
            "activation": None, # optional
        }) 

    print(f"CV is complete! Total Time: {timedelta(seconds=time() - cv_start_time)}")

    import pickle as pkl

    with open("train_log.pkl", "wb") as f:
        pkl.dump(train_log, f)
        print("\nLog save complete! \n")