import os

from Utils.Augmentation import get_train_transform
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
# ===== Warning Filtering =====
# Triton / xFormers 관련 워닝 숨기기 (성능 최적화 관련, 정확도에는 영향 없음)
warnings.filterwarnings(
    "ignore",
    message="A matching Triton is not available, some optimizations will not be enabled"
)
warnings.filterwarnings(
    "ignore",
    module="xformers"
)

# lr_scheduler.step() 호출 순서 관련 워닝 숨기기
warnings.filterwarnings(
    "ignore",
    message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`"
)

import torch
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader, Subset
from Utils.Dataset import HybridDataset, HybridCropDataset
from Utils.utils import compute_pos_weight, train, evaluate, set_seed
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import FocalLoss, LovaszLoss
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datetime import timedelta
from time import time
from transformers import get_cosine_schedule_with_warmup

from Utils.Model import DINOv2SegmentationModel, SwinTransformerSegmentationModel as Swin
from Utils.Model import SwinDinoEnsembleModel, SMPWithSelfCorr
from Utils.Loss import FocalTverskyLoss
from Utils.scheduler import cosine_with_min_lr
from torch.utils.tensorboard import SummaryWriter
from segmentation_models_pytorch.losses.soft_bce import SoftBCEWithLogitsLoss
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
    print(f"Run Name:           {RUN_NAME}")
    print(f"New α (cls loss):   {new_alpha}")
    print(f"New β (dice loss):  {new_beta}")
    print(f"New γ (img loss):   {new_gamma}")
    print(f"Pos W Ratio:        {POS_W_RATIO}")
    print(f"POS_W:              {POS_W}")


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
        train_transform=None,
        test_transform=None,
        use_amp=False,
        use_log=False,
        run_name=None,
        new_alpha=0.2,
        new_beta=0.8,
        new_gamma=0.0,
        use_t = False,
        warmup_ratio=0.3,
        pad_mode='reflect',
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
        val_ds.dataset.transform = test_transform 

        train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=ues_pin_memory)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=ues_pin_memory)

        #후에 제거
        # torch.autograd.set_detect_anomaly(True)

        total_steps = len(train_loader) * epoch
        if scheduler_cls == get_cosine_schedule_with_warmup:
            scheduler_params["num_warmup_steps"] = int(total_steps * warmup_ratio)
            scheduler_params["num_training_steps"] = total_steps
        elif scheduler_cls == cosine_with_min_lr:
            scheduler_params["warmup_steps"] = int(total_steps * warmup_ratio)
            scheduler_params["total_steps"] = total_steps
        else:
            raise ValueError("scheduler must be either get_cosine_schedule_with_warmup or cosine_with_min_lr")

        model = model_cls(**kwargs)
        if model_cls.__name__ == 'swinv2_base_22k_500k':
            
            ckpt = torch.load("swinv2_base_22k_500k.pth", map_location=device)
            sd = ckpt.get("model", ckpt)

            try:
                model.encoder.load_state_dict(sd, strict=True)
                print("✔ strict=True 성공 (완벽 매칭)")
            except RuntimeError:
                msg = model.encoder.load_state_dict(sd, strict=False)
                print("⚠ strict=False fallback")
                print("missing:", len(msg.missing_keys))
                print("unexpected:", len(msg.unexpected_keys))
        
        if fold == 0: pass # print(model)
        
        #TODO 후에 하드코딩아닌 함수인자로 받기
        # --- [모델 & 최적화 준비] ---
        # ... (model, optimizer_cls, lr 정의는 그대로) ...

        # 💡 모델 구조에 따라 파라미터 그룹 동적 설정
        if hasattr(model, 'encoder') or hasattr(model, 'backbone'):
            # A. SMP (Unet/DeepLab) 구조일 때
            if hasattr(model, 'encoder'):
                params = model.encoder.parameters()
            elif hasattr(model, 'backbone'): 
                params = model.backbone.parameters()
            
            if hasattr(model, 'decoder'):
                pass
            elif hasattr(model, 'segmentation_head'):
                params_group.append({"params": model.segmentation_head.parameters(), "lr": lr})
            elif hasattr(model, 'classification_head'):
                params_group.append({"params": model.classification_head.parameters(), "lr": lr})

            params_group = [
                {"params": params, "lr": lr*0.2},
                {"params": model.decoder.parameters(), "lr": lr},
            ]
            

        elif hasattr(model, 'backbone') and hasattr(model, 'decoder_low'):
            # B. DINOv2 (ViT) 기반 커스텀 구조일 때 (사용자님의 현재 모델)
            decoder_params = list(model.decoder_low.parameters()) + list(model.decoder_high.parameters())
            params_group = [
                {"params": model.backbone.parameters(), "lr": lr * 0.3}, # ViT는 훨씬 느리게 학습 (0.1)
                {"params": decoder_params, "lr": lr},                    # 디코더는 빠르게 학습
            ]

        elif model_cls.__name__ == "SwinDinoEnsembleModel":
            # C. Swin + DINOv2 앙상블 구조일 때
            swin_params = list(model.swin_model.backbone.parameters())
            decoder_params = list(model.swin_model.decoder.parameters())
            params_group = [
                {"params": swin_params, "lr": lr * 0.3},   # Swin은 매우 느리게 학습 (0.05)
                {"params": decoder_params, "lr": lr},       # 디코더는 빠르게 학습
            ]
        elif model_cls.__name__ == "SMPWithSelfCorr":
            # D. Unet with Self-Correlation 구조일 때
            backbone_params = model.smp.encoder.parameters()
            decoder_params = model.smp.decoder.parameters() if hasattr(model.smp, 'decoder') else None
            params_group = [
                {"params": backbone_params, "lr": lr * 0.3},  # 백본은 느리게 학습
                {"params": decoder_params, "lr": lr},         # 디코더는 빠르게 학습
            ]
        else:
            # C. 기타 구조이거나 파라미터 그룹 분리가 불필요할 때
            params_group = model.parameters()


        # 최종 Optimizer 정의
        optimizer = optimizer_cls(params_group, weight_decay=1e-4)  

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
            freeze_epoch=freeze_epoch, freeze_layer=freeze_layer, after_freeze_lr=after_freeze_lr,
            new_alpha=new_alpha, new_beta=new_beta, new_gamma=new_gamma, use_t=use_t,
            train_transform=train_transform, test_transform=test_transform,
            pad_mode=pad_mode, val_epcoch=10
        )
        
        score = evaluate(
            model, val_loader, device, cls_loss, dice_loss, alpha, beta, gamma, loss_scaler,
            interpolation=interpolation, threshold=threshold, pad_mode=pad_mode,
            min_area_ratio=min_area_ratio, low_conf_max_prob=low_conf_max_prob,
            low_viz_thr=low_viz_thr, low_conf_min_pixel=low_conf_min_pixel,
            train_transform=train_transform, test_transform=test_transform,
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

    #Post Processing HyperParams
    THRESHOLD = 0.5
    LOW_CONF_MAX_PROB = 0.06
    LOW_VIZ_THR = 0.04
    LOW_CONF_MIN_PIXEL = 128
    MIN_AREA_RATIO = 0.002
    TRAIN_SAMPLE_NUM = None
    TEST_SAMPLE_NUM = None
    TEMPERATURE = 2

    paths = {
        'train_authentic': os.path.join(TRAIN_DIR, "authentic"),
        'train_forged': os.path.join(TRAIN_DIR, "forged"),
        'train_masks': os.path.join(COMP_DIR, "train_masks"),
        'test_images': TEST_DIR
    }

    # Preprocessing
    IMG_SIZE = 512
    PAD_MODE = 'constant'  # 'constant', 'edge', 'symmetric', 'reflect', 'wrap'

    """
    train_transform = A.Compose([
        # HDF5에는 storage_size(256)로 저장되어 있음 -> 여기서 224로 랜덤 크롭
        A.RandomCrop(IMG_SIZE, IMG_SIZE), 
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    """

    train_transform = get_train_transform(img_size=IMG_SIZE, aug_version="crop")

    test_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    #Learning HyperParams
    full_ds = HybridCropDataset(
        f"train_data_crop_{IMG_SIZE}px.h5",
        paths['train_authentic'],
        paths['train_forged'],
        paths['train_masks'],
        img_size=IMG_SIZE,
        is_train=True,
        preload=True,
        verbose=True,   
        train_transform=train_transform,
        test_transform=test_transform,
        train_sample_num=TRAIN_SAMPLE_NUM,
        test_sample_num=TEST_SAMPLE_NUM,
        pad_mode=PAD_MODE
    )

    BATCH_SIZE = 8
    NUM_EPOCHS = 60
    NEW_LR_RATIO = 1 
    FREEZE_EPOCH = 61
    FREEZE_LAYER = 3
    LR = 1e-3
    WARMUP_RATIO = 0.3
    # POS_W = torch.tensor(1) # Resized
    MODEL_CLS = SMPWithSelfCorr
    base_model_config = {
        'encoder_name': "tu-efficientnet_b4",
        'encoder_weights': 'imagenet',
        'in_channels': 3,
        'classes': 1,
        'activation': None
    }

    POS_W_RATIO = 0.12
    POS_W = compute_pos_weight(dataset=full_ds, h5_path="train_data_dino.h5") * POS_W_RATIO
    cls_loss = SoftBCEWithLogitsLoss(pos_weight=POS_W, smooth_factor=0.1)
    DECODER_TYPE = 'cnn_cbam' # [simple_mlp, unet_style, cnn_se, cnn_cbam] 
    VIT_DIM = 384
    NUM_CLASSES = 1
    USE_T = True
    DECODER_EMB_CH = 256
    DECODER_DROPOUT_RATIO = 0.1

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
    
    
    # TODO 
    # dice_loss = LovaszLoss(mode='binary', per_image=False, ignore_index=None)
    
    dice_loss = FocalTverskyLoss(
        mode="binary",
        alpha=0.35,        # 0.3 → 0.4 : FP penalty 강화 → forged mask를 더 타이트하게
        beta=0.8,         # 0.8 → 0.7 : FN penalty 완화 → forged 영역 덜 탐색
        gamma=0.85,       # 0.9 → 0.85 : focusing 완화 → 안정성 증가
        log_loss=True,
        from_logits=True,
        smooth=1e-5
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

    alpha = 0.7  # Weight for combining BCE and Dice losses
    beta = 1 - alpha
    gamma = 0.5
    loss_scaler = 1
    new_alpha = alpha
    new_beta = beta
    new_gamma = gamma
    """
    new_alpha = 0.1
    new_beta = 1 - new_alpha
    new_gamma = 0.0
    """

    optimizer_cls = torch.optim.AdamW

    """
    scheduler_cls = get_cosine_schedule_with_warmup
    scheduler_params = {
        "num_warmup_steps": None,   # 아래에서 계산됨
        "num_training_steps": None, # 아래에서 계산됨
    }
    """

    scheduler_cls = cosine_with_min_lr
    scheduler_params = {
        "warmup_steps": None,       # 아래에서 계산됨

        "total_steps": None,        # 아래에서 계산됨
        "min_lr": 1e-6,
    }


    #Aceleration Params -> Default: cpu settings
    USE_PIN_MEM = torch.cuda.is_available() and "Windows" not in platform.platform()
    NUM_WORKERS = 4 if torch.cuda.is_available() and "Windows" not in platform.platform() else 0
    USE_AMP = False

    #ACK
    USE_LOG = True
    RUN_NAME = f"effiB6_Encoder_Freeze_FPN_Self_Corr_weak_aug_lr{LR:.0e}_unfreeze_{FREEZE_LAYER}stages"


    print_hyperparams()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    cv_start_time = time()
    train_log = cross_val_score(
        MODEL_CLS, k=5, dataset=full_ds, device=device, batch_size=BATCH_SIZE,
        ues_pin_memory=USE_PIN_MEM, num_workers=NUM_WORKERS, cls_loss=cls_loss, dice_loss=dice_loss, loss_scaler=loss_scaler,
        alpha=alpha, beta=beta, gamma=gamma, epoch=NUM_EPOCHS, interpolation=INTERPOLATION, pad_mode=PAD_MODE,
        threshold=THRESHOLD, min_area_ratio=MIN_AREA_RATIO, low_conf_max_prob=LOW_CONF_MAX_PROB,
        low_viz_thr=LOW_VIZ_THR, low_conf_min_pixel=LOW_CONF_MIN_PIXEL, lr=LR, use_log=USE_LOG, warmup_ratio=WARMUP_RATIO,
        optimizer_cls=optimizer_cls, scheduler_cls=scheduler_cls, scheduler_params=scheduler_params, use_amp=USE_AMP,
        freeze_epoch=FREEZE_EPOCH, freeze_layer=FREEZE_LAYER, after_freeze_lr=NEW_LR_RATIO,  run_name=RUN_NAME, use_t=USE_T,
        new_alpha=new_alpha, new_beta=new_beta, new_gamma=new_gamma, train_transform=train_transform, test_transform=test_transform,
        
        base_smp_cls=smp.FPN, corr_level = 3,
        corr_pool = 1,corr_topk = 1, temperature = 1, **base_model_config
            
    )

    '''
    # Self-Correlation UNet
    

    # Swin
    backbone_name='swin_base_patch4_window7_224',
    decoder_type='unet',
    num_classes=1,
    emb_ch=DECODER_EMB_CH,
    dropout_ratio=DECODER_DROPOUT_RATIO,

    #Swin + DINOv2 Ensemble
    # Model Specific Args
    vit_dim=384,
    dino_backbone_name='dinov2_vits14',
    swin_backbone_name='swin_base_patch4_window7_224',
    decoder_type='unet',
    num_classes=1,
    emb_ch=256,
    ca_emb_ch=256,
    dropout_ratio=0.1

    SMP
    , aux_params={
            "classes": 1,           # 출력 클래스 개수
            "pooling": "avg",       # global avg pooling
            "dropout": 0.3,
            "activation": None, # optional
        }
    , decoder_type=DECODER_TYPE, vit_dim=VIT_DIM, num_classes=1, img_size=IMG_SIZE
    '''


    print(f"CV is complete! Total Time: {timedelta(seconds=time() - cv_start_time)}")

    import pickle as pkl

    with open("train_log.pkl", "wb") as f:
        pkl.dump(train_log, f)
        print("\nLog save complete! \n")