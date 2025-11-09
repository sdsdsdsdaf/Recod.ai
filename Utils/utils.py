import json
from time import time
from typing import Optional, Union

import numba
import numpy as np
from numba import types
import numpy.typing as npt
import pandas as pd
import scipy.optimize
import cv2
import torch
import h5py
import os
from tqdm.auto import tqdm
import random
from collections import defaultdict
from torch.amp import autocast, GradScaler
from datetime import timedelta
import gc
from pytorch_toolbelt import losses as L
from torch.utils import tensorboard
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch

class ParticipantVisibleError(Exception):
    pass


def str_to_device(device:Union[torch.device,str]) -> torch.device:
    if not (isinstance(device, torch.device) or isinstance(device, str)):
        raise ValueError("device must torch.device or str")
    elif isinstance(device, str):
        device = torch.device(device)

    return device

def preprocessing(img:np.ndarray, img_size:int, interpolation=cv2.INTER_NEAREST, div=255.0):
    """
    Return Normalize and resize img
    """
    return cv2.resize(img.astype(np.float32) / div, (img_size, img_size), interpolation=interpolation)
    
"""
def postprocessing(proba_map:np.ndarray, original_size:tuple[int, int], threshold:float, low_conf_max_prob:float, low_viz_thr:float, low_conf_min_pixel:int):
    # DEBUG
    # print("[DEBUG] Forged pixel num before PostProcessing: ", (proba_map > threshold).astype(np.uint8).sum())
    if is_low_confidence(proba_map, low_conf_max_prob, low_viz_thr, low_conf_min_pixel):
        # print("[DEBUG]: low_confidence")
        proba_map = np.zeros_like(proba_map)

    #후에 bilinear사용 고려
    proba_map = cv2.resize(proba_map, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST) # (W, H)
    mask_pred = (proba_map > threshold).astype(np.uint8)

    #PostProcessing
    kernel = np.ones((3,3), np.uint8)
    mask_pred = cv2.morphologyEx(mask_pred, cv2.MORPH_OPEN, kernel)
    mask_pred = cv2.morphologyEx(mask_pred, cv2.MORPH_CLOSE, kernel)

    return mask_pred
"""

def postprocessing(proba_map: np.ndarray,
                   original_size: tuple[int, int],
                   threshold: float,
                   low_conf_max_prob: float,
                   low_viz_thr: float,
                   low_conf_min_pixel: int):

    # 1) Low confidence filtering stays the same
    if is_low_confidence(proba_map, low_conf_max_prob, low_viz_thr, low_conf_min_pixel):
        return np.zeros(original_size, dtype=np.uint8)

    # 2) Threshold FIRST (in model resolution → cheap)
    mask = (proba_map > threshold).astype(np.uint8)

    # 3) Noise cleanup (in small resolution → super cheap)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 4) Now resize to original resolution (only once)
    mask = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST) # (W, H)


    return mask

@numba.jit(nopython=True)
def _rle_encode_jit(x: npt.NDArray, fg_val: int = 1) -> list[int]:
    """Numba-jitted RLE encoder."""
    dots = np.where(x.T.flatten() == fg_val)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

'''
def rle_encode(masks: list[npt.NDArray], fg_val: int = 1) -> str:
    """
    Adapted from contrails RLE https://www.kaggle.com/code/inversion/contrails-rle-submission
    Args:
        masks: list of numpy array of shape (height, width), 1 - mask, 0 - background
    Returns: run length encodings as a string, with each RLE JSON-encoded and separated by a semicolon.
    """
    return ';'.join([json.dumps(_rle_encode_jit(x, fg_val)) for x in masks])
'''

def rle_encode(masks: list[np.ndarray], fg_val: int = 1) -> str:
    encoded_strings = []
    for x in masks:
        rl = _rle_encode_jit(x, fg_val)  # ✅ 공식 함수 그대로
        encoded_strings.append("[" + ", ".join(str(v) for v in rl) + "]")  # ✅ json과 동일 포맷
    return ";".join(encoded_strings)    

@numba.njit
def _rle_decode_jit(mask_rle: npt.NDArray, height: int, width: int) -> npt.NDArray:
    """
    s: numpy array of run-length encoding pairs (start, length)
    shape: (height, width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    if len(mask_rle) % 2 != 0:
        # Numba requires raising a standard exception.
        raise ValueError('One or more rows has an odd number of values.')

    starts, lengths = mask_rle[0::2], mask_rle[1::2]
    starts -= 1
    ends = starts + lengths
    for i in range(len(starts) - 1):
        if ends[i] > starts[i + 1]:
            raise ValueError('Pixels must not be overlapping.')
    img = np.zeros(height * width, dtype=np.bool_)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img


def rle_decode(mask_rle: str, shape: tuple[int, int]) -> npt.NDArray:
    """
    mask_rle: run-length as string formatted (start length)
              empty predictions need to be encoded with '-'
    shape: (height, width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """

    mask_rle = json.loads(mask_rle)
    mask_rle = np.asarray(mask_rle, dtype=np.int32)
    starts = mask_rle[0::2]
    if sorted(starts) != list(starts):
        raise ParticipantVisibleError('Submitted values must be in ascending order.')
    try:
        return _rle_decode_jit(mask_rle, shape[0], shape[1]).reshape(shape, order='F')
    except ValueError as e:
        raise ParticipantVisibleError(str(e)) from e


def calculate_f1_score(pred_mask: npt.NDArray, gt_mask: npt.NDArray):
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()

    tp = np.sum((pred_flat == 1) & (gt_flat == 1))
    fp = np.sum((pred_flat == 1) & (gt_flat == 0))
    fn = np.sum((pred_flat == 0) & (gt_flat == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    if (precision + recall) > 0:
        return 2 * (precision * recall) / (precision + recall)
    else:
        return 0


def calculate_f1_matrix(pred_masks: list[npt.NDArray], gt_masks: list[npt.NDArray]):
    """
    Parameters:
    pred_masks (np.ndarray):
            First dimension is the number of predicted instances.
            Each instance is a binary mask of shape (height, width).
    gt_masks (np.ndarray):
            First dimension is the number of ground truth instances.
            Each instance is a binary mask of shape (height, width).
    """

    num_instances_pred = len(pred_masks)
    num_instances_gt = len(gt_masks)
    f1_matrix = np.zeros((num_instances_pred, num_instances_gt))

    # Calculate F1 scores for each pair of predicted and ground truth masks
    for i in range(num_instances_pred):
        for j in range(num_instances_gt):
            pred_flat = pred_masks[i].flatten()
            gt_flat = gt_masks[j].flatten()
            f1_matrix[i, j] = calculate_f1_score(pred_mask=pred_flat, gt_mask=gt_flat)

    if f1_matrix.shape[0] < len(gt_masks):
        # Add a row of zeros to the matrix if the number of predicted instances is less than ground truth instances
        f1_matrix = np.vstack((f1_matrix, np.zeros((len(gt_masks) - len(f1_matrix), num_instances_gt))))

    return f1_matrix


def oF1_score(pred_masks: list[npt.NDArray], gt_masks: list[npt.NDArray]):
    """
    Calculate the optimal F1 score for a set of predicted masks against
    ground truth masks which considers the optimal F1 score matching.
    This function uses the Hungarian algorithm to find the optimal assignment
    of predicted masks to ground truth masks based on the F1 score matrix.
    If the number of predicted masks is less than the number of ground truth masks,
    it will add a row of zeros to the F1 score matrix to ensure that the dimensions match.

    Parameters:
    pred_masks (list of np.ndarray): List of predicted binary masks.
    gt_masks (np.ndarray): Array of ground truth binary masks.
    Returns:
    float: Optimal F1 score.
    """
    f1_matrix = calculate_f1_matrix(pred_masks, gt_masks)

    # Find the best matching between predicted and ground truth masks
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(-f1_matrix)
    # The linear_sum_assignment discards excess predictions so we need a separate penalty.
    excess_predictions_penalty = len(gt_masks) / max(len(pred_masks), len(gt_masks))
    return np.mean(f1_matrix[row_ind, col_ind]) * excess_predictions_penalty


def evaluate_single_image(label_rles: str, prediction_rles: str, shape_str: str) -> float:
    shape = json.loads(shape_str)
    label_rles = [rle_decode(x, shape=shape) for x in label_rles.split(';')]
    prediction_rles = [rle_decode(x, shape=shape) for x in prediction_rles.split(';')]
    return oF1_score(prediction_rles, label_rles)


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Args:
        solution (pd.DataFrame): The ground truth DataFrame.
        submission (pd.DataFrame): The submission DataFrame.
        row_id_column_name (str): The name of the column containing row IDs.
    Returns:
        float

    Examples
    --------
    >>> solution = pd.DataFrame({'row_id': [0, 1, 2], 'annotation': ['authentic', 'authentic', 'authentic'], 'shape': ['authentic', 'authentic', 'authentic']})
    >>> submission = pd.DataFrame({'row_id': [0, 1, 2], 'annotation': ['authentic', 'authentic', 'authentic']})
    >>> score(solution.copy(), submission.copy(), row_id_column_name='row_id')
    1.0

    >>> solution = pd.DataFrame({'row_id': [0, 1, 2], 'annotation': ['authentic', 'authentic', 'authentic'], 'shape': ['authentic', 'authentic', 'authentic']})
    >>> submission = pd.DataFrame({'row_id': [0, 1, 2], 'annotation': ['[101, 102]', '[101, 102]', '[101, 102]']})
    >>> score(solution.copy(), submission.copy(), row_id_column_name='row_id')
    0.0

    >>> solution = pd.DataFrame({'row_id': [0, 1, 2], 'annotation': ['[101, 102]', '[101, 102]', '[101, 102]'], 'shape': ['[720, 960]', '[720, 960]', '[720, 960]']})
    >>> submission = pd.DataFrame({'row_id': [0, 1, 2], 'annotation': ['[101, 102]', '[101, 102]', '[101, 102]']})
    >>> score(solution.copy(), submission.copy(), row_id_column_name='row_id')
    1.0

    >>> solution = pd.DataFrame({'row_id': [0, 1, 2], 'annotation': ['[101, 103]', '[101, 102]', '[101, 102]'], 'shape': ['[720, 960]', '[720, 960]', '[720, 960]']})
    >>> submission = pd.DataFrame({'row_id': [0, 1, 2], 'annotation': ['[101, 102]', '[101, 102]', '[101, 102]']})
    >>> score(solution.copy(), submission.copy(), row_id_column_name='row_id')
    0.9983739837398374

    >>> solution = pd.DataFrame({'row_id': [0, 1, 2], 'annotation': ['[101, 102];[300, 100]', '[101, 102]', '[101, 102]'], 'shape': ['[720, 960]', '[720, 960]', '[720, 960]']})
    >>> submission = pd.DataFrame({'row_id': [0, 1, 2], 'annotation': ['[101, 102]', '[101, 102]', '[101, 102]']})
    >>> score(solution.copy(), submission.copy(), row_id_column_name='row_id')
    0.8333333333333334
    """
    df = solution
    df = df.rename(columns={'annotation': 'label'})

    df['prediction'] = submission['annotation']
    # Check for correct 'authentic' label
    authentic_indices = (df['label'] == 'authentic') | (df['prediction'] == 'authentic')
    df['image_score'] = ((df['label'] == df['prediction']) & authentic_indices).astype(float)

    df.loc[~authentic_indices, 'image_score'] = df.loc[~authentic_indices].apply(
        lambda row: evaluate_single_image(row['label'], row['prediction'], row['shape']), axis=1
    )
    return float(np.mean(df['image_score']))


def freeze_encoder_after_epoch(
    model, current_epoch, freeze_at: int, optimizer=None, n_layers: Optional[int] = None, new_lr_ratio: Optional[float] = None,
):
    """
    Freeze encoder parameters after a given epoch.
    Optionally keep the last N stages (layers) unfrozen.

    Args:
        model (torch.nn.Module): smp.Unet model instance
        current_epoch (int): current epoch index
        freeze_at (int): epoch index to start freezing
        optimizer (torch.optim.Optimizer, optional): optimizer to rebuild after freezing
        n_layers (int, optional): number of last encoder stages to keep trainable.
                                 Default=None → freeze entire encoder.
    """
    if current_epoch == freeze_at:
        print(f"🔒 Freezing encoder at epoch {freeze_at}...")

        # encoder 내부 stage 목록 가져오기
        try:
            stages = model.encoder.get_stages()
        except AttributeError:
            # 일부 encoder는 get_stages() 대신 _blocks, features 등 이름 다름
            stages = list(model.encoder.children())

        if n_layers is None or n_layers <= 0:
            # 전체 encoder freeze
            for param in model.encoder.parameters():
                param.requires_grad = False
            print("➡️ Entire encoder frozen.")
        else:
            # 마지막 n_layers만 제외하고 freeze
            num_stages = len(stages)
            freeze_until = max(0, num_stages - n_layers)
            for i, stage in enumerate(stages):
                requires_grad = (i >= freeze_until)  # 뒤쪽 n_layers만 학습
                for param in stage.parameters():
                    param.requires_grad = requires_grad

            print(f"➡️ Encoder frozen except for last {n_layers} stage(s).")

        # optimizer 다시 구성 (필요 시)
        if optimizer is not None:
            defaults = optimizer.defaults.copy()
            if new_lr_ratio is not None:
                defaults["lr"] *= new_lr_ratio
            optimizer 
            optimizer = type(optimizer)(
                filter(lambda p: p.requires_grad, model.parameters()),
                **optimizer.defaults,
            )
            print("🧩 Optimizer reinitialized (encoder params removed).")
        return optimizer

    return optimizer



def compute_pos_weight(
    dataset=None,
    masks_path=None,
    authentic_path=None,
    forgded_path=None,
    h5_path=None,
    img_size=128,
    mode="resized",
    authentic_included=True,
    interpolation=cv2.INTER_NEAREST,
):
    """
    Compute pos_weight for BCE loss based on forged pixel ratio.

    Args:
        dataset: HybridDataset instance (optional)
        masks_path: path to train_masks directory
        h5_path: path to HDF5 file
        img_size: target resize (for 'resized' mode)
        mode: 'resized' or 'original'
        authentic_included: if True, counts authentic (no mask) images as all-zero masks

    Returns:
        torch.Tensor: scalar pos_weight
    """
    forged_pixels = 0
    total_pixels = 0
    processed = 0

    # ✅ Case 1: HDF5 (resized mask 기준)
    if h5_path and os.path.exists(h5_path) and mode == "resized":
        print(f"📊 Loading resized mask stats from HDF5: {h5_path}")
        with h5py.File(h5_path, "r") as h5f:
            masks = h5f["masks"][:]
            forged_pixels = masks.sum()
            total_pixels = np.prod(masks.shape)
            processed = masks.shape[0]
    
    # ✅ Case 2: Raw mask 폴더 기준
    elif masks_path and os.path.exists(masks_path):
        print(f"📊 Scanning {'resized' if mode=='resized' else 'original'} masks in: {masks_path}")
        mask_files = [f"{f.split('.')[0]}.npy" for f in (os.listdir(authentic_path) + os.listdir(forgded_path)) if f.endswith(('.png', '.jpg', '.jpeg', ))]
        print(f"🗂 Found {len(mask_files)} mask files.")
        print("[DEBUG] First 5 mask files:", mask_files[:5])
        for f in tqdm(mask_files, leave=False):
            mask_path = os.path.join(masks_path, f)
            try:
                mask = np.load(mask_path)
                if mask.ndim == 3:
                    mask = mask.max(axis=0) if mask.shape[0] <= 10 else mask.max(axis=-1)
                if mode == "resized":
                    mask = cv2.resize(mask.astype(np.uint8), (img_size, img_size), interpolation=interpolation)
                mask = (mask > 0).astype(np.uint8)
            except Exception:
                # 🚨 mask 파일 깨졌거나 누락된 경우
                mask = np.zeros((img_size, img_size), dtype=np.uint8)

            forged_pixels += mask.sum()
            total_pixels += mask.size
            processed += 1

        # ✅ authentic (mask 없는 이미지)를 0-mask로 추가
        if authentic_included and dataset is not None:
            missing = len(dataset.samples) - processed
            if missing > 0:
                total_pixels += missing * (img_size ** 2)
                print(f"📦 Added {missing} authentic samples as 0-masks")

    else:
        raise ValueError("You must provide either h5_path or masks_path.")

    ratio = forged_pixels / (total_pixels + 1e-8)
    pos_weight = (1 - ratio) / (ratio + 1e-8)

    print(f"\n📈 Forged pixel ratio: {ratio:.8f}")
    print(f"⚖️ pos_weight = {pos_weight:.2f}  (mode={mode}, total={processed} samples)")
    return torch.tensor(pos_weight, dtype=torch.float32)



def mask_to_instances(mask, min_area: int = 1) -> list[np.ndarray]:
    """
    Convert a mask map to individual connected-component masks (instances).
    
    Args:
        mask (np.ndarray): Model output probability map (H, W)
        min_area (int): Minimum pixel area to keep an instance
        
    Returns:
        list[np.ndarray]: List of instance masks (each with shape (H, W))
    """
    if mask.ndim == 3:
        mask = mask.max(axis=0)

    mask  = mask.astype(np.uint8)
    # 2️⃣ 너무 작으면 아예 인스턴스 없음
    if mask.sum() < min_area:
        return []

    # 3️⃣ 연결된 영역 찾기 (Connected Components)
    num, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    instances = []
    for i in range(1, num):  # label=0은 background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            inst = (lbl == i).astype(np.uint8)  # 해당 label 위치만 1로
            instances.append(inst)

    return instances

def is_low_confidence(prob_map, low_conf_max_prob=0.5, low_viz_thr=0.06, low_conf_min_pixel=100):
    if float(prob_map.max()) >= low_conf_max_prob: #일단 “low confidence 아님으로 판단
        return False
    cover = int((prob_map >= low_viz_thr).sum()) #검출됨
    return cover < low_conf_min_pixel


def train(
        model: torch.nn.Module, train_loader, val_loader, optimizer, epoch,
        device, cls_loss_fn, dice_loss_fn, alpha=0.5, beta=0.5, gamma=0.3, loss_scaler=8, scheduler=None,
        interpolation=cv2.INTER_NEAREST, threshold=0.5, min_area_ratio=0.02,
        low_conf_max_prob=0.06, low_viz_thr=0.04, low_conf_min_pixel=128, # PostProcessing Setting,
        fold=None, use_amp=False, use_log=False, freeze_epoch=15, freeze_layer=None, after_freeze_lr=None,
        writer: Optional[SummaryWriter] =None
    ):

    train_loss_log_dict = defaultdict(list)
    val_loss_log_dict = defaultdict(list)
    scaler = None
    best1_f1 = 0.0
    best5_f1 = [0.0, 0.0, 0.0, 0.0, 0.0]
    f1_log_list = []
    log_file = None
    os.makedirs("Weights", exist_ok=True)
    if fold is not None:
        os.makedirs(os.path.join("Weights", f"FOLD{fold}"), exist_ok=True)

    if use_amp:
        scaler = GradScaler(init_scale=64)

    # DEBUG
    print("Scaler is Not None in def Train: ", scaler is not None)
    

    for E in range(1, epoch + 1):
        epoch_start_time = time()
        if freeze_epoch is not None and E == freeze_epoch:
            optimizer = freeze_encoder_after_epoch(model, E, freeze_at=freeze_epoch, optimizer=optimizer, n_layers=freeze_layer, new_lr_ratio=after_freeze_lr)
            scheduler.optimizer = optimizer

        losses = train_one_epoch(
            model=model, 
            train_loader=train_loader,
            epoch=E, 
            optimizer=optimizer, 
            device=device, 
            cls_loss_fn=cls_loss_fn, 
            dice_loss_fn=dice_loss_fn, 
            scheduler=scheduler, 
            alpha=alpha, 
            beta=beta,
            gamma=gamma, 
            loss_scaler=loss_scaler, 
            scaler=scaler,
            log_file=None,
            writer=writer
        )
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[{E}/{epoch}] CLS LOSS: {losses['loss_cls']:.4f} DICE LOSS: {losses['loss_dice']:.4f} IS FORGED IMG LOSS: {losses['loss_img']:.4f} \nSCALING CLS LOSS: {alpha*loss_scaler*losses['loss_cls']:.4f} SCALING DICE LOSS: {beta*losses['loss_dice']:.4f} SCALING IS FORGED IMG LOSS: {gamma*losses['loss_img']:.4f} TRAIN TOTAL LOSS: {losses['loss_total']:.4f} CURRENT LR: {current_lr:.7f}")
        for k, v in losses.items():
                    train_loss_log_dict[k].append(v)
        if val_loader is None:
            continue

        metric_score = evaluate(
            model=model, 
            val_loader=val_loader, 
            device=device, 
            cls_loss_fn=cls_loss_fn, 
            dice_loss_fn=dice_loss_fn, 
            alpha=alpha, 
            beta=beta,
            gamma=gamma, 
            loss_scaler=loss_scaler, 
            interpolation=interpolation, 
            threshold=threshold, 
            min_area_ratio=min_area_ratio, 
            low_conf_max_prob=low_conf_max_prob, 
            low_viz_thr=low_viz_thr, 
            low_conf_min_pixel=low_conf_min_pixel, 
            scaler=scaler
        )
        
        print(f"[{E}/{epoch}] VAL F1: {metric_score['f1_score']:.4f}"
              f" TRAIN TOTAL LOSS: {metric_score['loss_total']:.4f}"
              f" CLS LOSS: {metric_score['loss_cls']:.4f} "
              f" DICE LOSS: {metric_score['loss_dice']:.4f} "
              f" IS FORGED IMG LOSS: {metric_score['loss_img']:.4f} \n"
              f" SCALING CLS LOSS: {alpha * loss_scaler * metric_score['loss_cls']:.4f} "
              f" SCALING DICE LOSS: {beta * metric_score['loss_dice']:.4f} "
              f" SCALING IS FORGED IMG LOSS: {gamma * metric_score['loss_img']:.4f} "
            )
        
    
        if writer is not None:
            # --- Train losses ---
            writer.add_scalar("Train/Loss_CLS", losses["loss_cls"], E)
            writer.add_scalar("Train/Loss_DICE", losses["loss_dice"], E)
            writer.add_scalar("Train/Loss_IMG", losses["loss_img"], E)
            writer.add_scalar("Train/Loss_TOTAL", losses["loss_total"], E)

            # --- Validation losses ---
            writer.add_scalar("Val/Loss_CLS", metric_score["loss_cls"], E)
            writer.add_scalar("Val/Loss_DICE", metric_score["loss_dice"], E)
            writer.add_scalar("Val/Loss_IMG", metric_score["loss_img"], E)
            writer.add_scalar("Val/Loss_TOTAL", metric_score["loss_total"], E)

            # --- Validation F1 ---
            writer.add_scalar("Val/F1_Score", metric_score["f1_score"], E)


        # ======= 📸 TENSORBOARD IMAGE LOGGING =======
        if writer is not None and (E % 5 == 1 or E == epoch):
            model.eval()
            with torch.no_grad():
                zero_imgs_list, zero_masks_list = [], []
                nonzero_imgs_list, nonzero_masks_list = [], []

                # ── ① Validation 전체 순회하며 샘플 선택 ──
                for imgs, masks, _, _ in val_loader:
                    # 각 샘플별로 GT 존재 여부 계산
                    gt_sums = masks.view(masks.shape[0], -1).sum(dim=1)

                    # GT=0인 샘플과 GT>0인 샘플 분리
                    for i in range(len(gt_sums)):
                        if gt_sums[i] == 0 and len(zero_imgs_list) < 2:
                            zero_imgs_list.append(imgs[i])
                            zero_masks_list.append(masks[i])
                        elif gt_sums[i] > 0 and len(nonzero_imgs_list) < 2:
                            nonzero_imgs_list.append(imgs[i])
                            nonzero_masks_list.append(masks[i])

                        # 둘 다 2개씩 모이면 끝
                        if len(zero_imgs_list) >= 2 and len(nonzero_imgs_list) >= 2:
                            break
                    if len(zero_imgs_list) >= 2 and len(nonzero_imgs_list) >= 2:
                        break

                # ── ② 선택된 샘플 합치기 ──
                if len(zero_imgs_list) == 0 or len(nonzero_imgs_list) == 0:
                    print("⚠️ 샘플을 충분히 찾지 못했습니다.")
                    return

                imgs = torch.stack(zero_imgs_list + nonzero_imgs_list).to(device)
                masks = torch.stack(zero_masks_list + nonzero_masks_list).to(device)

                # ── ③ 모델 예측 ──
                preds, _ = model(imgs)
                preds = torch.sigmoid(preds)
                preds_thr = (preds > threshold).float()

                # ── ④ Normalize ──
                imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min() + 1e-8)
                masks = (masks - masks.min()) / (masks.max() - masks.min() + 1e-8)
                preds = (preds - preds.min()) / (preds.max() - preds.min() + 1e-8)

                # ── ⑤ RGB 변환 ──
                gt_rgb = masks.repeat(1, 3, 1, 1)
                pred_rgb = preds.repeat(1, 3, 1, 1)
                pred_thr_rgb = preds_thr.repeat(1, 3, 1, 1)

                # ── ⑥ 행(row)별 시각화 구성 ──
                rows = []
                for i in range(imgs.shape[0]):  # 총 4개 샘플
                    row = torch.cat([imgs[i], gt_rgb[i], pred_rgb[i], pred_thr_rgb[i]], dim=2)
                    rows.append(row)

                # ── ⑦ 세로로 합치기 (4행 × 4열 구조 완성) ──
                grid = torch.cat(rows, dim=1)  # height 방향으로 stack
                grid = grid.clamp(0, 1).to(torch.float32)

                # ── ⑧ TensorBoard에 기록 ──
                writer.add_image(f"Samples/FOLD{fold}/epoch_{E}", grid, E)

                
            for k, v in metric_score.items():
                val_loss_log_dict[k].append(v)
            if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metric_score['f1_score'])

            f1_log_list.append(metric_score['f1_score'])
            if metric_score['f1_score'] > best1_f1:
                print(f"🏆 Best1 F1: {metric_score['f1_score']:.4f} before {best1_f1:.4f}")
                best1_f1 = metric_score['f1_score']
            
            if metric_score['f1_score'] > min(best5_f1):
                best5_f1.remove(min(best5_f1))
                best5_f1.append(metric_score['f1_score'])
                sorted_best = sorted(best5_f1, reverse=True)
                rank = sorted_best.index(metric_score['f1_score']) + 1

                if fold is not None:
                    save_path = os.path.join("Weights", f"FOLD{fold}", f"best{rank}.pth")
                else:
                    save_path = os.path.join("Weights", f"best{rank}.pth")
                
                try:
                    torch.save(model.state_dict(), save_path)
                    print(f"🔥 F1 {metric_score['f1_score']:.4f} entered Top5 → Rank {rank}/5")
                    print(f"💾 Saved checkpoint: {save_path}")
                except Exception as e:
                    print(f"⚠️ Failed to save model at {save_path}: {e}")
            epoch_end_time = time()
            print(f"[EPOCH {E}/{epoch}] TIME : {timedelta(seconds=epoch_end_time - epoch_start_time)}")

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        try:
            torch.cuda.memory._free_cached_blocks()  # PyTorch 2.6에서 동작
        except Exception:
            pass
 
    return {'train_loss': train_loss_log_dict, 'val_loss': val_loss_log_dict, 'f1_score': f1_log_list}



@torch.no_grad()
def evaluate(
        model, val_loader, device:Union[torch.device,str], cls_loss_fn, dice_loss_fn, alpha=0.5, beta=0.5, gamma=0.3, loss_scaler=8,
        interpolation=cv2.INTER_NEAREST, threshold=0.5, min_area_ratio=0.002,
        low_conf_max_prob=0.06, low_viz_thr=0.04, low_conf_min_pixel=128, scaler=None,
    ):
    model.eval()
    total_loss = 0
    cls_total = 0
    dice_total = 0
    img_total = 0
    solution = {'case_id': [], 'annotation': [], 'shape': []}
    file_path_list = []
    is_forged_img_loss_fn = torch.nn.BCEWithLogitsLoss()
    
    for imgs, masks, mask_path, is_forged in tqdm(val_loader, desc="Calculating Loss", leave=False):

        imgs, masks, is_forged = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True), is_forged.to(device, non_blocking=True)
        masks = masks.squeeze()
        with autocast(device_type=device.type, enabled=scaler is not None):
            if getattr(model, 'classification_head', None) is not None:
                logit_map, is_forged_logit = model(imgs)
                logit_map = logit_map.squeeze()
                is_forged_logit = is_forged_logit.squeeze()
                

            else:
                logit_map = model(imgs)

        logit_map, masks = logit_map.float(), masks.float()

        if getattr(model, 'classification_head', None) is not None:
            is_forged_logit, is_forged = is_forged_logit.float(), is_forged.float()

            is_forged_logit = torch.atleast_1d(is_forged_logit)
            is_forged = torch.atleast_1d(is_forged)

            cls_loss = cls_loss_fn(logit_map, masks)
            dice_loss = dice_loss_fn(logit_map, masks)
            img_loss = is_forged_img_loss_fn(is_forged_logit, is_forged)
            loss = alpha * cls_loss*loss_scaler + beta * dice_loss + img_loss*gamma

        total_loss += loss.item()
        cls_total += cls_loss.item()
        dice_total += dice_loss.item()
        img_total += img_loss.item() if hasattr(model, 'classification_head') else 0



        for img, mask_path, is_forged in zip(imgs, mask_path, is_forged):

            img_path = mask_path2img_path(mask_path, is_forged)
            file_path_list.append(img_path)
            if is_forged:
                gt = np.load(mask_path)
            else:
                img = cv2.imread(img_path)
                w, h = img.shape[:2]
                gt = np.zeros((1, h, w))

            if gt.ndim == 3:
                gt = gt.max(axis=0)
            gt_instances = mask_to_instances(gt)

            case_id = os.path.splitext(os.path.basename(img_path))[0]
            solution['case_id'].append(int(case_id))
            rle_str = rle_encode(gt_instances)
            solution['annotation'].append("authentic" if len(gt_instances) == 0 else rle_str)
            solution['shape'].append(json.dumps([h,w]))

        
    total_loss = total_loss / len(val_loader)
    cls_total = cls_total / len(val_loader)
    dice_total = dice_total / len(val_loader)
    img_total = img_total / len(val_loader)


    prediction= predict(
        model, None, device, test_path_file_list=file_path_list, img_size=img.shape[1],
        max_size=None, interpolation=interpolation, threshold=threshold, min_area_ratio=min_area_ratio,
        low_conf_max_prob=low_conf_max_prob, low_viz_thr=low_viz_thr, low_conf_min_pixel=low_conf_min_pixel,
    )
    del img, imgs, masks, logit_map, cls_loss, dice_loss, loss
    torch.cuda.empty_cache()

    solution['case_id'] = solution['case_id']
    prediction['case_id'] = prediction['case_id']

    if prediction['case_id'] != solution['case_id']:
        print(prediction['case_id'])
        print(solution['case_id'])
        raise ValueError("Prediction`s Key and Solution`s key must be match")
    
    solution_df = pd.DataFrame(solution)
    prediction_df = pd.DataFrame(prediction)

    score_ = score(solution_df, prediction_df, row_id_column_name='case_id')

    return {"f1_score": score_, "loss_total": total_loss, "loss_cls": cls_total, "loss_dice": dice_total, "loss_img": img_total}

    



def mask_path2img_path(mask_path, is_forged):
    """
    Convert a mask path to its corresponding image path.
    If is_forged == 1 → train_images/forged/
       is_forged == 0 → train_images/authentic/

    Args:
        mask_path (str): e.g., "C:\\...\\train_masks\\10.npy"
        is_forged (int): 1 (forged) or 0 (authentic)

    Returns:
        str: full path to the matching image file
    """
    #  base directory split
    mask_dir, mask_file = os.path.split(mask_path)  # (폴더, 파일)
    parent_dir = os.path.dirname(mask_dir)          # train_masks 상위 폴더 (e.g., .../recodai-luc-scientific-image-forgery-detection)
    
    # train_images 하위 폴더 이름 결정
    subfolder = "forged" if is_forged else "authentic"

    # 확장자 없는 파일명 (예: '10')
    file_stem = os.path.splitext(mask_file)[0]

    # 이미지 경로 후보 생성
    img_dir = os.path.join(parent_dir, "train_images", subfolder)
    candidates = [os.path.join(img_dir, file_stem + ext) for ext in [".jpg", ".jpeg", ".png"]]

    # 존재하는 파일 찾기
    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"No corresponding image found for mask: {mask_path}")
        




def train_one_epoch(model, epoch, train_loader, optimizer, device:Union[torch.device,str], cls_loss_fn, dice_loss_fn, scheduler, alpha=0.5, beta=0.5, gamma=0.3, loss_scaler=1.0, scaler: Optional[torch.amp.GradScaler] = None, log_file=None, writer:SummaryWriter=None) -> dict[str, float]:

    model.train()
    total_loss = 0
    bce_total = 0
    dice_total = 0
    is_forged_img_loss = 0
    activation_module = getattr(model.classification_head[-1], 'activation', None)

    if activation_module is None:
        is_forged_img_loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        is_forged_img_loss_fn = torch.nn.BCELoss()
        
    device = str_to_device(device)
    device_type = device.type

    for step, (imgs, masks, path, is_forged) in enumerate(tqdm(train_loader, desc="Training", leave=False)):

        imgs, masks, is_forged = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True), is_forged.to(device, non_blocking=True)
    
        if not torch.isfinite(imgs).all():
            print(f"[STEP {step}] imgs NaN")
            break
        if not torch.isfinite(masks).all():
            print(f"[STEP {step}] masks NaN")
            break
        if not torch.isfinite(is_forged).all():
            print(f"[STEP {step}] is_forged NaN")
            break
        #후에 resize하는 과정도 train할 지 결정
        optimizer.zero_grad()

        # DEBUG
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"❌ {name} contains NaN/Inf")

        with autocast(device_type=device_type, enabled=scaler is not None):
            if getattr(model, 'classification_head', None) is not None:
                outputs, is_forged_logit = model(imgs)
                if not torch.isfinite(outputs).all():
                    print("[NaN DETECTED] step=", step, " paths=", path)
                outputs = torch.nan_to_num(outputs, nan=0.0, neginf=-4.8, posinf=4.8)
                outputs = outputs.clamp(min=-4.8, max=4.8)
                # is_forged_logit = is_forged_logit.clamp(min=-20, max=20)
            else:
                outputs = model(imgs)
                
        if getattr(model, 'classification_head', None) is not None:
            cls_loss = cls_loss_fn(outputs, masks)
            
            with autocast(device_type=device_type, enabled=False):
                dice_loss = dice_loss_fn(outputs.float(), masks.float())
                img_loss = is_forged_img_loss_fn(is_forged_logit.squeeze(), is_forged.float())
                if torch.isnan(dice_loss):
                    print(f"[NaN DETECTED] Skipping backward at step {step}")
                    dice_loss = torch.nan_to_num(loss=dice_loss, nan=0.0, posinf=1.0, neginf=0.0)
            loss = alpha * cls_loss*loss_scaler + beta * dice_loss + img_loss*gamma
        else:
            cls_loss = cls_loss_fn(outputs, masks)
            dice_loss = dice_loss_fn(outputs, masks)
            loss = alpha * cls_loss*loss_scaler + beta * dice_loss

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"[STEP {step}] loss NaN")
            break


        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()

        if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        encoder_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), max_norm=float("inf"))
        decoder_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=float("inf"))
        cls_head_norm = torch.nn.utils. clip_grad_norm_(model.classification_head.parameters(), max_norm=float("inf")) if getattr(model, 'classification_head', None) is not None else None
        global_step = (epoch-1) * len(train_loader) + step

        if writer is not None:
            writer.add_scalar("Loss/Train_step", loss.item(), global_step)
            writer.add_scalar("Loss/Cls_step", cls_loss.item(), global_step)
            writer.add_scalar("Loss/Dice_step", dice_loss.item(), global_step)
            writer.add_scalar("Loss/Img_step", img_loss.item(), global_step)   
            writer.add_scalar("Loss/Total_step", loss.item(), global_step) 
            writer.add_scalar("Norm/Encoder", encoder_norm, global_step)
            writer.add_scalar("Norm/Decoder", decoder_norm, global_step)
            writer.add_scalar("Norm/Cls_head", cls_head_norm, global_step)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step)
            

        total_loss += loss.item()
        bce_total += cls_loss.item()
        dice_total += dice_loss.item()
        is_forged_img_loss += img_loss.item() if getattr(model, 'classification_head', None) is not None else 0


        # DEBUG
        if torch.isnan(dice_loss):
            print("⚠️ NaN in dice_loss detected")
            print("mask finite:", torch.isfinite(masks).all().item(), 
                "output finite:", torch.isfinite(outputs).all().item())
            print("mask sum:", masks.sum().item(), "output sum:", outputs.sum().item())
            print("mask unique:", masks.unique())
            print("output range:", outputs.min().item(), outputs.max().item())
            print("output shape:", outputs.shape, "mask shape:", masks.shape)
            break


    return {
        "loss_total": total_loss / len(train_loader),
        "loss_cls": bce_total / len(train_loader),
        "loss_dice": dice_total / len(train_loader),
        "loss_img": is_forged_img_loss / len(train_loader)
    }


def predict(model, test_path, device, test_path_file_list=None, img_size=128, max_size=None, interpolation=cv2.INTER_NEAREST, threshold=0.5, min_area_ratio=0.002, low_conf_max_prob = 0.06, low_viz_thr = 0.04, low_conf_min_pixel = 128, scaler: Optional[torch.amp.GradScaler] = None) -> dict[str, str]:

    """
    Return RLE string
    """
    device = str_to_device(device)
    device_type = device.type

    model.eval()
    predictions = {'case_id': [], 'annotation': []}
    if test_path_file_list is None:
        test_files = [f for f in os.listdir(test_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:max_size]
    else:
        test_files = test_path_file_list
    with torch.no_grad():
        for file in tqdm(test_files, desc="Predicting", leave=False):
            case_id = file.split('.')[0]

            #Load Img
            if test_path is None:
                img_path = file
                case_id =  os.path.splitext(os.path.basename(img_path))[0]
            else:
                img_path = os.path.join(test_path, file)
            img = cv2.imread(img_path)
            original_size = img.shape[:2]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            processed_img = preprocessing(img, img_size, interpolation)

            img_tensor = torch.from_numpy(processed_img)
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).contiguous()
            img_tensor = img_tensor.to(device)
            min_area = int(original_size[0] * original_size[1] * min_area_ratio)


            with autocast(device_type=device_type, enabled=scaler is not None):
                if getattr(model, 'classification_head', None) is not None:
                    logit_map, _ = model(img_tensor)
                else:
                    logit_map = model(img_tensor)
                
            proba_map = torch.sigmoid(logit_map)
            proba_map = proba_map.squeeze().cpu().numpy()  # (H, W)
            mask_pred = postprocessing(proba_map, original_size, threshold, low_conf_max_prob, low_viz_thr, low_conf_min_pixel)
            case_id = int(case_id)

            #후에 resize하기 전으로 변경 (img_size 똑같을 때)
            if mask_pred.sum() < min_area:
                predictions['case_id'].append(case_id)
                predictions['annotation'].append("authentic")
                # print("[DEBUG]: min_area, ", mask_pred.sum())
            else:
                instances = mask_to_instances(mask_pred, min_area=min_area)
                predictions['case_id'].append(case_id)
                predictions['annotation'].append(
                    ("authentic" if len(instances) == 0 else rle_encode(instances))
                )

    return predictions


def set_seed(seed: int = 42):
    random.seed(seed)                        # Python 랜덤 시드
    np.random.seed(seed)                     # NumPy 랜덤 시드
    torch.manual_seed(seed)                  # PyTorch CPU 시드
    torch.cuda.manual_seed(seed)             # PyTorch GPU 시드
    torch.cuda.manual_seed_all(seed)         # Multi-GPU 환경일 때

    os.environ["PYTHONHASHSEED"] = str(seed) # 해시 시드 (Python 3.3+)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # (CUDA ≥10.2용) 완전 결정적 연산

    # ⚠️ PyTorch 연산의 결정론적(deterministic) 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch 2.0+ (optional): 결정론 모드 강제
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    print(f"✅ Seed fixed at: {seed}\n")


if __name__ == "__main__":
    train()