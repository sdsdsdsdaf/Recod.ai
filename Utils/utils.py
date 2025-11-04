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


def rle_encode(masks: list[npt.NDArray], fg_val: int = 1) -> str:
    """
    Adapted from contrails RLE https://www.kaggle.com/code/inversion/contrails-rle-submission
    Args:
        masks: list of numpy array of shape (height, width), 1 - mask, 0 - background
    Returns: run length encodings as a string, with each RLE JSON-encoded and separated by a semicolon.
    """
    return ';'.join([json.dumps(_rle_encode_jit(x, fg_val)) for x in masks])


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
        device, cls_loss_fn, dice_loss_fn, alpha=0.5, beta=0.5, loss_scaler=8, scheduler=None,
        interpolation=cv2.INTER_NEAREST, threshold=0.5, min_area=100,
        low_conf_max_prob=0.06, low_viz_thr=0.04, low_conf_min_pixel=128, # PostProcessing Setting,
        fold=None, use_amp=False,
    ):

    train_loss_log_dict = defaultdict(list)
    val_loss_log_dict = defaultdict(list)
    scaler = None
    best1_f1 = 0.0
    best5_f1 = [0.0, 0.0, 0.0, 0.0, 0.0]
    f1_log_list = []
    os.makedirs("Weights", exist_ok=True)
    if fold is not None:
        os.makedirs(os.path.join("Weights", f"FOLD{fold}"), exist_ok=True)

    if use_amp:
        scaler = GradScaler()

    # DEBUG
    print("Scaler is Not None in def Train: ", scaler is not None)

    for E in range(1, epoch + 1):
        epoch_start_time = time()
        losses = train_one_epoch(model, train_loader, optimizer, device, cls_loss_fn, dice_loss_fn, scheduler, alpha, beta, loss_scaler, scaler)
        print(f"[{E}/{epoch}] TRAIN TOTAL LOSS: {losses['loss_total']:.4f} CLS LOSS: {losses['loss_cls']:.4f} SCALING CLS LOSS: {loss_scaler*losses['loss_cls']:.4f} DICE LOSS: {losses['loss_dice']:.4f}")
        for k, v in losses.items():
                    train_loss_log_dict[k].append(v)
        if val_loader is None:
            continue

        metric_score = evaluate(model, val_loader, device, cls_loss_fn, dice_loss_fn, alpha, beta, loss_scaler, interpolation, threshold, min_area, low_conf_max_prob, low_viz_thr, low_conf_min_pixel, scaler)
        print(f"[{E}/{epoch}] VAL F1: {metric_score['f1_score']:.4f} TOTAL LOSS: {metric_score['loss_total']:.4f} CLS LOSS: {metric_score['loss_cls']:.4f}  SCALING CLS LOSS: {loss_scaler*metric_score['loss_cls']:.4f} DICE LOSS: {metric_score['loss_dice']:.4f}")
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
        print(f"TIME [EPOCH {E}]: {timedelta(seconds=epoch_end_time - epoch_start_time)}")

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
        model, val_loader, device:Union[torch.device,str], cls_loss_fn, dice_loss_fn, alpha=0.5, beta=0.5, loss_scaler=8,
        interpolation=cv2.INTER_NEAREST, threshold=0.5, min_area=100,
        low_conf_max_prob=0.06, low_viz_thr=0.04, low_conf_min_pixel=128, scaler=None,
    ):
    model.eval()
    total_loss = 0
    cls_total = 0
    dice_total = 0
    solution = {'case_id': [], 'annotation': [], 'shape': []}
    file_path_list = []
    
    for imgs, masks, mask_path, is_forged in tqdm(val_loader, desc="Calculating Loss", leave=False):

        imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=scaler is not None):
            logit_map = model(imgs)
            cls_loss = cls_loss_fn(logit_map, masks)
            dice_loss = dice_loss_fn(logit_map, masks)
            loss = alpha * cls_loss*loss_scaler + beta * dice_loss

        total_loss += loss.item()
        cls_total += cls_loss.item()
        dice_total += dice_loss.item()

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

    prediction= predict(
        model, None, device, test_path_file_list=file_path_list, img_size=img.shape[1],
        max_size=None, interpolation=interpolation, threshold=threshold, min_area=min_area,
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

    return {"f1_score": score_, "loss_total": total_loss, "loss_cls": cls_total, "loss_dice": dice_total}

    



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
        




def train_one_epoch(model, train_loader, optimizer, device:Union[torch.device,str], cls_loss_fn, dice_loss_fn, scheduler, alpha=0.5, beta=0.5, loss_scaler=None, scaler: Optional[torch.amp.GradScaler] = None) -> dict[str, float]:

    model.train()
    total_loss = 0
    bce_total = 0
    dice_total = 0

    device = str_to_device(device)
    device_type = device.type

    for imgs, masks, path, is_forged in tqdm(train_loader, desc="Training", leave=False):

        imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        #후에 resize하는 과정도 train할 지 결정
        optimizer.zero_grad()

        with autocast(device_type=device_type, enabled=scaler is not None):
            outputs = model(imgs)
            cls_loss = cls_loss_fn(outputs, masks)
            dice_loss = dice_loss_fn(outputs, masks)
            loss = alpha * cls_loss*loss_scaler + beta * dice_loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        bce_total += cls_loss.item()
        dice_total += dice_loss.item()

    return {
        "loss_total": total_loss / len(train_loader),
        "loss_cls": bce_total / len(train_loader),
        "loss_dice": dice_total / len(train_loader)
    }


def predict(model, test_path, device, test_path_file_list=None, img_size=128, max_size=None, interpolation=cv2.INTER_NEAREST, threshold=0.5, min_area=100, low_conf_max_prob = 0.06, low_viz_thr = 0.04, low_conf_min_pixel = 128, scaler: Optional[torch.amp.GradScaler] = None) -> dict[str, str]:

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

            with autocast(device_type=device_type, enabled=scaler is not None):
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