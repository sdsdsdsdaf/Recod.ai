

recodai_luc_scientific_image_forgery_detection_path = r'C:\Users\user\.cache\kagglehub\competitions\recodai-luc-scientific-image-forgery-detection'

print('Data source import complete.')


# %%
print(recodai_luc_scientific_image_forgery_detection_path)

# %% [markdown]
# # **Init setting**

# %%
import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import random
import warnings
import torch.nn.init as init
import h5py
import segmentation_models_pytorch as smp

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

warnings.filterwarnings('ignore')

# %% [markdown]
# # **BaseLine U-Net**

# %%
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# %% [markdown]
# ## **Utils**

# %%
import json
import cv2
import numba
import numpy.typing as npt
from numba import types
import scipy.optimize

@numba.jit(nopython=True)
def _rle_encode_jit(x: npt.NDArray, fg_val:int=1) -> list[int]:
    """Numba-jitted RLE encoder."""
    dots = np.where(x.ravel(order='F') == fg_val)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1: # Is not continous 1
            run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b

    return run_lengths

def rle_encode(masks: list[npt.NDArray], fg_val:int=1) -> str:
    import json

class ParticipantVisibleError(Exception):
    pass


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
def _rle_decode_jit(mask_rle:npt.NDArray, height:int, width:int) -> npt.NDArray:
    """
    s: numpy array of run-length encoding pairs (start, length)
    shape: (height, width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """

    if not len(mask_rle) % 2 == 0:
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

def rle_decode(mask_rle:str, shape:tuple[int, int],) -> npt.NDArray:
    """
    mask_rle: run-length as string formatted (start length)
              empty predictions need to be encoded with '-'
    shape: (height, width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    mask_rle = json.loads(mask_rle)
    mask_rle = np.asarray(mask_rle, dtype=np.int32)
    starts = mask_rle[0::2]

    if not sorted(starts) == list(starts):
        raise ParticipantVisibleError('Submitted values must be in ascending order.')

    try:
        return _rle_decode_jit(mask_rle, shape[0], shape[1]).reshape(shape, order='F')
    except ValueError as e:
        raise ParticipantVisibleError(str(e)) from e


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
        for f in tqdm(mask_files):
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


# %% [markdown]
# ## 1) Environment Report

# %%
import sys, platform
print("Python: ", sys.version.split()[0])
print("OS: ", platform.platform())
print("Torch: ", torch.__version__)
print("DEVICE: ", DEVICE)

# %% [markdown]
# ## 2) Paths & Params

# %%
from pathlib import Path
import os

COMP_DIR = recodai_luc_scientific_image_forgery_detection_path
TEST_DIR = os.path.join(COMP_DIR, "test_images")
TRAIN_DIR = os.path.join(COMP_DIR, "train_images")

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
MIN_AREA = 16
TRAIN_SAMPLE_NUM = None
TEST_SAMPLE_NUM = None


#Aceleration Params -> Default: cpu settings
USE_PIN_MEM = torch.cuda.is_available
NUM_WORKERS = 4 if torch.cuda.is_available() and "Windows" not in platform.platform() else 0


print("COMP_DIR: ", COMP_DIR)
print("TEST_DIR: ", TEST_DIR)
print("TRAIN_DIR: ", TRAIN_DIR)
print("SUB_PATH: ", SUB_PATH)

# %% [markdown]
# ## 3) Simple U-Net

# %%
import torch
import torch.nn as nn

class SimpleUNet(nn.Module):

    def __init__(self, in_ch=3, latent_ch=None, out_ch=1, act=nn.ReLU, init_weight=True):
        super(SimpleUNet, self).__init__()

        if not isinstance(act, type) or not issubclass(act, nn.Module):
            raise TypeError(
                f"Expected an activation class (e.g., nn.ReLU), "
                f"but got an instance or invalid type: {act}"
            )

        self.act_cls = act
        self.enc1 = self.conv_block(in_ch, 32, latent_ch)
        self.enc2 = self.conv_block(32, 64, latent_ch)
        self.enc3 = self.conv_block(64, 128, latent_ch)

        self.bottleneck = self.conv_block(128, 256, latent_ch)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = self.conv_block(256, 128, latent_ch)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = self.conv_block(128, 64, latent_ch)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = self.conv_block(64, 32, latent_ch)

        #Output
        self.out = nn.Conv2d(32, out_ch, 1) #1x1 kernel
        self.pool = nn.MaxPool2d(2,2)

        if init_weight:
            self._initialize_weights()

    def conv_block(self, in_ch, out_ch, latent_ch=None):
        if latent_ch is None:
            latent_ch = out_ch
        #3x3 kernel Size remaining the same with padding=1
        return nn.Sequential(
            nn.Conv2d(in_ch, latent_ch, 3, padding=1),
            nn.BatchNorm2d(latent_ch),
            self.act_cls(inplace=True),
            nn.Conv2d(latent_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            self.act_cls(inplace=True),
        )

    def _initialize_weights(self):
        """
        Kaiming He initialization for ReLU/GELU/SiLU/LeakyReLU activations.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                act_name = self.act_cls.__name__.lower()

                if "relu" in act_name or "leaky" in act_name:
                    init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                elif "gelu" in act_name or "silu" in act_name or "swish" in act_name:
                    init.xavier_normal_(m.weight)   # ✅ GELU는 Xavier로 초기화
                else:
                    init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")

                if m.bias is not None:
                    init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        print("\nCOMPLETE WEIGHTS INITAILIZE\n")

    def forward(self, x):
        #Encoder
        e1  = self.enc1(x)                    # [32, H, W]
        e2  = self.enc2(self.pool(e1))        # [64, H/2, W/2]
        e3  = self.enc3(self.pool(e2))        # [128, H/4, W/4]
        #Bottleneck
        b   = self.bottleneck(self.pool(e3))  # [256, H/8, W/8]
        #Decoder
        d3  = self.up3(b)                     # [128, H/4, W/4]
        d3  = torch.cat([d3, e3], dim=1)      # [256, H/4, W/4] along channel dimension
        d3  = self.dec3(d3)                   # [128, H/4, W/4]

        d2  = self.up2(d3)                    # [64, H/2, W/2]
        d2  = torch.cat([d2, e2], dim=1)      # [128, H/2, W/2]
        d2  = self.dec2(d2)                   # [64, H/2, W/2]

        d1  = self.up1(d2)                    # [32, H, W]
        d1  = torch.cat([d1, e1], dim=1)      # [64, H, W]
        d1  = self.dec1(d1)                   # [32, H, W]

        out = self.out(d1)                    # [out_ch, H, W]
        return out

# %% [markdown]
# ## 4) Dataset

# %%
class FastDataset(Dataset):
    def __init__(self, authentic_path, forged_path, masks_path,
                img_size=128, is_train=True):

        self.img_size = img_size
        self.is_train = is_train
        self.samples = []

        #Authentic data and Forged data
        for path, is_forged in [(authentic_path, 0), (forged_path, 1)]:
            if not os.path.exists(path):
                continue
            for file in os.listdir(path)[:TRAIN_SAMPLE_NUM if is_train else TEST_SAMPLE_NUM]:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(path, file)
                    mask_path = os.path.join(masks_path, f"{file.split('.')[0]}.npy")
                    self.samples.append((img_path, mask_path, is_forged))

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, is_forged = self.samples[idx]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=INTERPOLATION)
        img = img.astype(np.float32) / 255.0

        # [H, W, C] -> [C, H, W]
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()

        #Load mask
        if is_forged and os.path.exists(mask_path):
            try:
                mask = np.load(mask_path) #[C, H, W]
                if mask.ndim == 3:
                    mask = mask.max(axis=0) if mask.shape[0] <= 10 else mask.max(axis=-1)
                    mask = cv2.resize(mask.astype(np.uint8), (self.img_size, self.img_size), interpolation=INTERPOLATION)
                    mask = (mask > 0).astype(np.float32)
            except:
                mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        else:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask

# %% [markdown]
# ### Lazy load dataset (HDF5)

# %%
class HybridDataset(Dataset):
    """
    Hybrid lazy/full-memory dataset.
    Supports loading from HDF5 (lazy) or fully preloading into memory.
    """

    def __init__(self, h5_path: str,
                 authentic_path: str,
                 forged_path: str,
                 masks_path: str,
                 img_size=128,
                 is_train=True,
                 preload=False,
                 verbose=False,
                ):
        """
        Args:
            h5_path: path to .h5 file (will be created if not exists)
            authentic_path, forged_path, masks_path: image & mask directories
            img_size: resize target
            preload: whether to fully load into memory
        """

        self.h5_path = h5_path
        self.img_size = img_size
        self.is_train = is_train
        self.preload = preload
        self.samples = []
        self.loaded = False

        # collect file paths
        for path, is_forged in [(authentic_path, 0), (forged_path, 1)]:
            if not os.path.exists(path):
                continue
            for file in os.listdir(path)[:TRAIN_SAMPLE_NUM if is_train else TEST_SAMPLE_NUM]:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(path, file)
                    mask_path = os.path.join(masks_path, f"{file.split('.')[0]}.npy")
                    self.samples.append((img_path, mask_path, is_forged))

        print(f"Loaded {len(self.samples)} samples")

        # HDF5 build if not exist
        if not os.path.exists(h5_path):
            print(f"⚙️ Building new HDF5 file at {h5_path}")
            self._build_h5(h5_path)
        else:
            print(f"✅ Using existing HDF5 file: {h5_path}")

        # preload to RAM (optional)
        if preload:
            print("📦 Preloading entire dataset into memory...")
            self.images, self.masks = self._load_all_from_h5()
            self.loaded = True

        if verbose:
            with h5py.File(self.h5_path, "r") as h5f:
                max_w = h5f.attrs["max_width"]
                max_h = h5f.attrs["max_height"]
                max_p = h5f.attrs["max_pixels"]
                min_w = h5f.attrs["min_width"]
                min_h = h5f.attrs["min_height"]
                min_p = h5f.attrs["min_pixels"]

            print(f"📊 Dataset Image Size Stats:")
            print(f"   Max Width: {max_w}, Max Height: {max_h}, Max Pixels: {max_p}")
            print(f"   Min Width: {min_w}, Min Height: {min_h}, Min Pixels: {min_p}")

    def _build_h5(self, h5_path):
        """Convert all images/masks into an HDF5 file."""
        with h5py.File(h5_path, "w") as h5f:
            n = len(self.samples)
            img_ds = h5f.create_dataset("images", (n, 3, self.img_size, self.img_size), dtype="float32")
            mask_ds = h5f.create_dataset("masks", (n, 1, self.img_size, self.img_size), dtype="float32")
            forged_ds = h5f.create_dataset("is_forged", (n,), dtype="uint8")
        
            max_w, max_h, max_p = 0, 0, 0
            min_w, min_h, min_p = float('inf'), float('inf'), float('inf')

            for i, (img_path, mask_path, is_forged) in enumerate(tqdm(self.samples)):
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # [H,W,C]

                max_w = max(max_w, img.shape[1])
                max_h = max(max_h, img.shape[0])
                max_p = max(max_p, img.shape[0] * img.shape[1])

                min_w = min(min_w, img.shape[1])
                min_h = min(min_h, img.shape[0])
                min_p = min(min_p, img.shape[0] * img.shape[1])

                img = cv2.resize(img, (self.img_size, self.img_size), interpolation=INTERPOLATION).astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))  # [C,H,W]

               

                if is_forged and os.path.exists(mask_path):
                    try:
                        mask = np.load(mask_path)
                        if mask.ndim == 3:
                            mask = mask.max(axis=0) if mask.shape[0] <= 10 else mask.max(axis=-1)
                            mask = cv2.resize(mask.astype(np.uint8), (self.img_size, self.img_size), interpolation=INTERPOLATION)
                            mask = (mask > 0).astype(np.float32)
                    except:
                        mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
                else:
                    mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)

                mask = np.expand_dims(mask, axis=0)  # [1,H,W]
                img_ds[i] = img
                mask_ds[i] = mask
                forged_ds[i] = is_forged

            h5f.attrs["max_width"] = max_w
            h5f.attrs["max_height"] = max_h
            h5f.attrs["max_pixels"] = max_p
            h5f.attrs["min_width"] = min_w
            h5f.attrs["min_height"] = min_h
            h5f.attrs["min_pixels"] = min_p
            

    def _load_all_from_h5(self):
        """Load all samples into RAM."""
        with h5py.File(self.h5_path, "r") as h5f:
            imgs = np.array(h5f["images"][:])
            masks = np.array(h5f["masks"][:])
        imgs = torch.from_numpy(imgs)
        masks = torch.from_numpy(masks)
        return imgs, masks

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.loaded:
            # ✅ Full memory mode
            img = self.images[idx]
            mask = self.masks[idx]
        else:
            # ✅ Lazy loading from HDF5
            with h5py.File(self.h5_path, "r") as h5f:
                img = torch.from_numpy(h5f["images"][idx])
                mask = torch.from_numpy(h5f["masks"][idx])

        return img, mask


# %% [markdown]
# ## 5) Train Seting

# %%
paths = {
    'train_authentic': os.path.join(TRAIN_DIR, "authentic"),
    'train_forged': os.path.join(TRAIN_DIR, "forged"),
    'train_masks': os.path.join(COMP_DIR, "train_masks"),
    'test_images': TEST_DIR
}
import os
print(paths['train_masks'], os.path.exists(paths['train_masks']))

# Dataset
print("\n[1/5] Loading data...")
train_dataset = HybridDataset(
    "train_data.h5",
    paths['train_authentic'],
    paths['train_forged'],
    paths['train_masks'],
    img_size=IMG_SIZE,
    is_train=True,
    preload=True,
    verbose=True
)


print(f"\nConfig: {IMG_SIZE}X{IMG_SIZE}, BS={BATCH_SIZE}, Epochs={NUM_EPOCHS} LR={LR}, THRESHOLD={THRESHOLD}, MIN_AREA={MIN_AREA}")
pos_w_resized = compute_pos_weight(h5_path="train_data.h5", mode="resized", img_size=IMG_SIZE, interpolation=INTERPOLATION)
pos_w_original = compute_pos_weight(dataset=train_dataset, authentic_path=paths['train_authentic'], forgded_path=paths['train_forged'], masks_path=paths['train_masks'], mode="original", img_size=IMG_SIZE, interpolation=INTERPOLATION)

print(f"\nResized 기준: {pos_w_resized.item():.2f}")
print(f"Original 기준: {pos_w_original.item():.2f}")

bce_loss_fn = bce_loss_cls(pos_weight=pos_w_resized)
dice_loss_fn = dice_loss_cls(mode='binary', from_logits=True)   

# %%
def is_low_confidence(prob_map):
    if float(prob_map.max()) >= LOW_CONF_MAX_PROB: #일단 “low confidence 아님으로 판단
        return False
    cover = int((prob_map >= LOW_VIZ_THR).sum()) #검출됨
    return cover < LOW_CONF_MIN_PIXEL

# %%
def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    bce_total = 0
    dice_total = 0

    for imgs, masks in tqdm(train_loader, desc="Training", leave=False):

        imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)

        #후에 resize하는 과정도 train할 지 결정
        optimizer.zero_grad()
        outputs = model(imgs)
        bce_loss = bce_loss_fn(outputs, masks)
        dice_loss = dice_loss_fn(outputs, masks)
        loss = alpha * bce_loss + beta * dice_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        bce_total += bce_loss.item()
        dice_total += dice_loss.item()

    return {
        "total_loss": total_loss / len(train_loader),
        "bce_loss": bce_total / len(train_loader),
        "dice_loss": dice_total / len(train_loader)
    }

def predict(model, test_path, device, img_size=128):
    model.eval()
    predictions = {}

    test_files = [f for f in os.listdir(test_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    with torch.no_grad():
        for file in tqdm(test_files, desc="Predicting"):
            case_id = file.split('.')[0]

            #Load Img
            img_path = os.path.join(test_path, file)
            img = cv2.imread(img_path)
            original_size = img.shape[:2]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, (img_size, img_size), interpolation=INTERPOLATION)

            img_tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0)
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).contiguous()
            img_tensor = img_tensor.to(device)

            mask_pred = model(img_tensor)
            mask_pred = torch.sigmoid(mask_pred)
            mask_pred = mask_pred.squeeze().detach().cpu().numpy()

            #DEBUG
            print("[DEBUG] Forged pixel num before PostProcessing: ", (mask_pred > 0.5).astype(np.uint8).sum())
            if is_low_confidence(mask_pred):
                print("DEBUG: low_confidence")
                mask_pred  = np.zeros_like(mask_pred)

            mask_pred = (mask_pred > 0.5).astype(np.uint8)
            #후에 bilinear사용 고려
            mask_pred = cv2.resize(mask_pred, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST) # (W, H)

            #PostProcessing
            kernel = np.ones((3,3), np.uint8)
            mask_pred = cv2.morphologyEx(mask_pred, cv2.MORPH_OPEN, kernel)
            mask_pred = cv2.morphologyEx(mask_pred, cv2.MORPH_CLOSE, kernel)

            #후에 resize하기 전으로 변경 (img_size 똑같을 때)
            if mask_pred.sum() < MIN_AREA:
                predictions[case_id] = "authentic"
                print("DEBUG: min_area, ", mask_pred.sum())
            else:
                predictions[case_id] = rle_encode([mask_pred])

    return predictions




# %% [markdown]
# ## 6) Train
# 

# %%
if __name__ == "__main__":
    import time
    from torchinfo import summary

    set_seed()

    start = time.time()

    print("="*60)
    print("BASELINE U-Net FORGEY DETECTION")
    print("="*60)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=USE_PIN_MEM,
    )

    # Model
    print("\n[2/5] Creating model...")

    
    model = smp.Unet(
        encoder_name="efficientnet-b3",     # backbone
        encoder_weights="imagenet",         # pretrained 가중치
        in_channels=3,                      # 입력 채널 수 (RGB면 3)
        classes=1,                          # 출력 채널 수 (binary mask)
        activation=None,                    # ⚠️ sigmoid는 loss 함수 쪽에서 처리
    )
    

   # model = SimpleUNet(in_ch=3, out_ch=1, act=nn.ReLU, init_weight=True)

    
    summary(
        model,
        input_size=(1, 3, IMG_SIZE, IMG_SIZE),
        col_names=(
            "input_size",
            "output_size",
            "num_params",
            "mult_adds",
            "kernel_size",
            "trainable"
        ),
        verbose=True
    )

    model = model.to(DEVICE)

    params_num = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params_num:,} (vs 11M+ for Mask R-CNN)")

    optimizer = optimizer_cls(model.parameters(), lr=LR)

    #Train
    print(f"\n[3/5] Training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - TOTAL Loss: {loss['total_loss']:.4f} BCE Loss: {loss['bce_loss']:.4f} Dice Loss: {loss['dice_loss']}")

    # Save
    print("\n[4/5] Saving model...")
    torch.save(model.state_dict(), "smp_model.pth")


    #Predict
    print("\n[5/5] Predicting on test set...")
    predictions = predict(model, paths['test_images'], DEVICE, IMG_SIZE)

    #Create submission
    sample = pd.read_csv(os.path.join(COMP_DIR, "sample_submission.csv"))
    submission = pd.DataFrame(columns=sample.columns)

    for case_id in sample['case_id']:
        annotation = predictions.get(str(case_id), "authentic")
        submission.loc[len(submission)] = [case_id, annotation]

    submission.to_csv("submission.csv", index=False)

    # Stats
    authentic = (submission['annotation'] == 'authentic').sum()
    forged = len(submission) - authentic

    print("\n" + "="*60)
    print("DONE! ✓")
    print("="*60)
    print(f"Predictions: {len(submission)}")
    print(f"  Authentic: {authentic}")
    print(f"  Forged: {forged}")
    print(f"Submission saved: submission.csv")
    print("="*60)

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    print(submission.head(10))

# %% [markdown]
# # **Visualize**


