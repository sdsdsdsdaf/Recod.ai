import h5py
from torch.utils.data import Dataset
import os, cv2, torch
import numpy as np
from tqdm.auto import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

def pad_to_min(img, mask, min_size, pad_mode="reflect"):
    """
    Pad image/mask so that H,W >= min_size.
    pad_mode: "reflect" | "replicate" | "zero"
    """
    H, W = img.shape[:2]
    pad_h = max(0, min_size - H)
    pad_w = max(0, min_size - W)
    if pad_h == 0 and pad_w == 0:
        return img, mask

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    if pad_mode == "reflect":
        img_border = cv2.BORDER_REFLECT_101
        img_p = cv2.copyMakeBorder(img, top, bottom, left, right, img_border)
    elif pad_mode == "replicate":
        img_border = cv2.BORDER_REPLICATE
        img_p = cv2.copyMakeBorder(img, top, bottom, left, right, img_border)
    elif pad_mode == "constant" or pad_mode == "zero":
        img_p = cv2.copyMakeBorder(img, top, bottom, left, right,
                                   borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:
        raise ValueError(f"Unknown pad_mode: {pad_mode}")

    # mask는 항상 0-padding이 안전
    mask_p = cv2.copyMakeBorder(mask, top, bottom, left, right,
                                borderType=cv2.BORDER_CONSTANT, value=0)
    return img_p, mask_p


def sample_crop_xy(mask, patch_size, pos_center_prob=0.7):
    """
    Choose top-left crop coord (x0,y0).
    - If mask has positives and rand < pos_center_prob: ensure a positive pixel is inside the crop.
    - Else: random crop.
    """
    H, W = mask.shape
    s = patch_size
    max_y = H - s
    max_x = W - s

    # pad_to_min should guarantee max_x/max_y >= 0
    if max_y < 0 or max_x < 0:
        return 0, 0

    if mask.sum() == 0 or np.random.rand() > pos_center_prob:
        y0 = np.random.randint(0, max_y + 1) if max_y > 0 else 0
        x0 = np.random.randint(0, max_x + 1) if max_x > 0 else 0
        return x0, y0

    ys, xs = np.where(mask > 0)
    k = np.random.randint(0, len(xs))
    cx, cy = int(xs[k]), int(ys[k])

    x0_min = max(0, cx - s + 1)
    x0_max = min(cx, max_x)
    y0_min = max(0, cy - s + 1)
    y0_max = min(cy, max_y)

    x0 = np.random.randint(x0_min, x0_max + 1) if x0_max >= x0_min else max(0, min(cx, max_x))
    y0 = np.random.randint(y0_min, y0_max + 1) if y0_max >= y0_min else max(0, min(cy, max_y))
    return x0, y0


class PatchSizeScheduler:
    def __init__(self, sizes, probs):
        assert len(sizes) == len(probs)
        self.sizes = list(map(int, sizes))
        p = np.array(probs, dtype=np.float64)
        self.probs = p / p.sum()

    def sample(self):
        return int(np.random.choice(self.sizes, p=self.probs))


class FastDataset(Dataset):
    def __init__(self, authentic_path, forged_path, masks_path,
                img_size=128, is_train=True, train_sample_num=None, test_sample_num=None, interpolation=cv2.INTER_NEAREST):

        self.img_size = img_size
        self.is_train = is_train
        self.samples = []
        self.interpolation = interpolation
        self.train_sample_num = train_sample_num
        self.test_sample_num = test_sample_num

        #Authentic data and Forged data
        for path, is_forged in [(authentic_path, 0), (forged_path, 1)]:
            if not os.path.exists(path):
                continue
            for file in os.listdir(path)[:self.train_sample_num if is_train else self.test_sample_num]:
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
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=self.interpolation)
        img = img.astype(np.float32) / 255.0

        # [H, W, C] -> [C, H, W]
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()

        #Load mask
        if is_forged and os.path.exists(mask_path):
            try:
                mask = np.load(mask_path) #[C, H, W]
                if mask.ndim == 3:
                    mask = mask.max(axis=0) if mask.shape[0] <= 10 else mask.max(axis=-1)
                    mask = cv2.resize(mask.astype(np.uint8), (self.img_size, self.img_size), interpolation=self.interpolation)
                    mask = (mask > 0).astype(np.float32)
            except:
                mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        else:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask

class HybridCropDataset(Dataset):
    """
    Hybrid lazy/full-memory dataset (HDF5 유지) + patch crop in __getitem__.
    - HDF5 format: group-per-sample (str(idx)/img, mask, is_forged, path)
    - No resize. Only pad + crop.
    - Supports preload (RAM)
    - __init__ signature matches HybridDataset
    """

    def __init__(self, h5_path: str,
                 authentic_path: str,
                 forged_path: str,
                 masks_path: str,
                 img_size=224,        # patch_size (가변 가능)
                 storage_size=256,    # (호환용 파라미터) 사용 안 함 / 무시 가능
                 is_train=True,
                 preload=False,
                 verbose=False,
                 train_transform: A.Compose=None,
                 test_transform: A.Compose=None,
                 train_sample_num=None,
                 test_sample_num=None,
                 # ---- 추가 옵션 (기본값 넣어도 HybridDataset 호환) ----
                 patch_scheduler: PatchSizeScheduler=None,
                 pos_center_prob: float=0.7,
                 pad_mode: str="reflect",   # "reflect" | "replicate" | "zero"
                 rebuild_h5_if_needed: bool=False,
                 supplemental_images_path: str=None,
                 supplemental_masks_path: str=None,
                 use_supplemental: bool=True,
                ):
        self.h5_path = h5_path
        self.img_size = int(img_size)
        self.storage_size = int(storage_size)  # kept for API compatibility
        self.is_train = is_train
        self.preload = preload
        self.loaded = False

        self.patch_scheduler = patch_scheduler
        self.pos_center_prob = float(pos_center_prob)
        self.pad_mode = pad_mode

        # Albumentations (resize 없는 것만 넣는 걸 추천)
        self.transform = train_transform if is_train else test_transform

        # collect file paths (same as HybridDataset)
        self.samples = []
        for path, is_forged in [(authentic_path, 0), (forged_path, 1)]:
            if not os.path.exists(path):
                continue
            files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            limit = train_sample_num if is_train else test_sample_num
            if limit:
                files = files[:limit]

            for file in files:
                img_path = os.path.join(path, file)
                mask_name = file.split('.')[0]
                mask_path = os.path.join(masks_path, f"{mask_name}.npy")
                self.samples.append((img_path, mask_path, is_forged))
                
        if is_train and use_supplemental and supplemental_images_path and supplemental_masks_path:
            if os.path.exists(supplemental_images_path) and os.path.exists(supplemental_masks_path):
                sup_files = [f for f in os.listdir(supplemental_images_path)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                for file in sup_files:
                    img_path = os.path.join(supplemental_images_path, file)
                    mask_name = file.rsplit('.', 1)[0]
                    mask_path = os.path.join(supplemental_masks_path, f"{mask_name}.npy")

                    # mask가 있으면 forged로 취급, 없으면 authentic 취급
                    is_forged = 1 if os.path.exists(mask_path) else 0
                    self.samples.append((img_path, mask_path, is_forged))

            else:
                print("⚠️ supplemental path not found. skip supplemental.")

        print(f"Loaded {len(self.samples)} samples")

        # build/validate HDF5
        if (not os.path.exists(h5_path)) or rebuild_h5_if_needed:
            print(f"⚙️ Building new group-based HDF5 at {h5_path}")
            self._build_h5_group(h5_path)
        else:
            print(f"✅ Using existing HDF5 file: {h5_path}")

        # preload
        if preload:
            print("📦 Preloading entire dataset into memory (group-based)...")
            self.images, self.masks, self.paths, self.is_forged_arr, img_size = self._load_all_from_h5_group()
            self.loaded = True

        if verbose:
            with h5py.File(self.h5_path, "r") as h5f:
                n = int(h5f.attrs.get("num_samples", len(h5f.keys())))
                print(f"📊 HDF5 groups: {n}")
                print(f"   pad_mode={self.pad_mode}, pos_center_prob={self.pos_center_prob}")
                print(f"   Dataset Image Size Stats:"
                      f" Max Width: {img_size['max_w']}, Max Height: {img_size['max_h']}, "
                      f"Min Width: {img_size['min_w']}, Min Height: {img_size['min_h']}")

    # ----------------------------
    # HDF5 builders/loaders (group-based)
    # ----------------------------
    def _build_h5_group(self, h5_path):
        with h5py.File(h5_path, "w") as h5f:
            n = len(self.samples)
            h5f.attrs["num_samples"] = n

            max_w, max_h = 0, 0
            for i, (img_path, mask_path, is_forged) in enumerate(tqdm(self.samples)):
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                H, W = img.shape[:2]
                max_w, max_h = max(max_w, W), max(max_h, H)

                mask = np.zeros((H, W), dtype=np.uint8)
                if is_forged and os.path.exists(mask_path):
                    try:
                        mask_arr = np.load(mask_path)
                        result = mask_arr

                        if result.ndim == 3:
                            # (C,H,W) -> (H,W,C) if needed
                            if result.shape[0] < result.shape[2]:
                                result = np.transpose(result, (1, 2, 0))
                            result = np.max(result, axis=-1)

                        result = (result > 0).astype(np.uint8)
                        if result.shape != (H, W):
                            # only if mismatch
                            result = cv2.resize(result, (W, H), interpolation=cv2.INTER_NEAREST)
                        mask = result
                    except Exception as e:
                        print(f"Mask error {mask_path}: {e}")
                        mask = np.zeros((H, W), dtype=np.uint8)

                grp = h5f.create_group(str(i))
                grp.create_dataset("img", data=img, dtype="uint8", compression="lzf")
                grp.create_dataset("masks", data=mask, dtype="uint8", compression="lzf")
                grp.create_dataset("is_forged", data=np.uint8(is_forged))
                grp.create_dataset("path", data=np.string_(mask_path))

            h5f.attrs["max_width"] = int(max_w)
            h5f.attrs["max_height"] = int(max_h)

    def _load_all_from_h5_group(self):
        imgs, masks, paths, forged = [], [], [], []
        img_size = {
            "min_h": float('inf'),
            "min_w": float('inf'),
            "max_h": 0,
            "max_w": 0,
        }
        with h5py.File(self.h5_path, "r") as h5f:
            n = int(h5f.attrs.get("num_samples", len(h5f.keys())))
            for i in tqdm(range(n)):
                grp = h5f[str(i)]
                img = grp["img"][...]
                mask = grp["masks"][...]

                img_size["min_h"] = min(img_size["min_h"], img.shape[0])
                img_size["min_w"] = min(img_size["min_w"], img.shape[1])
                img_size["max_h"] = max(img_size["max_h"], img.shape[0])
                img_size["max_w"] = max(img_size["max_w"], img.shape[1])

                is_f = int(grp["is_forged"][()])
                path = grp["path"][()]
                if isinstance(path, bytes):
                    path = path.decode("utf-8")
                imgs.append(img)
                masks.append(mask)
                paths.append(path)
                forged.append(is_f)
        return imgs, masks, paths, np.array(forged, dtype=np.uint8), img_size

    def _read_h5(self, idx):
        with h5py.File(self.h5_path, "r") as h5f:
            grp = h5f[str(idx)]
            img = grp["img"][...]
            mask = grp["masks"][...]
            is_forged = int(grp["is_forged"][()])
            path = grp["path"][()]
            if isinstance(path, bytes):
                path = path.decode("utf-8")
        return img, mask, path, is_forged

    # ----------------------------
    # Main
    # ----------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.loaded:
            img = self.images[idx]
            mask = self.masks[idx]
            path = self.paths[idx]
            is_forged = int(self.is_forged_arr[idx])
        else:
            img, mask, path, is_forged = self._read_h5(idx)

        # patch_size 결정
        patch_size = self.patch_scheduler.sample() if self.patch_scheduler is not None else self.img_size

        # pad + crop
        img, mask = pad_to_min(img, mask, patch_size, pad_mode=self.pad_mode)

        pos_p = self.pos_center_prob if self.is_train else 0.0
        x0, y0 = sample_crop_xy(mask, patch_size, pos_center_prob=pos_p)

        img_p = img[y0:y0 + patch_size, x0:x0 + patch_size]
        mask_p = mask[y0:y0 + patch_size, x0:x0 + patch_size]

        # Albumentations (resize 없는 것만)
        if self.transform is not None:
            out = self.transform(image=img_p, mask=mask_p)
            img_t = out["image"]
            mask_t = out["mask"].unsqueeze(0).float()
        else:
            img_t = torch.from_numpy(img_p).permute(2, 0, 1).float() / 255.0
            mask_t = torch.from_numpy(mask_p).unsqueeze(0).float()

        return img_t, mask_t, path, torch.tensor(is_forged, dtype=torch.float32)


class HybridDataset(Dataset):
    """
    Hybrid lazy/full-memory dataset with Albumentations.
    Supports loading from HDF5 (lazy) or fully preloading into memory.
    """

    def __init__(self, h5_path: str,
                 authentic_path: str,
                 forged_path: str,
                 masks_path: str,
                 img_size=224,        # 최종 출력 크기 (Crop Size)
                 storage_size=256,    # HDF5에 저장할 크기 (Resize Size, Crop을 위해 약간 큼)
                 is_train=True,
                 preload=False,
                 verbose=False,
                 train_transform: A.Compose=None,
                 test_transform: A.Compose=None,
                 train_sample_num=None,
                 test_sample_num=None
                ):
        """
        Args:
            storage_size: HDF5에 저장될 때의 크기 (RandomCrop을 위해 img_size보다 크게 잡는 것 추천)
            img_size: 모델에 들어갈 최종 크기
        """

        self.h5_path = h5_path
        self.img_size = img_size
        self.storage_size = storage_size # 저장용 크기 추가
        self.is_train = is_train
        self.preload = preload
        self.samples = []
        self.loaded = False

        # -----------------------------------------------------------
        # 🔥 [핵심] Albumentations 변환 정의
        # -----------------------------------------------------------
        if self.is_train:
            self.transform = train_transform
        else:
            self.transform = test_transform

        # collect file paths
        for path, is_forged in [(authentic_path, 0), (forged_path, 1)]:
            if not os.path.exists(path):
                continue
            # 파일 리스트 가져오기 및 개수 제한
            files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            limit = train_sample_num if is_train else test_sample_num
            if limit: files = files[:limit]

            for file in files:
                img_path = os.path.join(path, file)
                # 마스크 경로 추정 (npy 우선, 없으면 이미지)
                mask_name = file.split('.')[0]
                mask_path = os.path.join(masks_path, f"{mask_name}.npy")
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
            self.images, self.masks, self.path, self.is_forged = self._load_all_from_h5()
            self.loaded = True

        if verbose:
            with h5py.File(self.h5_path, "r") as h5f:
                # Stats 출력 (저장된 속성이 있다면)
                if "max_width" in h5f.attrs:
                    print(f"📊 Dataset Image Size Stats:")
                    print(f"   Max Width: {h5f.attrs['max_width']}, Max Height: {h5f.attrs['max_height']}")

    def _build_h5(self, h5_path):
        """Convert all images/masks into an HDF5 file."""
        with h5py.File(h5_path, "w") as h5f:
            n = len(self.samples)
            
            # 🔥 [변경점 1] float32 대신 uint8로 저장 (Albumentations용)
            # 🔥 [변경점 2] (N, 3, H, W) 대신 (N, H, W, 3)으로 저장 (HWC 포맷 유지)
            img_ds = h5f.create_dataset("images", (n, self.storage_size, self.storage_size, 3), dtype="uint8")
            mask_ds = h5f.create_dataset("masks", (n, self.storage_size, self.storage_size), dtype="uint8")
            forged_ds = h5f.create_dataset("is_forged", (n,), dtype="uint8")
            
            # 문자열 저장을 위한 가변 길이 타입
            dt_str = h5py.special_dtype(vlen=str) 
            path_ds = h5f.create_dataset("original_mask_path", (n,), dtype=dt_str)
        
            max_w, max_h, max_p = 0, 0, 0
            min_w, min_h, min_p = float('inf'), float('inf'), float('inf')

            for i, (img_path, mask_path, is_forged) in enumerate(tqdm(self.samples)):
                # 1. 이미지 로드
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # [H,W,C]

                # 통계 수집
                h, w, _ = img.shape
                max_w, max_h = max(max_w, w), max(max_h, h)
                min_w, min_h = min(min_w, w), min(min_h, h)

                # 2. 리사이즈 (저장용 크기로) - 정규화(Normalize)는 안함! uint8 유지!
                img = cv2.resize(img, (self.storage_size, self.storage_size), interpolation=cv2.INTER_AREA)
                
                # 3. 마스크 로드 및 처리
                mask = np.zeros((self.storage_size, self.storage_size), dtype=np.uint8)
                if is_forged and os.path.exists(mask_path):
                    try:
                        mask_arr = np.load(mask_path)
                        # 차원 정리 (Squeeze & Channel selection)
                        if mask_arr.ndim == 3:
                             # (C, H, W) -> (H, W, C)
                            if mask_arr.shape[0] < mask_arr.shape[2]:
                                mask_arr = np.transpose(mask_arr, (1, 2, 0))
                            
                            # 채널 병합 (최대값 기준)
                            mask_arr = np.max(mask_arr, axis=-1)
                        
                        # 리사이즈
                        mask_arr = cv2.resize(mask_arr.astype(np.float32), (self.storage_size, self.storage_size), interpolation=cv2.INTER_NEAREST)
                        mask = (mask_arr > 0).astype(np.uint8) # 0 or 1
                    except Exception as e:
                        print(f"Mask error {mask_path}: {e}")
                        mask = np.zeros((self.storage_size, self.storage_size), dtype=np.uint8)

                # 4. 저장 (HWC, uint8 상태)
                img_ds[i] = img
                mask_ds[i] = mask
                forged_ds[i] = is_forged
                path_ds[i] = mask_path

            h5f.attrs["max_width"] = max_w
            h5f.attrs["max_height"] = max_h

    def _load_all_from_h5(self):
        """Load all samples into RAM."""
        with h5py.File(self.h5_path, "r") as h5f:
            imgs = np.array(h5f["images"][:]) # [N, H, W, 3] uint8
            masks = np.array(h5f["masks"][:]) # [N, H, W] uint8
            path = [p.decode("utf-8") if isinstance(p, bytes) else p for p in h5f["original_mask_path"][:]]
            is_forged = np.array(h5f["is_forged"][:])
            
        # 여기서는 텐서로 바꾸지 않고 Numpy 상태 유지 (getitem에서 transform 하려고)
        return imgs, masks, path, is_forged

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.loaded:
            # Full memory mode (Numpy uint8 상태)
            img = self.images[idx]
            mask = self.masks[idx]
            path = self.path[idx]
            is_forged = self.is_forged[idx]
        else:
            # Lazy loading from HDF5
            with h5py.File(self.h5_path, "r") as h5f:
                img = h5f["images"][idx] # [H, W, 3] uint8
                mask = h5f["masks"][idx] # [H, W] uint8
                path = h5f["original_mask_path"][idx]
                if isinstance(path, bytes): path = path.decode("utf-8")
                is_forged = h5f["is_forged"][idx]

        # 🔥 [핵심] Albumentations 적용
        # img: [H, W, 3] uint8, mask: [H, W] uint8 상태여야 함
        transformed = self.transform(image=img, mask=mask)
        
        img_tensor = transformed["image"] # [3, 224, 224] Float Tensor (Normalized)
        mask_tensor = transformed["mask"] # [224, 224] Tensor
        
        # 마스크 차원 추가 [H, W] -> [1, H, W] 및 float 변환
        mask_tensor = mask_tensor.unsqueeze(0).float()

        return img_tensor, mask_tensor, path, torch.tensor(is_forged, dtype=torch.float32)

'''
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
                 interpolation=cv2.INTER_NEAREST,
                 train_sample_num=None,
                 test_sample_num=None
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
        self.interpolation = interpolation


        # collect file paths
        for path, is_forged in [(authentic_path, 0), (forged_path, 1)]:
            if not os.path.exists(path):
                continue
            for file in os.listdir(path)[:train_sample_num if is_train else test_sample_num]:
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
            self.images, self.masks, self.path, self.is_forged = self._load_all_from_h5()
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
            mask_path_ds = h5f.create_dataset("original_mask_path", (n,), dtype=h5py.string_dtype(encoding="utf-8"))
        
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

                img = preprocessing(img, self.img_size, interpolation=self.interpolation, div=255.0)
                img = np.transpose(img, (2, 0, 1))  # [C,H,W]

               

                if is_forged and os.path.exists(mask_path):
                    try:
                        mask = np.load(mask_path)
                        if mask.ndim == 3:
                            mask = mask.max(axis=0) if mask.shape[0] <= 10 else mask.max(axis=-1)
                            mask = cv2.resize(mask.astype(np.uint8), (self.img_size, self.img_size), interpolation=self.interpolation)
                            mask = (mask > 0).astype(np.float32)
                    except:
                        mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
                else:
                    mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)

                mask = np.expand_dims(mask, axis=0)  # [1,H,W]
                img_ds[i] = img
                mask_ds[i] = mask
                forged_ds[i] = is_forged
                mask_path_ds[i] = mask_path

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
            path = [p.decode("utf-8") for p in h5f["original_mask_path"][:]]
            is_forged = np.array(h5f["is_forged"][:])
        imgs = torch.from_numpy(imgs)
        masks = torch.from_numpy(masks)
        
        is_forged = torch.from_numpy(is_forged)
        return imgs, masks, path, is_forged


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.loaded:
            # ✅ Full memory mode
            img = self.images[idx]
            mask = self.masks[idx]
            path = self.path[idx]
            is_forged = self.is_forged[idx]
        else:
            # ✅ Lazy loading from HDF5
            with h5py.File(self.h5_path, "r") as h5f:
                img = torch.from_numpy(h5f["images"][idx])
                mask = torch.from_numpy(h5f["masks"][idx])
                path = torch.from_numpy(h5f["original_mask_path"][idx])
                is_forged = torch.from_numpy(h5f["is_forged"][idx])

        return img, mask, path, is_forged
'''
    

if __name__ == "__main__":
    COMP_DIR = r"C:\Users\user\.cache\kagglehub\competitions\recodai-luc-scientific-image-forgery-detection"
    TRAIN_DIR = os.path.join(COMP_DIR, "train_images")
    TEST_DIR = os.path.join(COMP_DIR, "test_images")
    IMG_SIZE = 384
    PAD_MODE = 'constant'
    
    paths = {
        'train_authentic': os.path.join(TRAIN_DIR, "authentic"),
        'train_forged': os.path.join(TRAIN_DIR, "forged"),
        'train_masks': os.path.join(COMP_DIR, "train_masks"),
        'test_images': TEST_DIR,
        'sup_images': os.path.join(COMP_DIR, "supplemental_images"),
        'sup_masks': os.path.join(COMP_DIR, "supplemental_masks")
    }
    
    # Test the HybridDataset
    train_transform = A.Compose([
        A.Resize(256, 256),
        A.RandomCrop(224, 224),
        A.HorizontalFlip(),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    test_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    dataset = HybridCropDataset(
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
        train_sample_num=None,
        test_sample_num=None,
        pad_mode=PAD_MODE,
        rebuild_h5_if_needed=False,
        supplemental_images_path=paths['sup_images'],
        supplemental_masks_path=paths["sup_masks"]
    )


    print(f"Dataset length: {len(dataset)}")
    img, mask, path, is_forged = dataset[0]
    print(f"Image shape: {img.shape}, Mask shape: {mask.shape}, Path: {path}, Is Forged: {is_forged}")
    
    for img, mask, path, is_forged in dataset:
        if mask.sum() > 0:
            print("mask 존재")
            break