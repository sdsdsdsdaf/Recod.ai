import h5py
from torch.utils.data import Dataset
import os, cv2, torch
import numpy as np
from tqdm.auto import tqdm

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

                img = cv2.resize(img, (self.img_size, self.img_size), interpolation=self.interpolation).astype(np.float32) / 255.0
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

    

