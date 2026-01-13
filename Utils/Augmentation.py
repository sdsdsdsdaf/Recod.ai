import albumentations as A

def get_forge_augmentation(version="weak", config=None):
    """
    Fully parameterized Albumentations-based augmentation builder.
    Suitable for Swin Transformer / EfficientNet / UNet forgery segmentation.

    Args:
        version (str): "weak" or "strong"
        config (dict): dictionary of augmentation settings.
                       If None → use default config.

    Returns:
        A.Compose
    """

    # ===== Load default config =====
    default_config = {
        "image_size": 224,

        # Geometry
        "flip_p": 0.5,
        "vflip_p": 0.5,
        "rotate90_p": 0.0,   # weak에서는 거의 사용 안 함

        "rrc_scale": (0.8, 1.0),
        "rrc_ratio": (0.8, 1.2),
        "rrc_p": 0.5,

        "shift_limit": 0.05,
        "scale_limit": 0.1,
        "rotate_limit": 10,
        "ssr_p": 0.3,

        # Texture / Noise / Photometric
        "color_brightness": 0.2,
        "color_contrast": 0.2,
        "color_saturation": 0.2,
        "color_hue": 0.05,
        "color_p": 0.5,

        "noise_var": (5, 20),
        "noise_p": 0.25,

        "jpeg_lower": 70,
        "jpeg_upper": 100,
        "jpeg_p": 0.3,

        # Blur
        "motion_blur_limit": 7,
        "motion_blur_p": 0.3,

        "median_blur_limit": 5,
        "median_blur_p": 0.2,

        # HSV
        "hsv_h": 10,
        "hsv_s": 20,
        "hsv_v": 15,
        "hsv_p": 0.4,

        # BrightnessContrast
        "bc_brightness": 0.3,
        "bc_contrast": 0.3,
        "bc_p": 0.5,

        # Downscale
        "downscale_min": 0.4,
        "downscale_max": 0.75,
        "downscale_p": 0.4,

        # Dropout
        "drop_max_holes": 5,
        "drop_h": 32,
        "drop_w": 32,
        "drop_p": 0.3,
    }

    # override config
    cfg = default_config if config is None else {**default_config, **config}

    # helper lambda
    img_size = cfg["image_size"]

    # ========== Weak Augmentation ==========
    if version == "weak":
        aug_list = [

            A.HorizontalFlip(p=cfg["flip_p"]),
            A.VerticalFlip(p=cfg["vflip_p"]),

            A.RandomResizedCrop(
                size=(img_size, img_size),
                scale=cfg["rrc_scale"],
                ratio=cfg["rrc_ratio"],
                p=cfg["rrc_p"]
            ),

            A.ColorJitter(
                brightness=cfg["color_brightness"],
                contrast=cfg["color_contrast"],
                saturation=cfg["color_saturation"],
                hue=cfg["color_hue"],
                p=cfg["color_p"]
            ),

            A.GaussNoise(var_limit=cfg["noise_var"], p=cfg["noise_p"]),

            A.ImageCompression(
                quality_lower=cfg["jpeg_lower"],
                quality_upper=cfg["jpeg_upper"],
                p=cfg["jpeg_p"]
            ),

            A.ShiftScaleRotate(
                shift_limit=cfg["shift_limit"],
                scale_limit=cfg["scale_limit"],
                rotate_limit=cfg["rotate_limit"],
                border_mode=0,
                p=cfg["ssr_p"]
            ),
        ]

    # ========== Strong Augmentation ==========
    elif version == "strong":
        aug_list = [

            A.HorizontalFlip(p=cfg["flip_p"]),
            A.VerticalFlip(p=cfg["vflip_p"]),

            A.RandomRotate90(p=cfg["rotate90_p"]),

            A.RandomResizedCrop(
                size=(img_size, img_size),
                scale=cfg["rrc_scale"],
                ratio=cfg["rrc_ratio"],
                p=cfg["rrc_p"]
            ),

            A.ShiftScaleRotate(
                shift_limit=cfg["shift_limit"],
                scale_limit=cfg["scale_limit"],
                rotate_limit=cfg["rotate_limit"],
                border_mode=0,
                p=cfg["ssr_p"]
            ),

            # Texture / Noise
            A.GaussNoise(var_limit=cfg["noise_var"], p=cfg["noise_p"]),

            A.JpegCompression(
                quality_lower=cfg["jpeg_lower"],
                quality_upper=cfg["jpeg_upper"],
                p=cfg["jpeg_p"]
            ),

            A.Downscale(
                scale_min=cfg["downscale_min"],
                scale_max=cfg["downscale_max"],
                p=cfg["downscale_p"]
            ),

            A.MotionBlur(blur_limit=cfg["motion_blur_limit"], p=cfg["motion_blur_p"]),
            A.MedianBlur(blur_limit=cfg["median_blur_limit"], p=cfg["median_blur_p"]),

            A.HueSaturationValue(
                hue_shift_limit=cfg["hsv_h"],
                sat_shift_limit=cfg["hsv_s"],
                val_shift_limit=cfg["hsv_v"],
                p=cfg["hsv_p"]
            ),

            A.RandomBrightnessContrast(
                brightness_limit=cfg["bc_brightness"],
                contrast_limit=cfg["bc_contrast"],
                p=cfg["bc_p"]
            ),

            A.CoarseDropout(
                max_holes=cfg["drop_max_holes"],
                max_height=cfg["drop_h"],
                max_width=cfg["drop_w"],
                mask_fill_value=0,
                p=cfg["drop_p"]
            ),
        ]

    else:
        raise ValueError("version must be 'weak' or 'strong'")

    # Build Compose
    return A.Compose(aug_list, additional_targets={"mask": "mask"})

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform(aug_version="weak", aug_config=None, img_size=224):
    """
    Combine custom augmentation + your base HDF5 crop/normalize pipeline.

    Args:
        aug_version (str): "weak", "strong", or "none"
        aug_config   (dict): override config for augmentation
        img_size (int): final training size (224 commonly)

    Returns:
        A.Compose
    """

    # 1) aug_version = "none" → 기존 transform 그대로 사용
    if aug_version == "none":
        return A.Compose([
            A.RandomCrop(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ], additional_targets={"mask": "mask"})
    
    if aug_version == "crop":
        return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),

        A.Normalize(),
        ToTensorV2(),
    ], additional_targets={"mask": "mask"})

    # 2) weak/strong augmentation 가져오기
    aug = get_forge_augmentation(
        version=aug_version,
        config=aug_config
    )

    # 3) augmentation + crop + normalize + tensor 변환
    transform = A.Compose([
        # ------- ① 위조 augmentation 단계 -------
        aug,

        # ------- ② HDF5 256 → 224 랜덤 크롭 -------
        A.RandomCrop(img_size, img_size),

        # ------- ③ Normalize -------
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),

        # ------- ④ Tensor 변환 -------
        ToTensorV2()
    ], additional_targets={"mask": "mask"})

    return transform

