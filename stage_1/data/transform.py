"""MONAI preprocessing and augmentation pipelines (Sec. 4.2 of the paper)."""
import math

import monai.transforms as T

import config


def _preprocessing(keys, has_label):
    """RAS, 1 mm isotropic, foreground crop, HU window [-10, 140] -> [0, 1], non-zero z-score."""
    return [
        T.EnsureChannelFirstd(keys=keys),
        T.Orientationd(keys=keys, axcodes='RAS'),
        T.Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode=["trilinear", "nearest"] if has_label else "trilinear"),
        T.CropForegroundd(keys=keys, source_key='image'),
        T.ScaleIntensityRanged(keys=["image"], a_min=-10, a_max=140, b_min=0.0, b_max=1.0, clip=True),
        T.SpatialPadd(keys=keys, spatial_size=config.img_size, mode='constant', value=0.0),
        T.NormalizeIntensityd(keys=['image'], channel_wise=True, nonzero=config.non_zero_norm),
    ]


def _augmentation(keys, has_label):
    """Random affine (rotation +-30 deg, scale +-10%), intensity jitter +-10%, left-right flip."""
    return [
        T.RandAffined(keys=keys, prob=0.5, rotate_range=math.radians(30), scale_range=0.1,
                      mode=['bilinear', 'nearest'] if has_label else 'bilinear', padding_mode='border'),
        T.RandScaleIntensityd(keys=['image'], factors=0.1, prob=0.5),
        T.RandShiftIntensityd(keys=['image'], offsets=0.1, prob=0.5),
        T.RandFlipd(keys=keys, prob=0.5, spatial_axis=[0]),
    ]


def get_segmentation_transforms(subset, dataset_name):
    """Segmentation volumes. In training, 2 patches of 96^3 per volume, centred on a class
    drawn with the dataset-specific ratios of config.class_ratios_per_dataset.
    Validation returns the whole volume (sliding-window inference)."""
    keys = ["image", "label"]
    transforms = [
        T.LoadImaged(keys=keys),
        *_preprocessing(keys, has_label=True),
    ]
    if subset == 'train':
        ratios = config.class_ratios_per_dataset[dataset_name]
        transforms += _augmentation(keys, has_label=True) + [
            T.RandCropByLabelClassesd(keys=keys, label_key='label', spatial_size=config.img_size,
                                      num_samples=config.patch_per_case, ratios=ratios,
                                      num_classes=len(ratios)),
        ]
    return T.Compose(transforms)


def get_classification_transforms(subset, has_mask, num_samples=1):
    """Classification volumes, one or two 96^3 patches per volume.

    has_mask=True (positives with a pseudo-mask): the pseudo-mask guides the crop,
        5:1 positive/negative in training, always positive in validation, and is
        kept as the dense target of the "cls_pseudo" task.
    has_mask=False (healthy volumes, or positives with data.cls_crop=random): random crop
        of `num_samples` patches (2 for the positives, to match the guided crop).
    """
    keys = ["image", "label"] if has_mask else ["image"]
    transforms = [
        T.LoadImaged(keys=keys, image_only=True),
        *_preprocessing(keys, has_label=has_mask),
    ]
    if subset == 'train':
        transforms += _augmentation(keys, has_label=has_mask)
        if has_mask:
            transforms.append(T.RandCropByPosNegLabeld(
                keys=keys, image_key='image', label_key='label',
                pos=5.0, neg=1.0, spatial_size=config.img_size, num_samples=2))
        else:
            transforms.append(T.RandSpatialCropSamplesd(
                keys=["image"], roi_size=config.img_size, num_samples=num_samples, random_size=False))
    else:
        if has_mask:
            transforms.append(T.RandCropByPosNegLabeld(
                keys=keys, image_key='image', label_key='label',
                pos=1.0, neg=0.0, spatial_size=config.img_size, num_samples=1))
        else:
            transforms.append(T.RandSpatialCropSamplesd(
                keys=["image"], roi_size=config.img_size, num_samples=1))

    transforms.append(T.ToTensord(keys=["image", "class_label"] + (["label"] if has_mask else [])))
    return T.Compose(transforms)
