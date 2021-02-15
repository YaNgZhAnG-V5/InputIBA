from iba.datasets import build_dataset
import numpy as np
import torch


def test_imagenet():
    dataset_type = 'ImageNet'
    data_root = 'data/'
    img_norm_cfg = dict(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    img_size = 224
    train_pipeline = [
        dict(type='Resize', height=img_size, width=img_size, always_apply=True),
        dict(type='Normalize', always_apply=True, **img_norm_cfg),
        dict(type='ToTensor')]

    test_pipeline = [
        dict(type='Resize', height=img_size, width=img_size, always_apply=True),
        dict(type='Normalize', always_apply=True, **img_norm_cfg),
        dict(type='ToTensor')]

    data = dict(
        data_loader=dict(
            batch_size=1,
            shuffle=True,
            num_workers=0),
        train=dict(
            type=dataset_type,
            img_root=data_root + 'imagenet/images/train/',
            annot_root=data_root + 'imagenet/annotations/train/',
            ind_to_cls_file=data_root + 'imagenet_class_index.json',
            pipeline=train_pipeline),
        val=dict(
            type=dataset_type,
            img_root=data_root + 'imagenet/images/val/',
            annot_root=data_root + 'imagenet/annotations/val/',
            ind_to_cls_file=data_root + 'imagenet_class_index.json',
            pipeline=test_pipeline,
            with_bbox=True))

    val_set = build_dataset(data['val'])
    print(len(val_set))

    sample = val_set[0]
    assert 'bboxes' in sample.keys()

    img = sample['img']
    bboxes = sample['bboxes']

    assert isinstance(img, torch.Tensor) and img.shape[0] == 3
    assert isinstance(bboxes, np.ndarray) and bboxes.shape[-1] == 4 and bboxes.dtype == int
