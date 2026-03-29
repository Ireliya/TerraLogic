# Enhanced HiViT configuration for SAR detection
# Based on newer implementation from codelab/code/detection

_base_ = [
    'mmdet::_base_/models/faster_rcnn_r50_fpn.py',
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_1x.py', 
    'mmdet::_base_/default_runtime.py'
]

# Model configuration with enhanced HiViT backbone
model = dict(
    type='FasterRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[0, 0, 0],  # SAR images typically don't need normalization
        std=[1, 1, 1],
        bgr_to_rgb=True,
        pad_size_divisor=32
    ),
    backbone=dict(
        _delete_=True,
        type='HiViT',
        img_size=224,
        patch_size=16,
        embed_dim=512,  # Base model configuration
        frozen_stages=-1,
        depths=[2, 2, 20],
        num_heads=8,
        mlp_ratio=4.0,
        rpe=False,
        drop_path_rate=0.0,
        with_fpn=True,
        out_indices=['H', 'M', 19, 19],
        use_checkpoint=True,
        global_indices=[4, 9, 14, 19],
        window_size=14,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoint-800.pth'
        )
    ),
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512, 512],
        out_channels=256,
        num_outs=5
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]
        ),
        loss_cls=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=True, 
            loss_weight=1.0
        ),
        loss_bbox=dict(
            type='L1Loss', 
            loss_weight=1.0
        )
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=5,  # Enhanced SAR classes: ship, bridge, harbor, oil_tank, aircraft
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]
            ),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', 
                use_sigmoid=False, 
                loss_weight=1.0
            ),
            loss_bbox=dict(
                type='L1Loss', 
                loss_weight=1.0
            )
        )
    ),
    # Model training and testing config
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1
            ),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False
        ),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        ),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1
            ),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True
            ),
            pos_weight=-1,
            debug=False
        )
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        ),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100
        )
    )
)

# Dataset configuration
dataset_type = 'CocoDataset'
data_root = 'data/SAR/'

# Enhanced data pipeline for SAR images
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

# Enhanced optimizer configuration
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=5e-4,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='HiViTLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=20, layer_decay_rate=0.9)
)

# Learning rate configuration
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[27, 33]
)

# Runtime configuration
runner = dict(type='EpochBasedRunner', max_epochs=36)
evaluation = dict(interval=1, metric='bbox', save_best='auto')

# Enhanced class names for SAR detection
classes = ('ship', 'bridge', 'harbor', 'oil_tank', 'aircraft')

# Default hooks configuration
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1)
)

# Work directory
work_dir = './work_dirs/hivit_enhanced_sar'
