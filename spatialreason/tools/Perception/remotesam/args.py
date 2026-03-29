"""
Argument parser for RemoteSAM model.
"""

import argparse


def get_parser():
    """Get argument parser for RemoteSAM model."""
    parser = argparse.ArgumentParser(description='RemoteSAM Model Arguments')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='lavt_one',
                        help='Model architecture')
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='Device to run the model on')
    parser.add_argument('--pretrained', type=str, default='',
                        help='Path to pretrained model')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training/inference')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for data')
    parser.add_argument('--dataset', type=str, default='refcoco',
                        help='Dataset name')
    parser.add_argument('--split', type=str, default='val',
                        help='Dataset split')
    
    # Model specific arguments
    parser.add_argument('--window12', action='store_true', default=True,
                        help='Use window size 12 for attention')
    parser.add_argument('--swin_type', type=str, default='base',
                        help='Swin transformer type')
    parser.add_argument('--mha', type=str, default='',
                        help='Multi-head attention configuration')
    
    # Architecture arguments
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of layers')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Input/Output arguments
    parser.add_argument('--input_size', type=int, default=896,
                        help='Input image size')
    parser.add_argument('--max_query_len', type=int, default=20,
                        help='Maximum query length')
    parser.add_argument('--word_len', type=int, default=20,
                        help='Maximum word length')
    
    # Loss arguments
    parser.add_argument('--loss_type', type=str, default='ce',
                        help='Loss function type')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    
    # Optimization arguments
    parser.add_argument('--optimizer', type=str, default='adamw',
                        help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='polynomial',
                        help='Learning rate scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Warmup epochs')
    
    # Evaluation arguments
    parser.add_argument('--eval_freq', type=int, default=1,
                        help='Evaluation frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Model save frequency')
    
    # Augmentation arguments
    parser.add_argument('--augment', action='store_true', default=False,
                        help='Use data augmentation')
    parser.add_argument('--flip_prob', type=float, default=0.5,
                        help='Probability of horizontal flip')
    parser.add_argument('--color_jitter', type=float, default=0.1,
                        help='Color jitter strength')
    
    # Inference arguments
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Segmentation threshold')
    parser.add_argument('--min_area', type=int, default=100,
                        help='Minimum area for valid segments')
    
    # Logging arguments
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory for logs')
    parser.add_argument('--exp_name', type=str, default='remotesam',
                        help='Experiment name')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Verbose logging')
    
    # Distributed training arguments
    parser.add_argument('--distributed', action='store_true', default=False,
                        help='Use distributed training')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for distributed training')
    parser.add_argument('--world_size', type=int, default=1,
                        help='World size for distributed training')
    
    # Checkpoint arguments
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    
    # Visualization arguments
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='Save visualization results')
    parser.add_argument('--vis_dir', type=str, default='./visualizations',
                        help='Directory for visualizations')
    
    # Testing arguments
    parser.add_argument('--test_only', action='store_true', default=False,
                        help='Only run testing')
    parser.add_argument('--test_split', type=str, default='test',
                        help='Test split name')
    
    # Memory optimization
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False,
                        help='Use gradient checkpointing to save memory')
    parser.add_argument('--mixed_precision', action='store_true', default=False,
                        help='Use mixed precision training')
    
    # Model ensemble
    parser.add_argument('--ensemble', action='store_true', default=False,
                        help='Use model ensemble')
    parser.add_argument('--ensemble_weights', type=str, nargs='+', default=[],
                        help='Paths to ensemble model weights')
    
    return parser


class Args:
    """Simple args class for compatibility."""
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __repr__(self):
        items = [f"{k}={v}" for k, v in self.__dict__.items()]
        return f"Args({', '.join(items)})"


def get_default_args():
    """Get default arguments for RemoteSAM."""
    return Args(
        model='lavt_one',
        device='cuda:2',  # Use GPU 2 for perception tools
        pretrained='',
        batch_size=1,
        lr=1e-4,
        epochs=100,
        data_root='./data',
        dataset='refcoco',
        split='val',
        window12=True,
        swin_type='base',
        mha='',
        num_layers=3,
        hidden_dim=256,
        dropout=0.1,
        input_size=896,
        max_query_len=20,
        word_len=20,
        loss_type='ce',
        weight_decay=1e-4,
        optimizer='adamw',
        scheduler='polynomial',
        warmup_epochs=10,
        eval_freq=1,
        save_freq=10,
        augment=False,
        flip_prob=0.5,
        color_jitter=0.1,
        threshold=0.7,
        min_area=100,
        log_dir='./logs',
        exp_name='remotesam',
        verbose=False,
        distributed=False,
        local_rank=0,
        world_size=1,
        resume='',
        checkpoint_dir='./checkpoints',
        visualize=False,
        vis_dir='./visualizations',
        test_only=False,
        test_split='test',
        gradient_checkpointing=False,
        mixed_precision=False,
        ensemble=False,
        ensemble_weights=[],
        # Additional attributes needed by the model
        fusion_drop=0.0,
        ddp_trained_weights=False,
        ck_bert='/home/yuhang/Downloads/SpatialreasonAgent/bert-base-uncased',
        bert_tokenizer='/home/yuhang/Downloads/SpatialreasonAgent/bert-base-uncased',
        lang_model='bert-base-uncased',
        fusion_layers=3,
        fusion_dim=256,
        use_checkpoint=False,
        drop_path_rate=0.3,
        patch_norm=True,
        ape=False,
        out_indices=(0, 1, 2, 3)
    )
