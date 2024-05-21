import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
from enums import BackboneArch, OptimizerType, LossName

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for Visual Place Recognition model.")

    # Model arguments
    parser.add_argument("--backbone_arch", type=BackboneArch, choices=list(BackboneArch), default=BackboneArch.RESNET50, help="Backbone architecture")
    parser.add_argument("--pretrained", type=bool, default=True, help="Use pretrained weights")
    parser.add_argument("--layers_to_freeze", type=int, default=2, help="Number of layers to freeze in the backbone")
    parser.add_argument("--layers_to_crop", type=list, default=[], help="Layers to crop in the backbone")
    parser.add_argument("--agg_arch", type=str, default="ConvAP", help="Aggregator architecture")
    parser.add_argument("--agg_config", type=dict, default={}, help="Configuration for the aggregator")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--optimizer", type=OptimizerType, choices=list(OptimizerType), default=OptimizerType.ADAM, help="Optimizer type")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
    parser.add_argument("--warmup_steps", type=int, default=600, help="Number of warmup steps")
    parser.add_argument("--milestones", type=list, default=[5, 10, 15, 25], help="Milestones for learning rate scheduler")
    parser.add_argument("--lr_mult", type=float, default=0.3, help="Learning rate multiplier for scheduler")
    parser.add_argument("--loss_name", type=LossName, choices=list(LossName), default=LossName.MULTI_SIMILARITY_LOSS, help="Loss function name")
    parser.add_argument("--miner_name", type=str, default="MultiSimilarityMiner", help="Miner name")
    parser.add_argument("--miner_margin", type=float, default=0.1, help="Margin for the miner")
    parser.add_argument("--faiss_gpu", type=bool, default=False, help="Use FAISS GPU for validation")

    # Data module arguments
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument("--img_per_place", type=int, default=4, help="Images per place")
    parser.add_argument("--min_img_per_place", type=int, default=4, help="Minimum images per place")
    parser.add_argument("--shuffle_all", type=bool, default=False, help="Shuffle all images or shuffle in-city only")
    parser.add_argument("--random_sample_from_each_place", type=bool, default=True, help="Random sample from each place")
    parser.add_argument("--image_size", type=tuple, default=(320, 320), help="Image size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for dataloader")
    parser.add_argument("--show_data_stats", type=bool, default=True, help="Show data statistics")
    parser.add_argument("--val_set_names", type=list, default=["pitts30k_val", "msls_val"], help="Validation set names")

    # ModelCheckpoint arguments
    parser.add_argument('--monitor', type=str, default='pitts30k_val/R1', help='Metric to monitor for checkpointing')
    parser.add_argument('--filename', type=str, default='{model.encoder_arch}_epoch({epoch:02d})_step({step:04d})_R1[{pitts30k_val/R1:.4f}]_R5[{pitts30k_val/R5:.4f}]', help='Filename pattern for saved checkpoints')
    parser.add_argument('--auto_insert_metric_name', type=bool, default=False, help='Whether to automatically insert the metric name in the filename')
    parser.add_argument('--save_weights_only', type=bool, default=True, help='Whether to save only the model weights')
    parser.add_argument('--save_top_k', type=int, default=3, help='Number of top checkpoints to save')
    parser.add_argument('--mode', type=str, default='max', help='Mode for selecting checkpoints (max or min)')

    # Trainer arguments
    parser.add_argument("--accelerator", type=str, default="gpu", help="Accelerator type")
    parser.add_argument("--devices", type=list, default=[0], help="Devices to use")
    parser.add_argument("--default_root_dir", type=str, default="./LOGS/", help="Default root directory for logs")
    parser.add_argument("--num_sanity_val_steps", type=int, default=0, help="Number of sanity validation steps")
    parser.add_argument("--precision", type=int, default=16, help="Precision (16 or 32)")
    parser.add_argument("--max_epochs", type=int, default=30, help="Maximum number of epochs")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1, help="Check validation every n epochs")
    parser.add_argument("--reload_dataloaders_every_n_epochs", type=int, default=1, help="Reload dataloaders every n epochs")
    parser.add_argument("--log_every_n_steps", type=int, default=20, help="Log every n steps")
    parser.add_argument("--fast_dev_run", type=bool, default=False, help="Fast development run (debugging)")

    return parser