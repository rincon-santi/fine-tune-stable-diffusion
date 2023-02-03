"""
This file is the main file of the project. It is responsible for reading the parameters from the DVC file, and then
calling the dreambooth script with the parameters.
"""

import sys
import argparse
sys.path.insert(1, '.')
from src.model.templates import dreambooth_script
import dvc.api

params = dvc.api.params_show()
MODEL_NAME=params["model_id"]
PRIOR_PRESERVATION=params["prior_preservation"]=="True"
PRIOR_LOSS_WEIGHT=params["prior_loss_weight"]
RESOLUTION=params["resolution"]
TRAIN_BATCH_SIZE=params["train_batch_size"]
GRADIENT_ACCUMULATION_STEPS=params["gradient_accumulation_steps"]
LEARNING_RATE=params["learning_rate"]
LR_SCHEDULER=params["lr_scheduler"]
LR_WARMUP_STEPS=params["lr_warmup_steps"]
NUM_CLASS_IMAGES=params["num_class_images"]
MAX_TRAIN_STEPS=params["max_train_steps"]

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="Prompt to be learned",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        required=False,
        help=(
            "Prompt to generate regularization images"
        ),
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args

def start_training(args):
    namespace = argparse.Namespace()

    namespace.pretrained_model_name_or_path=MODEL_NAME
    namespace.instance_data_dir="data/dog"
    namespace.class_data_dir="data/class"
    namespace.output_dir="trained_model"
    namespace.logging_dir="logging"
    namespace.with_prior_preservation=PRIOR_PRESERVATION
    namespace.prior_loss_weight=PRIOR_LOSS_WEIGHT
    namespace.instance_prompt=args.instance_prompt
    namespace.class_prompt=args.class_prompt
    namespace.resolution=RESOLUTION
    namespace.train_batch_size=TRAIN_BATCH_SIZE
    namespace.gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS
    namespace.learning_rate=LEARNING_RATE
    namespace.lr_scheduler=LR_SCHEDULER
    namespace.lr_warmup_steps=LR_WARMUP_STEPS
    namespace.num_class_images=NUM_CLASS_IMAGES
    namespace.max_train_steps=MAX_TRAIN_STEPS
    namespace.mixed_precision=None
    namespace.report_to="tensorboard"
    namespace.train_text_encoder=False
    namespace.seed=None
    namespace.prior_generation_precision=None
    namespace.revision=None
    namespace.sample_batch_size=4
    namespace.tokenizer_name=None
    namespace.center_crop=False
    namespace.checkpointing_steps=500
    namespace.resume_from_checkpoint=None
    namespace.gradient_checkpointing=False
    namespace.scale_lr=False
    namespace.lr_num_cycles=1
    namespace.lr_power=1.0
    namespace.use_8bit_adam=False
    namespace.dataloader_num_workers=0
    namespace.adam_beta1=0.9
    namespace.adam_beta2=0.999
    namespace.adam_weight_decay=1e-2
    namespace.adam_epsilon=1e-08
    namespace.max_grad_norm=1.0
    namespace.push_to_hub=False
    namespace.hub_token=None
    namespace.hub_model_id=None
    namespace.allow_tf32=False
    namespace.local_rank=-1
    namespace.enable_xformers_memory_efficient_attention=False
    namespace.set_grads_to_none=False

    print("Starting dreambooth script with namespace: {ns}".format(ns=namespace))
    dreambooth_script.main(namespace)
    
if __name__ == "__main__":
    args = parse_args()
    start_training(args)
