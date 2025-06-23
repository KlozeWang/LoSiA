import argparse
import torch
from peft_pretraining import args_utils
from peft_pretraining import training_utils

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--train_on_inputs", default=False, action="store_true")
    parser.add_argument("--pad_to_max_len", default=False, action="store_true")
    parser.add_argument("--val_size", type=int, default=0)
    parser.add_argument("--val_data", type=str, default=None)
    parser.add_argument("--train_size", type=int, default=-1)
    parser.add_argument("--continue_from", type=str, default=None)

    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--grad_clipping", type=float, default=0.0)   

    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--activation_checkpointing", action="store_true", default=False)

    parser.add_argument("--num_training_steps", type=int, default=None,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")
    
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default=None)

    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)

    
    # LoSiA parameters
    parser.add_argument("--rank_factor", type=float, default=1.0/8.0)
    parser.add_argument("--update_type", type=str, default="asy_period", choices=["asy_period", "syn_period"])
    parser.add_argument("--taylor_type", type=str, default="param_mix", choices=["param_second", "param_mix", "param_first"], \
                                         help="Mode of importance estimation calculation")
    parser.add_argument("--imp_beta1", type=float, default=0.85, help="Exponetial average factor for importance")
    parser.add_argument("--imp_beta2", type=float, default=0.85, help="Exponetial average factor for uncertainty")
    parser.add_argument("--period", type=int, default=100, help="Time slot T, controlling the period of subnet reselection")
    parser.add_argument("--losia_scale", type=float, default=1.0, help="Usually set to 1")
    parser.add_argument("--output_dim_factor", type=float, default=0.0, \
                                        help="Dimension reduction factor of output layer, set to 0 for full output dimension fine-tuning")
    parser.add_argument("--use_pro", default=False, action="store_true", \
                                     help="Training with losia-pro, which is a fine implementation of losia.")

    # beta1 for adafactor
    parser.add_argument("--beta1", type=float, default=0.0)
    
    # GaLore parameters
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--update_proj_gap", type=int, default=500)
    parser.add_argument("--galore_scale", type=float, default=1.0)
    parser.add_argument("--proj_type", type=str, default="std")
    
    parser.add_argument("--single_gpu", default=False, action="store_true")
    parser.add_argument("--cuda_history", default=False, action="store_true")
    
    args = parser.parse_args(args)

    args = args_utils.check_args_torchrun_main(args)
    return args