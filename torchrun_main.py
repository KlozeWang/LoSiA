import os
import time
import json
import random
import numpy as np
import shutil

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
from arguments import parse_args

import matplotlib.pyplot as plt
from torch.profiler import profile, ProfilerActivity

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM

from collections import OrderedDict
import datasets
import datasets.distributed
import wandb
from torch import cuda
from torch.utils.data import random_split
from tasks import get_preprocessor

from tqdm import tqdm
from loguru import logger

from peft_pretraining import training_utils
from peft_pretraining.dataloader import PreprocessedIterableDataset_TrainOnInputs
from peft_pretraining.dataloader import PreprocessedIterableDataset_TrainOnLabels

from functools import partial
from peft_pretraining.modeling_llama import LlamaForCausalLM

import bitsandbytes as bnb
from peft_pretraining.losia_pro import replace_linear_recursive

from losia_torch.scheduler import Recorder
from optimizer import get_optimizer

transformers.logging.set_verbosity_error()


@torch.no_grad()
def evaluate_model(model, val_data, pad_idx, train_on_inputs, tokenizer, batch_size, max_length, device):
    if train_on_inputs:
        dataset = PreprocessedIterableDataset_TrainOnInputs(val_data, tokenizer, batch_size=batch_size, max_length=max_length)
    else:
        dataset = PreprocessedIterableDataset_TrainOnLabels(val_data, tokenizer, batch_size=batch_size, max_length=max_length, gemma_template=args.gemma_template)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=args.workers)

    world_size = int(os.environ["WORLD_SIZE"])

    target_eval_tokens = 10_000_000
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 1

    for batch_idx, batch in enumerate(dataloader):
        if evaluated_on_tokens > target_eval_tokens:
            break
        total_batches += 1

        batch = {k: v.to(device) for k, v in batch.items()}
        if args.train_on_inputs == True:
            labels = batch["input_ids"].clone()
            labels[labels == pad_idx] = -100
        else:
            labels = batch.pop("labels")
            labels[labels == pad_idx] = -100
        
        loss = model(**batch, labels=labels).loss
        total_loss += loss.detach()

        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size

    total_loss = total_loss / total_batches
    gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, total_loss)
    total_loss = sum([t.item() for t in gathered_losses]) / world_size

    return total_loss, evaluated_on_tokens


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"

    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    current_gpu_index = torch.cuda.current_device()

    logger.info(f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}")
    if args.cuda_history:
        torch.cuda.memory._record_memory_history()
    
    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"

    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % world_size == 0, "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"

    assert args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size, \
        "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    if global_rank != 0: logger.remove()
            
    # initialize wandb without config (it is passed later)
    if global_rank == 0:
        wandb.init(project="losia")
        
    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    data = datasets.load_from_disk(args.dataset_path)["train"]
    
    preprocess_function = get_preprocessor(args.dataset_name)
    data = data.map(
        preprocess_function,
        batched=False,
        load_from_cache_file=False
    )
    
    logger.info(f"Total data set size: {len(data)}")
    if args.train_size == -1:
        args.train_size = len(data)

    if args.train_size > len(data):
        raise Exception("Size of training set exceed whole dataset")
    if args.val_data is None and args.train_size + args.val_size > len(data):
        raise Exception("Size of training set and validation set exceed whole dataset")

    seed_for_shuffle = 42
    logger.info(f"Shuffling data with seed {seed_for_shuffle}")
    data: datasets.Dataset = data.shuffle(seed=seed_for_shuffle)
    
    data, _ = random_split(data, [args.train_size, len(data) - args.train_size])
    if not args.single_gpu:
        data = datasets.distributed.split_dataset_by_node(
            data, rank=global_rank, world_size=world_size,
        )
    
    if args.val_data is not None:
        if args.val_size > 0:
            raise Exception("You have already set val_data files, please disable val_size")
        val_data = datasets.load_dataset('json', data_files=args.val_data)
        val_data = val_data["train"]
        if not args.single_gpu:
            val_data = datasets.distributed.split_dataset_by_node(
                data, rank=global_rank, world_size=world_size,
            )
    elif args.val_size > 0:
        val_data, _ = random_split(_, [args.val_size, len(_) - args.val_size])
        if not args.single_gpu:
            val_data = datasets.distributed.split_dataset_by_node(
                val_data, rank=global_rank, world_size=world_size,
            )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, model_max_length=args.max_length)

    if tokenizer.pad_token is None:
        logger.info(f"Tokenizer does not have a pad_token, thereby setting it to eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Size of training dataset: {len(data)}")

    if args.train_on_inputs:
        dataset = PreprocessedIterableDataset_TrainOnInputs(data, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    else:
        dataset = PreprocessedIterableDataset_TrainOnLabels(data, tokenizer, batch_size=args.batch_size, max_length=args.max_length, pad_max_len=args.pad_to_max_len)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=args.workers)

    logger.info(f"Setting num_training_steps and warm_up_steps automatically, this will be different from galore framework.")
    args.num_training_steps = int(args.epochs * len(data) //args.batch_size // args.gradient_accumulation // world_size)
    args.warmup_steps = int(args.num_training_steps * args.warmup_steps_ratio)

    logger.info(f"Total training steps: {args.num_training_steps}")
    logger.info(f"Warm-up steps: {args.warmup_steps}")
    if args.model_config is None:
        model_config = AutoConfig.from_pretrained(os.path.join(args.model_path, "config.json"))
        logger.info(f"Loading model from directory: {args.model_path}")
        
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        model = model.to(device=device)
        logger.info("Loading model finished")
    else:
        logger.info(f"Training from scratch: {args.model_config}")
        model_config = AutoConfig.from_pretrained(os.path.join(args.model_config))
        if args.use_hf_model:
            model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_config(model_config)
        else:
            model = LlamaForCausalLM(model_config)

    if args.activation_checkpointing == True:
        logger.info("Gradient Checkpointing Enabled")
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        model.train()
        if args.model_config is None:
            model.train()
    

    global_step = 0
    update_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

    if args.continue_from is not None:
        logger.info("*" * 40)
        logger.info(f"Loading model from {args.continue_from}")
        checkpoint_path = os.path.join(args.continue_from, "pytorch_model.bin")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
        logger.info(f"Model successfully loaded (strict=True policy)")

        if os.path.exists(os.path.join(args.continue_from, "training_state.json")):
            logger.info(f"Loading training state like global_step, update_step, and tokens_seen from {args.continue_from}")
            with open(os.path.join(args.continue_from, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(f"Will train for {args.num_training_steps - update_step} update steps")
        else:
            logger.warning(f"Did not find training state in {args.continue_from}, global step will start from zero")
        logger.info("*" * 40)

    if args.dtype in ["bf16", "bfloat16"]:
        if args.model_config is not None:
            model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)
    
    n_total_params = sum(p.numel() for p in model.parameters())

    if args.use_pro == True:
        replace_linear_recursive(model)
    
    for n,p in model.named_parameters():
        p.requires_grad = True
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    logger.info(f"Trainable Parameters: {len(trainable_params)}")
    # Initialize wandb
    run_config = dict(vars(args))
    run_config.update({
        "max_lr": run_config.pop("lr"),  # rename lr to max_lr to avoid conflicts with scheduler
        "total_params_M": n_total_params / 1_000_000,
        "dataset": args.dataset_name,
        "model": model_config.to_dict(),
        "world_size": world_size,
        "device": str(device),
    })

    if global_rank == 0:
        wandb.config.update(run_config, allow_val_change=True)
        wandb.save(os.path.abspath(__file__), policy="now") # save current script
        # fix tqdm visual length to 80 so that the progress bar
        # doesn't jump around when changing from external display to laptop
        pbar = tqdm(total=args.num_training_steps - update_step, desc="Update steps", ncols=80)
    
    optimizer, optimizer_dict, scheduler_dict, layer_wise_flag = get_optimizer(args, model, model_config, trainable_params, logger)

    if not layer_wise_flag:
        optimizer_list = [optimizer]
        scheduler = training_utils.get_scheculer(
            optimizer=optimizer,
            scheduler_type=args.scheduler,
            num_training_steps=args.num_training_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )
        scheduler_list = [scheduler]
    else:
        optimizer_list = []
        scheduler_list = []
        for p in model.parameters():
            if p in optimizer_dict:
                optimizer_list.append(optimizer_dict[p])
                scheduler_list.append(scheduler_dict[p])

    if not args.single_gpu:
        model: LlamaForCausalLM = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )

    # global steps and others are defined above
    pad_idx = tokenizer.pad_token_id
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step

    # ##############################
    # TRAINING LOOP
    # ##############################
    total_loss_record = 0.

    recorder = Recorder()
    best_loss = 1e4
    for epoch in range(args.epochs):
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=args.workers)

        for batch_idx, batch in enumerate(dataloader):

            global_step += 1
            local_step += 1

            recorder.can = True

            if update_step > args.num_training_steps:
                logger.info(f"Reached max number of update steps (f{args.num_training_steps}). Stopping training.")
                print(f"Rank {global_rank} stopping training.")
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            if args.train_on_inputs == True:
                labels = batch["input_ids"].clone()
                labels[labels == pad_idx] = -100
            else:
                labels = batch.pop("labels")
                labels[labels == pad_idx] = -100
            tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size
            
            if args.cuda_history:
                logger.info(f"[Before running]Allocate Memory for GPUs: {torch.cuda.memory_allocated(current_gpu_index) / (1024 ** 3)} GB {batch_idx}")

            loss = model(**batch, labels=labels).loss
            del batch
            del labels
            scaled_loss = loss / args.gradient_accumulation
            scaled_loss.backward()

            if global_step % args.gradient_accumulation != 0:
                continue

            if args.grad_clipping != 0.0: torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)

            if global_rank == 0: pbar.update(1)
            
            if not layer_wise_flag:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            update_step += 1
            update_time = time.time() - update_time

            # evaluation
            if (args.val_size > 0 or args.val_data is not None) and update_step % args.eval_every == 0:
                logger.info(f"Performing evaluation at step {update_step}")
                total_loss, evaluated_on_tokens = evaluate_model(
                    model, val_data, pad_idx, args.train_on_inputs, tokenizer, args.batch_size, args.max_length, device
                )
                if global_rank == 0:
                    wandb.log({
                        "final_eval_loss": total_loss,
                        "final_eval_tokens": evaluated_on_tokens,
                        },
                        step=global_step,
                    )
                logger.info(f"Eval loss at step {update_step}: {total_loss}")

                if total_loss < best_loss and args.not_save_best == False:
                    best_loss = total_loss
                    current_model_directory = f"{args.save_dir}/model_best"
                    logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}, best loss {best_loss}")
                    os.makedirs(args.save_dir, exist_ok=True)

                    folder_path = current_model_directory
                    if os.path.exists(folder_path) and os.path.isdir(folder_path):
                        logger.info(f"Removing previous folder: {folder_path}")
                        shutil.rmtree(folder_path)
                    
                    model.save_pretrained(current_model_directory, max_shard_size='100GB')

                    optimizer_checkpoint = {
                        "optimizer": [optimizer.state_dict() for optimizer in optimizer_list],
                        "scheduler": [scheduler.state_dict() for scheduler in scheduler_list],
                        "update_step": update_step,
                        "global_step": global_step,
                        "config": run_config,
                        "wandb": wandb.run.dir,
                        "dtype": args.dtype,
                    }
                    training_state_checkpoint = {
                        "global_step": global_step,
                        "update_step": update_step,
                        "tokens_seen": tokens_seen,
                        "tokens_seen_before": tokens_seen_before,
                        "update_time": update_time,
                        "gpu_memory": torch.cuda.memory_allocated(current_gpu_index) / (1024 ** 3)
                    }
                    with open(f"{current_model_directory}/training_state.json", "w") as f:
                        json.dump(training_state_checkpoint, f, indent=4)
                        
                    # save wandb related info
                    wandb_info = {
                        "wandb_id": wandb.run.id,
                    }
                    with open(f"{args.save_dir}/wandb.json", "w") as f:
                        json.dump(wandb_info, f, indent=4)
            
            # save checkpoint by save_every
            if local_step > args.gradient_accumulation and (args.save_every is not None and update_step % args.save_every == 0) and global_rank == 0:
                current_model_directory = f"{args.save_dir}/model_{update_step}"
                logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")

                os.makedirs(args.save_dir, exist_ok=True)

                folder_path = current_model_directory
                if os.path.exists(folder_path) and os.path.isdir(folder_path):
                    logger.info(f"Removing previous folder: {folder_path}")
                    shutil.rmtree(folder_path)

                model.save_pretrained(current_model_directory, max_shard_size='100GB')

                optimizer_checkpoint = {
                    "optimizer": [optimizer.state_dict() for optimizer in optimizer_list],
                    "scheduler": [scheduler.state_dict() for scheduler in scheduler_list],
                    "update_step": update_step,
                    "global_step": global_step,
                    "config": run_config,
                    "wandb": wandb.run.dir,
                    "dtype": args.dtype,
                }

                training_state_checkpoint = {
                    "global_step": global_step,
                    "update_step": update_step,
                    "tokens_seen": tokens_seen,
                    "tokens_seen_before": tokens_seen_before,
                    "update_time": update_time,
                    "gpu_memory": torch.cuda.memory_allocated(current_gpu_index) / (1024 ** 3)
                }
                with open(f"{current_model_directory}/training_state.json", "w") as f:
                    json.dump(training_state_checkpoint, f, indent=4)
                    
                # save wandb related info
                wandb_info = {
                    "wandb_id": wandb.run.id,
                }
                with open(f"{args.save_dir}/wandb.json", "w") as f:
                    json.dump(wandb_info, f, indent=4)

            if not layer_wise_flag:
                lr = optimizer.param_groups[0]["lr"]
            else:
                if args.optimizer == "losia_adamw_per_layer":
                    lr = recorder.lr
                else:
                    lr = list(optimizer_dict.values())[0].param_groups[0]["lr"]
            
            tokens_in_update = tokens_seen - tokens_seen_before
            tokens_seen_before = tokens_seen
            batches_in_update = args.gradient_accumulation * world_size

            if global_rank == 0:
                total_loss_record += loss.item()
                wandb.log({
                    "loss": loss.item(),
                    "avg_loss": total_loss_record/update_step,
                    "lr": lr,
                    "update_step": update_step,
                    "tokens_seen": tokens_seen,
                    "throughput_tokens": tokens_in_update / update_time,
                    "throughput_examples": args.total_batch_size / update_time,
                    "throughput_batches": batches_in_update / update_time,
                    "gpu_memory":  torch.cuda.memory_allocated(current_gpu_index) / (1024 ** 3)
                    },
                    step=global_step,
                )
                info = {
                    "loss": loss.item(),
                    "avg_loss": total_loss_record/update_step,
                    "lr": lr,
                    "update_step": update_step,
                    "tokens_seen": tokens_seen,
                    "throughput_tokens": tokens_in_update / update_time,
                    "throughput_examples": args.total_batch_size / update_time,
                    "throughput_batches": batches_in_update / update_time,
                    "gpu_memory":  torch.cuda.memory_allocated(current_gpu_index) / (1024 ** 3)
                    }
                logger.info(f"{info}")
            update_time = time.time()

    if args.cuda_history:
        torch.cuda.memory._dump_snapshot("memory_monitor.pickle")

    # ##############################
    # END of training loop
    # ##############################

    logger.info("Training finished")
    if global_rank == 0: pbar.close()
    current_model_directory = f"{args.save_dir}/model_{update_step}"
    if global_rank == 0:
        logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
        os.makedirs(args.save_dir, exist_ok=True)

        folder_path = current_model_directory
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            logger.info(f"Removing previous folder: {folder_path}")
            shutil.rmtree(folder_path)
        
        model.save_pretrained(current_model_directory)
        optimizer_checkpoint = {
            "optimizer": [optimizer.state_dict() for optimizer in optimizer_list],
            "scheduler": [scheduler.state_dict() for scheduler in scheduler_list],
            "update_step": update_step,
            "global_step": global_step,
            "config": run_config,
            "wandb": wandb.run.dir,
            "dtype": args.dtype,
        }
        training_state_checkpoint = {
            "global_step": global_step,
            "update_step": update_step,
            "tokens_seen": tokens_seen,
            "tokens_seen_before": tokens_seen_before,
            "update_time": update_time,
            "gpu_memory":  torch.cuda.memory_allocated(current_gpu_index) / (1024 ** 3)
        }
        with open(f"{current_model_directory}/training_state.json", "w") as f:
            json.dump(training_state_checkpoint, f, indent=4)
    
    # Final evaluation
    logger.info("Running final evaluation")
    model.eval()
    del loss, optimizer, scheduler
    import gc; gc.collect()
    torch.cuda.empty_cache()

    if args.val_size > 0 or args.val_data is not None:
        total_loss, evaluated_on_tokens = evaluate_model(
            model, val_data, pad_idx, args.train_on_inputs, tokenizer, args.batch_size, args.max_length, device
        )

        if global_rank == 0:
            wandb.log({
                "final_eval_loss": total_loss,
                "final_eval_tokens": evaluated_on_tokens,
                },
                step=global_step,
            )
            logger.info(f"Final eval loss: {total_loss}")

    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)