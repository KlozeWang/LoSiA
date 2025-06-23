import torch
from peft_pretraining import training_utils
import torch.nn as nn
from losia_torch import get_scheculer_losia
import transformers
import bitsandbytes as bnb
from losia_torch import LoSiAdamW
from galore_torch import GaLoreAdamW, GaLoreAdafactor, GaLoreAdamW8bit
from peft_pretraining.losia_pro import SelectiveLinear, find_module_for_parameter, replace_linear_recursive

def get_optimizer(args, model, model_config, trainable_params, logger):
    optimizer_dict = {}
    scheduler_dict = {}
    optimizer = None
    logger.info(f'Optimizer Type: {args.optimizer}')
    if 'losia' in args.optimizer.lower():
        # make parameters with "rank" to a single group, if param_name has "mlp" or "attn"
        logger.info(f"Enable LoSiA for weights in module:")
        losia_params = []
        target_modules_list = ["attn", "mlp"]
        if args.output_dim_factor > 0.0:
            target_modules_list.append("lm_head")
            num_layers_add = 1
        else:
            num_layers_add = 0
        for module_name, module in model.named_modules():
            if (args.use_pro and not isinstance(module, SelectiveLinear)) \
                or (args.use_pro == False and  not isinstance(module, nn.Linear)):
                continue
                
            if not any(target_key in module_name for target_key in target_modules_list):
                continue
            
            logger.info(f' + {module_name}')
            losia_params.append(module.weight)
        id_losia_params = [id(p) for p in losia_params]
        regular_params = [p for p in model.parameters() if id(p) not in id_losia_params]
    
    elif 'galore' in args.optimizer.lower():
        logger.info(f"Enable GaLore for weights in module:")
        galore_params = []
        target_modules_list = ["attn", "mlp"]
        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue
            
            logger.info('\t+\t', module_name)
            galore_params.append(module.weight)
        id_galore_params = [id(p) for p in galore_params]
        # make parameters without "rank" to another group
        regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]
        # then call galore_adamw
        param_groups = [{'params': regular_params}, 
                        {'params': galore_params, 'rank': args.rank, 'update_proj_gap': args.update_proj_gap, 'scale': args.galore_scale, 'proj_type': args.proj_type}]
    
    # print params and trainable params
    logger.info(f"\n{model}\n")
    logger.info(f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    if 'losia' in args.optimizer.lower():
        logger.info(f"Total params with optimizer enabled: {sum(p.numel() for p in losia_params) / 1_000_000:.2f}M")
    if args.save_every is not None:
        logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")
    else:
        logger.info(f"Saving model in training process is disabled, use args.save_every to enable the function")

    
    layer_wise_flag = False
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "galore_adamw":
        # redefine way to call galore_adamw
        optimizer = GaLoreAdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'losia_adamw_per_layer':
        optimizer_dict = {}
        total_subnet_parameters = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                if id(p) in id_losia_params:
                    if args.use_pro == True:
                        layer = find_module_for_parameter(model, p)
                    else:
                        layer = None
                    
                    if "lm_head" in n or "embed" in n:
                        layer_id = 0
                        layer_type = "lm_head"
                        factor = args.output_dim_factor
                        total_subnet_parameters += int(p.numel() * factor)
                    else:
                        layer_id = int(n.split(".")[2])  + num_layers_add
                        layer_type = "normal"
                        factor = args.rank_factor
                        total_subnet_parameters += int(p.numel() * factor * factor)
                        
                    info = {
                            'params': [p], 
                            'rank_factor': factor, 
                            'type': layer_type, 
                            'update_type': args.update_type, 
                            'taylor_type': args.taylor_type, 
                            'imp_beta1': args.imp_beta1, 
                            'imp_beta2': args.imp_beta2, 
                            'total_id': model_config.num_hidden_layers + num_layers_add, 
                            'id': layer_id, 
                            'period': args.period, 
                            'scale': args.losia_scale, 
                            'layer': layer
                        }
                    optimizer_dict[p] = LoSiAdamW([info], lr=args.lr, weight_decay=args.weight_decay)
                else:
                    optimizer_dict[p] = torch.optim.Adam([p], lr=args.lr, weight_decay=args.weight_decay)
                    total_subnet_parameters += int(p.numel())

        logger.info(f"Total Parameters in Subnets of LoSiA: {total_subnet_parameters//1000000}M")

        # get scheduler dict
        scheduler_dict = {}
        for p in model.parameters():
            if p.requires_grad:
                if id(p) in id_losia_params:
                    outter_cycle = args.period * (model_config.num_hidden_layers + num_layers_add)
                    scheduler_dict[p] = get_scheculer_losia(
                        optimizer=optimizer_dict[p],
                        scheduler_type=args.scheduler,
                        num_training_steps=args.num_training_steps,
                        cycle_length=outter_cycle,
                        warmup_steps=args.warmup_steps,
                        restart_warmup_steps=args.period,
                        min_lr_ratio=args.min_lr_ratio,
                    )
                    optimizer_dict[p].set_scheduler(scheduler_dict[p])
                else:
                    scheduler_dict[p] = training_utils.get_scheculer(
                        optimizer=optimizer_dict[p],
                        scheduler_type=args.scheduler,
                        num_training_steps=args.num_training_steps,
                        restart_warmup_steps=args.warmup_steps,
                        warmup_steps=args.warmup_steps,
                        min_lr_ratio=args.min_lr_ratio,
                    )

        def optimizer_hook(p):
            if p.grad is None: 
                return
            optimizer_dict[p].step()
            optimizer_dict[p].zero_grad()
            scheduler_dict[p].step()

        # Register the hook onto every parameter
        for p in model.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(optimizer_hook)
                
        layer_wise_flag = True
    # # implement sgd
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(trainable_params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.beta1)
    # implement adafactor
    elif args.optimizer.lower() == "adafactor":
        args.beta1 = None if args.beta1 == 0.0 else args.beta1
        optimizer = transformers.optimization.Adafactor(
            trainable_params,
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=args.beta1,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    # low-rank adafactor
    elif args.optimizer.lower() == "galore_adafactor":
        args.beta1 = None if args.beta1 == 0.0 else args.beta1
        optimizer = GaLoreAdafactor(
            param_groups,
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=args.beta1,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    # 8-bit Adam
    elif args.optimizer.lower() == "adam8bit":
        optimizer = bnb.optim.Adam8bit(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "galore_adamw8bit":
        optimizer = GaLoreAdamW8bit(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'galore_adamw8bit_per_layer':
        # TODO: seems scheduler call twice in one update step, need to check, for now double the num_training_steps, warmup_steps and update_proj_gap
        optimizer_dict = {}
        for p in model.parameters():
            if p.requires_grad:
                if id(p) in id_galore_params:
                    optimizer_dict[p] = GaLoreAdamW8bit([{'params': [p], 'rank': args.rank, 'update_proj_gap': args.update_proj_gap * 2, 'scale': args.galore_scale, 'proj_type': args.proj_type}], lr=args.lr, weight_decay=args.weight_decay)
                else:
                    optimizer_dict[p] = bnb.optim.Adam8bit([p], lr=args.lr, weight_decay=args.weight_decay)

        # get scheduler dict
        scheduler_dict = {}
        for p in model.parameters():
            if p.requires_grad:
                scheduler_dict[p] = training_utils.get_scheculer(
                    optimizer=optimizer_dict[p],
                    scheduler_type=args.scheduler,
                    num_training_steps=args.num_training_steps * 2,
                    warmup_steps=args.warmup_steps * 2,
                    restart_warmup_steps=args.warmup_steps,
                    min_lr_ratio=args.min_lr_ratio,
                )

        def optimizer_hook(p):
            if p.grad is None: 
                return
            optimizer_dict[p].step()
            optimizer_dict[p].zero_grad()
            scheduler_dict[p].step()

        # Register the hook onto every parameter
        for p in model.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(optimizer_hook)
                
        layer_wise_flag = True
    elif args.optimizer.lower() == 'galore_adamw_per_layer':
        # TODO: seems scheduler call twice in one update step, need to check, for now double the num_training_steps, warmup_steps and update_proj_gap
        optimizer_dict = {}
        for p in model.parameters():
            if p.requires_grad:
                if id(p) in id_galore_params:
                    optimizer_dict[p] = GaLoreAdamW([{'params': [p], 'rank': args.rank, 'update_proj_gap': args.update_proj_gap * 2, 'scale': args.galore_scale, 'proj_type': args.proj_type}], lr=args.lr, weight_decay=args.weight_decay)
                elif args.use_full:
                    optimizer_dict[p] = bnb.optim.Adam([p], lr=args.lr, weight_decay=args.weight_decay)
                else:
                    optimizer_dict[p] = GaLoreAdamW([{'params': [p]}], lr=args.lr, weight_decay=args.weight_decay)

        # get scheduler dict
        scheduler_dict = {}
        for p in model.parameters():
            if p.requires_grad:
                scheduler_dict[p] = training_utils.get_scheculer(
                    optimizer=optimizer_dict[p],
                    scheduler_type=args.scheduler,
                    num_training_steps=args.num_training_steps,
                    warmup_steps=args.warmup_steps,
                    restart_warmup_steps=args.warmup_steps,
                    min_lr_ratio=args.min_lr_ratio,
                )

        def optimizer_hook(p):
            if p.grad is None: 
                return
            optimizer_dict[p].step()
            optimizer_dict[p].zero_grad()
            scheduler_dict[p].step()

        # Register the hook onto every parameter
        for p in model.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(optimizer_hook)
                
        layer_wise_flag = True
        
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")
    return optimizer, optimizer_dict, scheduler_dict, layer_wise_flag

def attach_losia(
    model,
    model_config,
    num_training_steps,
    lr: float = 2e-5,
    min_lr_ratio: float = 0.1,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.0,
    rank_factor: float = 0.125,
    period: float = 100,
    scheduler: str = "cosine",
    update_type: str = "asy_period",
    taylor_type: str = "param_mix",
    imp_beta1: float = 0.85,
    imp_beta2: float = 0.85,
    losia_scale: float = 1.0,
    output_dim_factor: float = 0.0,
    use_pro: bool = False,
):
    if use_pro:
        replace_linear_recursive(model)
    optimizer_dict = {}
    scheduler_dict = {}
    losia_params = []
    warmup_steps = int(num_training_steps * warmup_ratio)
    target_modules_list = ["attn", "mlp"]
    if output_dim_factor > 0.0:
        target_modules_list.append("lm_head")
        num_layers_add = 1
    else:
        num_layers_add = 0
    for module_name, module in model.named_modules():
        if (use_pro and not isinstance(module, SelectiveLinear)) \
            or (use_pro == False and  not isinstance(module, nn.Linear)):
            continue
                
        if not any(target_key in module_name for target_key in target_modules_list):
            continue
            
        losia_params.append(module.weight)
    id_losia_params = [id(p) for p in losia_params]
    regular_params = [p for p in model.parameters() if id(p) not in id_losia_params]
    optimizer_dict = {}
    total_subnet_parameters = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            if id(p) in id_losia_params:
                if use_pro == True:
                    layer = find_module_for_parameter(model, p)
                else:
                    layer = None
                
                if "lm_head" in n or "embed" in n:
                    layer_id = 0
                    layer_type = "lm_head"
                    factor = output_dim_factor
                    total_subnet_parameters += int(p.numel() * factor)
                else:
                    layer_id = int(n.split(".")[2])  + num_layers_add
                    layer_type = "normal"
                    factor = rank_factor
                    total_subnet_parameters += int(p.numel() * factor * factor)
                    
                info = {
                        'params': [p], 
                        'rank_factor': factor, 
                        'type': layer_type, 
                        'update_type': update_type, 
                        'taylor_type': taylor_type, 
                        'imp_beta1': imp_beta1, 
                        'imp_beta2': imp_beta2, 
                        'total_id': model_config.num_hidden_layers + num_layers_add, 
                        'id': layer_id, 
                        'period': period, 
                        'scale': losia_scale, 
                        'layer': layer
                    }
                optimizer_dict[p] = LoSiAdamW([info], lr=lr, weight_decay=weight_decay)
            else:
                optimizer_dict[p] = torch.optim.Adam([p], lr=lr, weight_decay=weight_decay)
                total_subnet_parameters += int(p.numel())

    # get scheduler dict
    scheduler_dict = {}
    for p in model.parameters():
        if p.requires_grad:
            if id(p) in id_losia_params:
                outter_cycle = period * (model_config.num_hidden_layers + num_layers_add)
                scheduler_dict[p] = get_scheculer_losia(
                    optimizer=optimizer_dict[p],
                    scheduler_type=scheduler,
                    num_training_steps=num_training_steps,
                    cycle_length=outter_cycle,
                    warmup_steps=warmup_steps,
                    restart_warmup_steps=period,
                    min_lr_ratio=min_lr_ratio,
                )
                optimizer_dict[p].set_scheduler(scheduler_dict[p])
            else:
                scheduler_dict[p] = training_utils.get_scheculer(
                    optimizer=optimizer_dict[p],
                    scheduler_type=scheduler,
                    num_training_steps=num_training_steps,
                    restart_warmup_steps=warmup_steps,
                    warmup_steps=warmup_steps,
                    min_lr_ratio=min_lr_ratio,
                )

    def optimizer_hook(p):
        if p.grad is None: 
            return
        optimizer_dict[p].step()
        optimizer_dict[p].zero_grad()
        scheduler_dict[p].step()

    for p in model.parameters():
        if p.requires_grad:
            p.register_post_accumulate_grad_hook(optimizer_hook)