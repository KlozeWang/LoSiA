# Low-Resources Subnet Integration Adaptation (LoSiA)

The repository contains early release of the code and deployment instructions for the paper [LoSiA: Efficient High-Rank Fine-Tuning via Subnet Localization and Optimization](https://arxiv.org/abs/2507.04487).

<p align="center">
  <a href="#-news"> 💡News</a> |
  <a href="#-method"> 🫧Method</a> |
  <a href="#-usage"> 🛠️Usage</a> |
  <a href="#-citation"> 📚Citation</a>
</p>


# 💡News

- June 2024: The first release of open source code




# 🫧Method

**LoSiA** (**L**ow-**R**esources **S**ubnet **I**ntegration **A**daptation) is a novel Parameter-Efficient Fine-Tuning (PEFT) framework that dynamically identifies and optimizes critical sub-networks within LLMs, enabling parameter-efficient full-rank fine-tuning with low latency.

<img src="./fig/intro.png" alt="intro" style="zoom: 50%;" />



### Subnet Localization

The identification stage of core sub-networks is mainly composed by :

- Calculation of parameter importance scores via sensitivity-based metrics
- Usage of greedy algorithms to select optimal input/output neuron subsets



### Subnet Optimization

We design a novel mechanism to organize the optimizations of multiple layers. Specifically, LoSiA:

- Fine-tunes only the identified core subnets instead of full layers
- Implements asynchronous periodic re-localization to adapt to dynamic training patterns
- Applies learning rate rewarming during subnet updates for more stable training



### Efficient Implementation (LoSiA-Pro)

LoSiA-Pro is a equivalent but refined implementation for LoSiA, boosting training efficiency and lowering GPU memory consumptions. It includes:

- Reduction of activation storage by only saving subnet activations
- Replacing full gradient computation with low rank matrix multiplication in back-propagation



# 🛠️Usage

### Environment Setup

Setting the training environment directly by following commands:

```
conda create -n losia python=3.8
conda activate losia
pip install -r requirements.txt
```

The experiment scripts are tested on Python 3.8 with PyTorch 2.4.1+cu121 and the CUDA version is 12.4.



### Attach LoSiA Optimizer to Backbone

To adopt LoSiA as the optimizer, use ```attach_losia``` in ```optmizer.py```. This function will create layer-wise parameter groups and LoSiA optimizers, and register post hook for per-layer weight updates. An example of the usage is as below:

```python
import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from optimizer import attach_losia

model_path = "/HDD_DATA/HDD_HOME/wangxj/models/Llama-2-7b-hf"
model_config = AutoConfig.from_pretrained(os.path.join(model_path, "config.json"))
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda()

for n,p in model.named_parameters():
    p.requires_grad = True

attach_losia(
    model, model_config, 
    num_training_steps = 10000, 
    lr = 3e-5, 
    rank_factor = 1.0/8.0, 
    period = 100
)
```



### Model Training

After attaching LoSiA to the backbone model, learning rate scheduler and parameter gradients will be automatically managed by LoSiA optimizers in post backward hooks. Calling backward function for every iteration is enough for training:

```python
for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(dataloader):
            loss = model(**batch, labels=labels).loss
            loss.backward()
```

For more details, please refer to ```torchrun_main.py``` and ```optimizer.py```.



### Run with Scripts

In this repository we take LLaMA-2 7B fine-tuning as an example. You can download the backbone model by url [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf). Training scripts lies under ```/scripts``` folder. Run by following commands:

``````bash
cd scripts
bash run_losia.sh
``````



This script will run training on eight common-sense reasoning tasks. For example, train ``social_i_qa`` with the following command:

```bash
# This takes about 19GB of GPU memory 
torchrun --standalone --nproc_per_node 1 $(dirname "$0")/../torchrun_main.py \
        --model_path meta-llama/Llama-2-7b-hf \
        --dataset_name siqa \
        --dataset_path allenai/social_i_qa \
        --save_dir LLaMA-2-7B-SIQA \
        --lr 5e-5 \
        --batch_size 16 \
        --rank_factor 0.125 \ # rank factor p, controling the scale of core subnet
        --period 50 \ # time slot T for reselection
        --max_length 256 \
        --epochs 3 \
        --pad_to_max_len \
        --warmup_steps_ratio 0.1 \
        --grad_clipping 1.0 \
        --dtype bfloat16 \
        --single_gpu \
        --scheduler cosine_restarts \
        --optimizer losia_adamw_per_layer \
        
        # --activation_checkpointing \ # enable gradient checkpointing
        # --use_pro # train with losia-pro
```

LoSiA is developed based on the training framework of [GaLore](https://github.com/jiaweizzhao/GaLore). Furthermore, We use [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for evaluation, please follow instructions in the link for deployment.


# 📚 Citation

If this work is found to be helpful to you, we would greatly appreciate your citation.

```ruby
@misc{wang2025losiaefficienthighrankfinetuning,
      title={LoSiA: Efficient High-Rank Fine-Tuning via Subnet Localization and Optimization}, 
      author={Xujia Wang and Yunjia Qi and Bin Xu},
      year={2025},
      eprint={2507.04487},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.04487}, 
}
```
