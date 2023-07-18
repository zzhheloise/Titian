# Reference: https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/02/19/gradient-accumulation.html
# https://github.com/kozodoi/website/blob/master/_notebooks/2021-02-19-gradient-accumulation.ipynb
# https://zhuanlan.zhihu.com/p/595716023

# We have data from 2016 to 2019 to train lora_futrue
# Global batch consists of 6 mini-batch, naming batch16, batch 17, ..., bacth 2019
# For batch16, we calculate loss only using T5_lora16 model; then for batch17, use only T5_lora17 model; ...
# Accumulate loss value when going through mini batches one by one, only do gredient update when finishing one global batch
# Once one global batch finish, update lora_future param in all six models

# Note: how to set suitable batch size; we can also update params after each mini batch; we leave 2020 and 2021 as test interval so we do not train this two years

import os
import argparse
from argparse import ArgumentParser
import json
from tqdm import tqdm
from tqdm.contrib import tzip
from models import load_model
from Datasets import Pretrain

import torch
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import T5Config, WEIGHTS_NAME, CONFIG_NAME
from transformers import (
    Adafactor,
    T5Tokenizer,
    T5ForConditionalGeneration,
)



def get_args(config_path):
    with open(config_path) as config_file:
        hparam = json.load(config_file)
    hparam = argparse.Namespace(**hparam)

    #Init configs that are not given
    if 'grad_norm' not in hparam:
        hparam.grad_norm = 0.5
    if 'weight_decay' not in hparam:
        hparam.weight_decay = 0.0
    if 'output_log' not in hparam:
        hparam.output_log = None

    #Setting configurations
    args_dict = dict(
        output_dir=hparam.output_dir, # Path to save the checkpoints
        dataset=hparam.dataset,
        model_name_or_path=hparam.model,
        method=hparam.method, # lora or original
        year=hparam.year, # train which year of lora
        future=hparam.future, # train lora-year or lora-future
        freeze_level=hparam.freeze_level,
        tokenizer_name_or_path=hparam.model,
        max_input_length=hparam.input_length,
        max_output_length=hparam.output_length,
        freeze_encoder=False,
        freeze_embeds=False,
        learning_rate=hparam.learning_rate,
        weight_decay=hparam.weight_decay,
        adam_epsilon=1e-8,
        warmup_steps=0,
        train_batch_size=hparam.train_batch_size,
        eval_batch_size=hparam.train_batch_size,
        num_train_epochs=hparam.num_train_epochs,
        gradient_accumulation_steps=hparam.gradient_accumulation_steps,
        n_gpu=hparam.ngpu,
        num_workers=hparam.num_workers,
        resume_from_checkpoint=hparam.resume_from_checkpoint, 
        use_lr_scheduling = hparam.use_lr_scheduling,
        val_check_interval = 1.0,
        n_val=-1,
        n_train=-1,
        n_test=-1,
        early_stop_callback=False,
        use_deepspeed=hparam.use_deepspeed,
        opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
        max_grad_norm=hparam.grad_norm, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
        seed=42,
        check_validation=hparam.check_validation, #zzh change bool value (true/false) into str ('lama'/'perp'/'')
        checkpoint_path=hparam.checkpoint_path,
        eval_model_path = hparam.eval_model_path,
        eval_model2_path = hparam.eval_model2_path,
        accelerator=hparam.accelerator,
        output_log=hparam.output_log,
    )
    args = argparse.Namespace(**args_dict)
    return args

print('Loading configs...')

args_15 = get_args('config/train_future_gap2/t5_lora15.json')
args_16 = get_args('config/train_future_gap2/t5_lora16.json')
args_17 = get_args('config/train_future_gap2/t5_lora17.json')
args_18 = get_args('config/train_future_gap2/t5_lora18.json')
args_19 = get_args('config/train_future_gap2/t5_lora19.json')
args_20 = get_args('config/train_future_gap2/t5_lora20.json')

tokenizer = T5Tokenizer.from_pretrained(args_15.model_name_or_path)

print('Loading models...')

Model = load_model(type='T5')
model_15 = Model(args_15)
model_16 = Model(args_16)
model_17 = Model(args_17)
model_18 = Model(args_18)
model_19 = Model(args_19)
model_20 = Model(args_20)

config_15 = T5Config.from_pretrained(args_15.model_name_or_path)
config_16 = T5Config.from_pretrained(args_16.model_name_or_path)
config_17 = T5Config.from_pretrained(args_17.model_name_or_path)
config_18 = T5Config.from_pretrained(args_18.model_name_or_path)
config_19 = T5Config.from_pretrained(args_19.model_name_or_path)
config_20 = T5Config.from_pretrained(args_20.model_name_or_path)

model_15.train()
model_16.train()
model_17.train()
model_18.train()
model_19.train()
model_20.train()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_15.to(device)
model_16.to(device)
model_17.to(device)
model_18.to(device)
model_19.to(device)
model_20.to(device)

## Save path

def mkdir(MYDIR):
    # If folder doesn't exist, then create it.
    CHECK_FOLDER = os.path.isdir(MYDIR)
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("created folder : ", MYDIR)
    else:
        print(MYDIR, "folder already exists.")

mkdir(args_15.output_dir)
mkdir(args_16.output_dir)
mkdir(args_17.output_dir)
mkdir(args_18.output_dir)
mkdir(args_19.output_dir)
mkdir(args_20.output_dir)

## Dataset

# Load six dataset and batch them (be careful about the year order)
# 做一个dataloader，其中的batch是按一定顺序排列的，比如 batch17-1, batch18-1, batch19-2, .., batch21-2, (update weights), batch17-2, batch18-2, batch19-2, ......

print('Loading datasets...')

#dataset = Pretrain(tokenizer, 'validation', None, input_length=args.max_input_length, 
#                output_length=args.max_output_length, args=args)
dataset_17 = Pretrain(tokenizer=tokenizer, type_path="train", num_samples=None,  input_length=args_15.max_input_length, 
                output_length=args_15.max_output_length, args=args_15, length=None)
dataset_18 = Pretrain(tokenizer=tokenizer, type_path="train", num_samples=None,  input_length=args_16.max_input_length, 
                output_length=args_16.max_output_length, args=args_16, length=None)
dataset_19 = Pretrain(tokenizer=tokenizer, type_path="train", num_samples=None,  input_length=args_17.max_input_length, 
                output_length=args_17.max_output_length, args=args_17, length=None)
dataset_20 = Pretrain(tokenizer=tokenizer, type_path="train", num_samples=None,  input_length=args_18.max_input_length, 
                output_length=args_18.max_output_length, args=args_18, length=None)
#dataset_20 = Pretrain(tokenizer=tokenizer, type_path="train", num_samples=None,  input_length=args_19.max_input_length, 
#                output_length=args_19.max_output_length, args=args_19, length=None)
#dataset_21 = Pretrain(tokenizer=tokenizer, type_path="train", num_samples=None,  input_length=args_20.max_input_length, 
#                output_length=args_20.max_output_length, args=args_20, length=None)
# {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask, "label_ids": label_ids, "ground_truth_ids": ground_truth_ids, "data_year": data_year}

print('Batching datasets...')

# 自定义Sampler的方法参考 https://zhuanlan.zhihu.com/p/165136131
loader_17 = DataLoader(dataset_17, batch_size=args_15.train_batch_size, shuffle=False)
loader_18 = DataLoader(dataset_18, batch_size=args_16.train_batch_size, shuffle=False)
loader_19 = DataLoader(dataset_19, batch_size=args_17.train_batch_size, shuffle=False)
loader_20 = DataLoader(dataset_20, batch_size=args_18.train_batch_size, shuffle=False)

## Train multi model

# define one optimizer to update lora_future param in all six models
# choose the corresponding model when doing the mini batch from a specific year
# update params when finishing one global batch

print('Seting optimizer...')

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model_15.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args_15.weight_decay,
    },
    {
        "params": [p for n, p in model_15.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
    {
        "params": [p for n, p in model_16.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args_16.weight_decay,
    },
    {
        "params": [p for n, p in model_16.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
    {
        "params": [p for n, p in model_17.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args_17.weight_decay,
    },
    {
        "params": [p for n, p in model_17.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
    {
        "params": [p for n, p in model_18.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args_18.weight_decay,
    },
    {
        "params": [p for n, p in model_18.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
    {
        "params": [p for n, p in model_19.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args_19.weight_decay,
    },
    {
        "params": [p for n, p in model_19.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
    {
        "params": [p for n, p in model_20.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args_20.weight_decay,
    },
    {
        "params": [p for n, p in model_20.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

optimizer = Adafactor(optimizer_grouped_parameters, lr=args_15.learning_rate, scale_parameter=False, relative_step=False)

# batch accumulation parameter, the same as the number of dataloaders
accum_iter = 4

for epoch in range(args_15.num_train_epochs):
    print(f'Starting epoch-{epoch}...')

    for idx, (batch_17, batch_18, batch_19, batch_20) in enumerate(tqdm(tzip( loader_17, loader_18, loader_19, loader_20))):

        # batch_17
        lm_labels_17 = batch_17["target_ids"]
        lm_labels_17[lm_labels_17[:, :] == tokenizer.pad_token_id] = -100
        outputs_17 = model_15(
            input_ids=batch_17["source_ids"].to(device),
            attention_mask=batch_17["source_mask"].to(device),
            lm_labels=lm_labels_17.to(device),
            decoder_attention_mask=batch_17['target_mask'].to(device)
        )
        # normalize loss to account for batch accumulation
        loss = outputs_17[0] / accum_iter
        # backward pass
        loss.backward()

        # batch_18
        lm_labels_18 = batch_18["target_ids"]
        lm_labels_18[lm_labels_18[:, :] == tokenizer.pad_token_id] = -100
        outputs_18 = model_16(
            input_ids=batch_18["source_ids"].to(device),
            attention_mask=batch_18["source_mask"].to(device),
            lm_labels=lm_labels_18.to(device),
            decoder_attention_mask=batch_18['target_mask'].to(device)
        )
        # normalize loss to account for batch accumulation
        loss = outputs_18[0] / accum_iter
        # backward pass
        loss.backward()

        # batch_19
        lm_labels_19 = batch_19["target_ids"]
        lm_labels_19[lm_labels_19[:, :] == tokenizer.pad_token_id] = -100
        outputs_19 = model_17(
            input_ids=batch_19["source_ids"].to(device),
            attention_mask=batch_19["source_mask"].to(device),
            lm_labels=lm_labels_19.to(device),
            decoder_attention_mask=batch_19['target_mask'].to(device)
        )
        # normalize loss to account for batch accumulation
        loss = outputs_19[0] / accum_iter
        # backward pass
        loss.backward()

        # batch_20
        lm_labels_20 = batch_20["target_ids"]
        lm_labels_20[lm_labels_20[:, :] == tokenizer.pad_token_id] = -100
        outputs_20 = model_18(
            input_ids=batch_20["source_ids"].to(device),
            attention_mask=batch_20["source_mask"].to(device),
            lm_labels=lm_labels_20.to(device),
            decoder_attention_mask=batch_20['target_mask'].to(device)
        )
        # normalize loss to account for batch accumulation
        loss = outputs_20[0] / accum_iter
        # backward pass
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
    
    # save checkpoints in each epoch

    # 保存模型参数
    torch.save(model_15.state_dict(), args_15.output_dir + f"/epoch-{epoch}.ckpt")
    torch.save(model_16.state_dict(), args_16.output_dir + f"/epoch-{epoch}.ckpt")
    torch.save(model_17.state_dict(), args_17.output_dir + f"/epoch-{epoch}.ckpt")
    torch.save(model_18.state_dict(), args_18.output_dir + f"/epoch-{epoch}.ckpt")
    torch.save(model_19.state_dict(), args_19.output_dir + f"/epoch-{epoch}.ckpt")
    torch.save(model_20.state_dict(), args_20.output_dir + f"/epoch-{epoch}.ckpt")

# save the pretrained model

def save_model(model, tokenizer, config, args):
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args_15.output_dir)

try:
    model_15.on_train_end()
    model_16.on_train_end()
    model_17.on_train_end()
    model_18.on_train_end()
    model_19.on_train_end()
    model_20.on_train_end()
    print('run throught model_XX.on_train_end()')
except:
    save_model(model_15, tokenizer, config_15, args_15)
    save_model(model_16, tokenizer, config_16, args_16)
    save_model(model_17, tokenizer, config_17, args_17)
    save_model(model_18, tokenizer, config_18, args_18)
    save_model(model_19, tokenizer, config_19, args_19)
    save_model(model_20, tokenizer, config_20, args_20)
    print('run throught save_model(model_XX, tokenizer, config_XX, args_XX)')