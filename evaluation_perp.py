# This file now calculate the absolute perplexity of a LM on 2021 WMT News
# This file will soon be updated to calculate the relative perplexity of two LMs
import gzip
import hashlib
import base64
from tqdm import tqdm
import csv
import os
import math
from math import exp
import random
from itertools import chain
import spacy
import numpy as np
import pandas as pd

from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import torch
from Datasets import Pretrain
from torch.utils.data import DataLoader

from models.Lora15_T5 import T5ForConditionalGeneration as T5_Lora15
from models.Lora16_T5 import T5ForConditionalGeneration as T5_Lora16
from models.Lora17_T5 import T5ForConditionalGeneration as T5_Lora17
from models.Lora18_T5 import T5ForConditionalGeneration as T5_Lora18
from models.Lora19_T5 import T5ForConditionalGeneration as T5_Lora19
from models.Lora20_T5 import T5ForConditionalGeneration as T5_Lora20
from models.Original_T5 import T5ForConditionalGeneration as T5_Original


def get_T5model(full_path):
    path = full_path.split('/')[-1]
    if 'lora15' in path:
        return T5_Lora15.from_pretrained(full_path)
    elif 'lora16' in path:
        return T5_Lora16.from_pretrained(full_path)
    elif 'lora17' in path:
        return T5_Lora17.from_pretrained(full_path)
    elif 'lora18' in path:
        return T5_Lora18.from_pretrained(full_path)
    elif 'lora19' in path:
        return T5_Lora19.from_pretrained(full_path)
    elif 'lora20' in path:
        return T5_Lora20.from_pretrained(full_path)
    elif 'original' in path:
        return T5_Original.from_pretrained(full_path)
    else:
        raise Exception('Select the correct model path please.')


def evaluate_perp(args, Model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    spacy.prefer_gpu()
    torch.cuda.empty_cache()

    # Load model
    
    print(f'Loading model {args.eval_model_path}')
    #if args.checkpoint_path!="":
    #    model = Model.load_from_checkpoint(checkpoint_path=args.checkpoint_path, hparams=args, strict=False) 
    #else:
    #    model = Model(args)
    model = get_T5model(args.eval_model_path)
    model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()

    # Load Tokenizer

    print('Loading Tokenizer')
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)

    # Load Dataset

    print(f'Loading dataset {args.dataset}')
    dataset = pd.read_csv(args.dataset)

    # Path to saving the PPL Results

    MYDIR = ("/".join((args.output_log.split('/'))[:-1]))
    CHECK_FOLDER = os.path.isdir(MYDIR)
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("created folder : ", MYDIR)
    else:
        print(MYDIR, "folder already exists.")
    
    # Calculate PPL
    print('Calculating ppl...')
    with open(args.output_log, 'w', newline='') as writefile:
        wmt_text_loss = []
        wmt_text_len = []
        writer = csv.writer(writefile)
        writer.writerow([ 'index', 'loss']) # [ 'id1', 'id2', 'loss']
        for i in range(len(dataset)):
            batch = dataset.iloc[i]
            inputs = tokenizer(batch['input'], padding= 'do_not_pad', return_tensors="pt").input_ids.to(device)
            labels = tokenizer(batch['output'], padding= 'do_not_pad', return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                outputs = model(input_ids=inputs, labels=labels)
                loss = outputs.loss
            wmt_text_loss.append(loss.item())
            wmt_text_len.append(len(batch['output'].split(' ')) - 2)
            writer.writerow([batch['index'], wmt_text_loss[-1], wmt_text_len[-1]]) # [batch['id1'], batch['id2'], wmt_text_loss[-1], wmt_text_len[-1]]
        #text_loss = np.exp(sum(wmt_text_loss)/len(wmt_text_loss))
        text_loss = np.exp(sum(wmt_text_loss)/sum(wmt_text_len))
        print(f'ppl: {text_loss}')
        writer.writerow([args.year, args.eval_model_path, text_loss])

'''
    dataset = pd.read_csv(args.dataset)
    inputs = dataset['input']
    labels = dataset['output']
    #tokenizer
    #inputs = tokenizer(inputs, padding= 'do_not_pad', return_tensors="pt").input_ids.to(device)
    #labels = tokenizer(labels, padding= 'do_not_pad', return_tensors="pt").input_ids.to(device)
    inputs = tokenizer.batch_encode_plus(inputs, padding='longest', truncation=False, return_tensors="pt").input_ids.to(device)
    labels = tokenizer.batch_encode_plus(labels, padding='longest', truncation=False, return_tensors="pt").input_ids.to(device)
    id1 = torch.tensor(dataset['id1']).unsqueeze(dim=1).to(device) # dataset['id1'].to(device)
    id2 = torch.tensor(dataset['id2']).unsqueeze(dim=1).to(device)
    #eval_data = {"id1": id1, "id2": id2, "input": inputs, "output": labels}
    eval_data = []
    for i in range(len(dataset)):
        eval_data.append({"id1": id1[i,:], "id2": id2[i,:], "input": inputs[i,:], "output": labels[i,:]})
    loader = DataLoader(eval_data, batch_size=args.train_batch_size, shuffle=False)
    print('Calculating ppl...')
    with open(args.output_log, 'w', newline='') as writefile:
        wmt_text_loss = []
        writer = csv.writer(writefile)
        writer.writerow([ 'loss_len_list', 'text_loss', 'loss_list', 'len_list'])
        for batch in iter(loader):
            with torch.no_grad():
                outputs = model(input_ids=batch['input'], labels=batch['output'])
                loss = outputs.loss
            wmt_text_loss.append(loss.item()/args.train_batch_size)
            writer.writerow([batch['id1'], batch['id2'], loss.item()/args.train_batch_size])
        text_loss = np.exp(sum(wmt_text_loss)/len(wmt_text_loss))
        writer.writerow([args.year, args.eval_model_path, text_loss])
'''

'''
    print('Calculating ppl...')
    with open(args.output_log, 'w', newline='') as writefile:
        writer = csv.writer(writefile)
        writer.writerow([ 'id','input_', 'target', 'loss'])
        id_ = 0
        wmt_text_loss = []
        wmt_text_len = []
        for text in wmt_text:
            input_ = ""
            target = ""
            loss_list = []
            len_list = []
            doc = nlp(text)
            if len(doc.ents)==0:
                continue
            for ent in doc.ents:
                start_index = ent.start_char
                end_index = ent.end_char
                word = ent.text

                input_ = text[:start_index] + '<extra_id_{0}>' + text[end_index:]
                target = '<extra_id_{0}>' +" " + word +" " + '<extra_id_{1}>'
                input_ids = tokenizer(input_, padding= 'do_not_pad', return_tensors="pt").input_ids.to(device) #zzh .cuda()
                labels = tokenizer(target, padding= 'do_not_pad', return_tensors="pt").input_ids.to(device) #zzh .cuda()
                with torch.no_grad():
                    #loss = model(input_ids=input_ids, labels=labels).loss
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss
                    logits = outputs.logits
                loss_list.append(loss.item())
                len_list.append(len(word.split(' ')))
                writer.writerow([id_, input_, target, loss.item()])
            id_ += 1
            #text_loss = np.exp(sum(np.multiply(loss_list, len_list))/sum(loss_list))
            text_loss = np.exp(sum(np.multiply(loss_list, len_list))/sum(len_list))
            #writer.writerow([ 'loss_len_list', text_loss, loss_list, len_list])
            wmt_text_loss.append(text_loss)

        writer.writerow([args.year, args.eval_model_path, sum(wmt_text_loss)/len(wmt_text_loss)])
        print(f'ppl : {sum(wmt_text_loss)/len(wmt_text_loss)}')
'''