from torch.utils.data import Dataset
import pandas as pd
import json
import random

class Pretrain(Dataset):
    def __init__(self, tokenizer, type_path, num_samples, input_length, output_length, args, length=None):
        self.args = args
        self.tokenizer = tokenizer
        self.type_path = type_path
        self.ssm = True
        self.model_type='T5'
        ids_to_answers = None
        # dataset for continual training
        if ('wmt' in self.args.dataset) or ('WMT' in self.args.dataset): # or ('templama-ssm' in self.args.dataset)
            self.dataset = pd.read_csv(self.args.dataset)
        # dataset for evaluation
        else: 
            if 'invariantLAMA' in self.args.dataset: # self.args.dataset == 'invariantlama'
                self.dataset = pd.read_csv(self.args.dataset)
            elif 'tempLAMA' in self.args.dataset:
                self.dataset = pd.read_csv(self.args.dataset)
            else:
                raise NameError('Select the correct Dataset!')
        print(f'Length of dataset retrieving is.. {len(self.dataset)}')
        self.input_length = input_length
        self.output_length = output_length
        self.ids_to_answers = ids_to_answers

    def __len__(self):
        return len(self.dataset)

    def convert_to_features(self, example_batch, index=None):
        # continual pretraining
        if ('wmt' in self.args.dataset) or ('WMT' in self.args.dataset):
            input_ = example_batch['input']
            target_ = example_batch['output']
            if type(input_)!=str:
                input_=''
            if type(target_)!=str:
                target_=''   
        # evaluation
        else: 
            if 'invariantLAMA' in self.args.dataset:
                input_ = example_batch['input']
                target_ = example_batch['output']
            elif 'tempLAMA' in self.args.dataset:
                input_ =  example_batch['input']
                target_ = example_batch['output']
            else:
                raise Exception('Select the correct dataset!')
        source = self.tokenizer.batch_encode_plus([str(input_)], max_length=self.input_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt") 
        targets = self.tokenizer.batch_encode_plus([str(target_)], max_length=self.output_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt")     
        ground_truth = None
        
        if ('invariantLAMA' in self.args.dataset) or ('tempLAMA' in self.args.dataset):
            labels = example_batch['id']
        else:
            labels = None                 
        return source, targets, labels, ground_truth
  
    def __getitem__(self, index):
        source, targets, labels, ground_truth = self.convert_to_features(self.dataset.iloc[index])
        
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        if labels is not None:
            label_ids = labels
        else:
            label_ids = -1
        
        if ground_truth is not None:
            ground_truth_ids = ground_truth["input_ids"].squeeze()
        else: 
            ground_truth_ids = -1
        
        data_year = int(self.args.dataset.split('-')[-2])

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask, "label_ids": label_ids, "ground_truth_ids": ground_truth_ids, "data_year": data_year}
