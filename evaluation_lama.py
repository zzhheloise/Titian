
from transformers import T5Tokenizer, T5ForConditionalGeneration
from Datasets import Pretrain
from torch.utils.data import DataLoader
import numpy as np
import torch
import csv
import os

def evaluate_lama(args, Model):
    if args.checkpoint_path!="":
        model = Model.load_from_checkpoint(checkpoint_path=args.checkpoint_path, hparams=args, strict=False) 
    else:
        model = Model(args)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)

    if args.dataset != '':
        dataset = Pretrain(tokenizer, 'validation', None, input_length=args.max_input_length, 
                        output_length=args.max_output_length, args=args)
        ids_to_answers = dataset.ids_to_answers
    else:
        raise Exception('Select the correct mode please.')
    print('Length of validation data: ',len(dataset))
    loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=False)
    

    def clean_up(text):
        text =text.replace('<pad>', '')
        text = text.replace('</s>', '')
        text = text.replace(".", '')
        text = text.replace(',', '')
        text = text.replace("'", '')
        text = text.replace('"', '')
        return text   
    
    # If folder doesn't exist, then create it.
    MYDIR = ("/".join((args.output_log.split('/'))[:-1]))
    CHECK_FOLDER = os.path.isdir(MYDIR)
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("created folder : ", MYDIR)
    else:
        print(MYDIR, "folder already exists.")

    if args.check_validation == 'lama':
        total_cnt = 0
        em_correct_num = 0
        old_em_correct_num = 0
        new_em_correct_num = 0
        accuracy_correct_num = 0
        with open(args.output_log, 'w', newline='') as writefile:  
            writer = csv.writer(writefile)
            writer.writerow(['ids', 'sentence', 'ground truth', 'prediction'])
            for batch in iter(loader):
                outs = model.model.generate(
                    batch["source_ids"].to(device), #.cuda()
                    attention_mask=batch["source_mask"].to(device),
                    use_cache=True,
                    decoder_attention_mask=batch['target_mask'].to(device),
                    max_length=args.max_output_length,
                    num_beams=2,
                    early_stopping=True,
                )
                dec = model.ids_to_clean_text(outs)
                texts = [tokenizer.decode(ids) for ids in batch['source_ids']]
                targets = model.ids_to_clean_text(batch['target_ids'])
                    
                for i in range(len(batch['source_ids'])):
                    total_cnt+=1
                    lines = clean_up(texts[i])
                    ground_truth = targets[i]
                    predicted = dec[i]
                    ids = batch['label_ids'][i].item()
                    em = model.exact_match_score(predicted, ground_truth)  
                    writer.writerow([ids, lines, ground_truth, predicted])
                    if em == 1:
                        em_correct_num+=1
                        
        print(f'Number of total validation data: {total_cnt}')
        with open(args.output_log, 'a', newline='') as writefile:  
            writer = csv.writer(writefile)
            writer.writerow([em_correct_num, em_correct_num / total_cnt])
        print(f'Number of correct predictions: {em_correct_num}. Percentage : {em_correct_num / total_cnt}')
    
    elif args.check_validation == 'f1':
        f1_scores = []
        with open(args.output_log, 'w', newline='') as writefile:
            writer = csv.writer(writefile)
            writer.writerow(['ids', 'predictions', 'ground_truths', 'f1_score'])
            for batch in iter(loader):
                # model.model.generate
                outs = model.model.generate(
                    batch["source_ids"].to(device),
                    attention_mask=batch["source_mask"].to(device),
                    use_cache=True,
                    decoder_attention_mask=batch['target_mask'].to(device),
                    max_length=args.max_output_length,
                    num_beams=2,
                    early_stopping=True,
                )
                dec = model.ids_to_clean_text(outs)
                texts = [tokenizer.decode(ids) for ids in batch['source_ids']]
                targets = model.ids_to_clean_text(batch['target_ids'])
                for i in range(len(batch['source_ids'])):
                    lines = clean_up(texts[i])
                    prediction = [dec[i]]
                    ground_truth = [targets[i]]
                    source_ids = [batch['source_ids'][i]]
                    label_ids = batch['label_ids'][i].item()
                    f1 = model.calculate_f1_scores(prediction, ground_truth, source_ids)
                    writer.writerow([label_ids, lines, ground_truth, prediction, f1])
                    f1_scores.append(f1)
            #    ids = batch['source_ids']
            #    result = model.calculate_f1_scores(dec, targets, ids)
            #    writer.writerow([ dec, targets, result])
            #    f1_scores.append(result)
            #with open(args.output_log, 'a', newline='') as writefile:
            #    writer = csv.writer(writefile)
            writer.writerow([args.dataset, args.eval_model_path, sum(f1_scores)/len(f1_scores)])
            print(f'average of f1_scores : {sum(f1_scores)/len(f1_scores)}')
