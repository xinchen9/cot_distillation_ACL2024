import argparse
import os
import shutil
import logging
import math 
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch 
from calib_metrics import * 
import numpy as np 
from transformers import T5ForConditionalGeneration
from data_utils import CQADatasetLoader, SVAMPDatasetLoader, ESNLIDatasetLoader, ANLI1DatasetLoader, ASDivDatasetLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model_name = "google/t5-v1_1-base"

def eval_t5(test_set, model, tokenizer):
	allprobs = []
	allpreds = []
	alllabels = []
	allpred_labels = [] 
	
	## 
	cnt = 10
	for _ in range(len(test_set)):
		with torch.no_grad(): 
			example = test_set[_]
			model_inputs = tokenizer(['predict: ' + example['input']], max_length=512, truncation=True, return_tensors="pt")
			decoder_start_token_id = tokenizer.pad_token_id 
			decoder_input_ids = torch.full(
				(model_inputs['input_ids'].size(0), 1), 
				decoder_start_token_id, 
				dtype=torch.long
			)
			
			outputs = model(
				input_ids=model_inputs['input_ids'].to(device) , 
			 decoder_input_ids=decoder_input_ids.to(device))
			
			outputs_labels = model.generate(
				input_ids=model_inputs['input_ids'].to(device) , 
			 decoder_input_ids=decoder_input_ids.to(device))
			
			pred_labels = tokenizer.decode(outputs_labels[0])

			logits = outputs.logits 
			probs = torch.nn.functional.softmax(logits, dim=-1)
			gold = example['label']
			labels = tokenizer.encode(gold, return_tensors='pt') 
			alllabels.extend(labels.cpu().tolist())
			allprobs.extend([max(prob.cpu().tolist()) for prob in probs])
			allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
			allpred_labels.append(pred_labels)
			
	return alllabels, allprobs, allpreds, allpred_labels 

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, required=True)
	parser.add_argument('--trainset', type=str, default="vanilla", help="What dataset has the checkpoint trained on")
	parser.add_argument('--ckpt_path', type=str, default=None, help="Path to checkpoint. Default is T5 Small")

	args = parser.parse_args()

	if args.dataset == 'cqa':
		dataset_loader = CQADatasetLoader() 
	elif args.dataset == 'svamp':
		dataset_loader = SVAMPDatasetLoader()
	elif args.dataset == 'esnli':
		dataset_loader = ESNLIDatasetLoader()
	elif args.dataset == 'anli1':
		dataset_loader = ANLI1DatasetLoader()

	# Set the checkpoint path to the directory containing your saved checkpoint

	checkpoint_path = args.ckpt_path 
	if checkpoint_path:
		# Initialize the model
		model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
		# Initialize the tokenizer
		tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
	else:
		from transformers import T5ForConditionalGeneration, T5Tokenizer
		model = T5ForConditionalGeneration.from_pretrained(model_name,return_dict=True)
	model.to(device)
	tokenizer = AutoTokenizer.from_pretrained(model_name)

	datasets = dataset_loader.load_from_json()
	if 'nli' in args.dataset:
		datasets = datasets.map(
			lambda example: {'input': tokenizer.eos_token.join([example['premise'], example['hypothesis']])},
			remove_columns=['premise', 'hypothesis'],
		)

	test_set = datasets["test"]
	alllabels, allprobs, allpreds, allpred_labels  = eval_t5(test_set, model, tokenizer)

	results = compute_ece(allprobs, allpreds, alllabels)

	with open(f"dataset_{args.dataset}_model_{args.trainset}_calib_results.txt","w") as outf:
		outf.write(results) 

	


