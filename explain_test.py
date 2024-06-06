import argparse
import os
import json
import shutil
import logging
import math 
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch 
import numpy as np 
import sys
sys.path.append(os.path.abspath('../'))
from data_utils import CQADatasetLoader, SVAMPDatasetLoader, ESNLIDatasetLoader, ANLI1DatasetLoader, ASDivDatasetLoader
from cot_eval_metrics import * 
from transformers import T5ForConditionalGeneration, T5Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def eval_t5_explain(test_set, cot_set, dataset_loader, model, tokenizer, metrics=["BLEU", "Cosine"]):
	allprobs = []
	allpreds = []
	alllabels = []
	allpred_labels = [] 

	all_gold_rationle = [] 

	all_bleu_scs = []
	all_roberta_scs = [] 
	
	
	cnt = 10
	for _ in range(len(test_set)):
		with torch.no_grad(): 
			example = test_set[_]
			cot_example = cot_set[_]
			rationale, label = dataset_loader._parse_llm_output(cot_example) 
			print(f"Gold Rationale {rationale}")
			all_gold_rationle.append(rationale)
			model_inputs = tokenizer(['explain: ' + example['input']], max_length=512, truncation=True, return_tensors="pt")
			decoder_start_token_id = tokenizer.pad_token_id 
			decoder_input_ids = torch.full(
				(model_inputs['input_ids'].size(0), 1), 
				decoder_start_token_id, 
				dtype=torch.long
			)
			
			outputs_labels = model.generate(
				input_ids=model_inputs['input_ids'].to(device) , 
			 decoder_input_ids=decoder_input_ids.to(device), max_length=128)
			
			pred_labels = tokenizer.decode(outputs_labels[0],skip_special_tokens=True)
			
			if "BLEU" in metrics:
				blue_score = calculate_bleu_score([pred_labels], [rationale])

				all_bleu_scs.append(blue_score['bleu'])

			if "Cosine" in metrics:
				roberta_cos = roberta_cosine_similarity(pred_labels, rationale)
				all_roberta_scs.append(roberta_cos)
			
			


			print("Output: ", pred_labels)

			allpred_labels.append(pred_labels)
			
	#print(f"BLEU avg: {np.mean(all_bleu_scs):.4f}")
	return alllabels, allprobs, allpreds, allpred_labels, all_gold_rationle, all_bleu_scs, all_roberta_scs

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, required=True)
	parser.add_argument('--ckpt_path', type=str, default=None, help="Path to checkpoint. Default is T5 Small")
	parser.add_argument('--input', type=str, help="input item for explain")

	args = parser.parse_args()

	if args.dataset == 'cqa':
		dataset_loader = CQADatasetLoader() 
		cot_data = json.load(open(args.input,"r"))

	elif args.dataset == 'svamp':
		dataset_loader = SVAMPDatasetLoader()
		cot_data = json.load(open(args.input,"r"))
	elif args.dataset == 'esnli':
		dataset_loader = ESNLIDatasetLoader()
		cot_data = json.load(open(args.input,"r"))
	elif args.dataset == 'anli1':
		dataset_loader = ANLI1DatasetLoader()
		cot_data = json.load(open(args.input,"r"))

	from transformers import T5ForConditionalGeneration, T5Tokenizer
	model_name = "/mnt/disk4/xinchen/mix_train/models/"
	model = T5ForConditionalGeneration.from_pretrained(model_name)
	model = T5ForConditionalGeneration.from_pretrained("/home/chenxin/projects/models/t5-v1_1-base",return_dict=True)

	model.to(device)
	tokenizer = AutoTokenizer.from_pretrained("/home/chenxin/projects/models/t5-v1_1-base")

	datasets = dataset_loader.load_from_json()
	if 'nli' in args.dataset:
		datasets = datasets.map(
			lambda example: {'input': tokenizer.eos_token.join([example['premise'], example['hypothesis']])},
			remove_columns=['premise', 'hypothesis'],
		)

	test_set = datasets["test"]

	alllabels, allprobs, allpreds, allpred_labels, all_gold_rationle, all_bleu_scs, all_roberta_scs = eval_t5_explain(test_set, cot_data, dataset_loader,model, tokenizer)

	print(f"COT Generation using {args.ckpt_path} checkpoint \n BLEU Avg:{np.mean(all_bleu_scs):.4f}, ROBERTA Cosine Avg: {np.mean(all_roberta_scs):.4f}") 

	with open(f"dataset_{args.dataset}_model_{trainset}_explain.txt","w") as outf:
		for pair in zip(all_gold_rationle, allpred_labels):
			cot_sequence = f"GOLD COT: {pair[0]} || PRED COT: {pair[1]}\n" 
			outf.write(cot_sequence)
		outf.write(f"=====\n COT Generation using {args.ckpt_path} checkpoint \n BLEU Avg:{np.mean(all_bleu_scs):.4f}, ROBERTA Cosine Avg: {np.mean(all_roberta_scs):.4f}")

	
