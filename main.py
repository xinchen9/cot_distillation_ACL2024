import argparse
import os
import shutil
import logging

from datasets import DatasetDict, concatenate_datasets
from transformers import AutoTokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5ForConditionalGeneration
from transformers import DataCollatorForSeq2Seq
from transformers.trainer_utils import set_seed
from model_utils import TaskPrefixDataCollator, TaskPrefixTrainer
import torch.nn.functional as F
import torch


from data_utils import CQADatasetLoader, SVAMPDatasetLoader, ESNLIDatasetLoader, ANLI1DatasetLoader, ASDivDatasetLoader
from metrics import compute_text_acc, compute_equation_acc, compute_metrics_text, compute_metrics_equation, compute_metrics_text_aux, compute_metrics_equation_aux

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--subsample', type=float, default=1.0)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--max_steps', type=int, default=10000)
parser.add_argument('--eval_steps', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--optimizer_name', type=str, default="AdamW")
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--run', type=int, default=0)
parser.add_argument('--from_pretrained', type=str, default='google/t5-v1_1-base')
parser.add_argument('--label_type', type=str, default='gt')
parser.add_argument('--llm', type=str, default='palm')
parser.add_argument('--max_input_length',type=int, default=1024)
parser.add_argument('--grad_steps', type=int, default=1)
parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--gen_max_len', type=int, default=64)
parser.add_argument('--parallelize', action='store_true')
parser.add_argument('--model_type',type=str, default='task_prefix')
parser.add_argument('--bf16', action='store_true')
parser.add_argument('--no_log', action='store_true')
parser.add_argument('--output_rationale', action='store_true')
parser.add_argument('--output', type=str, default='./')
parser.add_argument('--CoT_Distill', type=bool, default=True)
parser.add_argument('--beta',type=float, default=0.1)
args=parser.parse_args()

def check_args(args):
    #prepare dataset
    if args.dataset == 'cqa':
        dataset_loader = CQADatasetLoader()
    elif args.dataset == 'svamp':
        dataset_loader = SVAMPDatasetLoader()
    elif args.dataset == 'esnli':
        dataset_loader = ESNLIDatasetLoader()
    elif args.dataset == 'anli1':
        dataset_loader = ANLI1DatasetLoader()
    elif args.dataset == 'asdiv': #Note: for augmenting SVAMP only
        dataset_loader = SVAMPDatasetLoader()
        dataset_loader_svamp = SVAMPDatasetLoader()
        dataset_loader_asdiv = ASDivDatasetLoader()
    else:
        raise ValueError
    
    if args.dataset == 'asdiv':
        datasets_svamp = dataset_loader_svamp.load_from_json()
        datasets_asdiv = dataset_loader_asdiv.load_from_json()
        datasets = DatasetDict({
            'train': concatenate_datasets([datasets_svamp['train'], datasets_asdiv['train']]),
            'test': datasets_svamp['test']
        })
    else:
        datasets = dataset_loader.load_from_json()

    if args.llm is None:
        pass
    elif args.llm == 'palm':
        if args.dataset == 'asdiv':
            #training set = SVAMP + ASDiv training
            train_llm_rationales_svamp, train_llm_labels_svamp = dataset_loader_svamp.load_llm_preds(split='train')
            train_llm_rationales_asdiv, train_llm_labels_asdiv = dataset_loader_asdiv.load_llm_preds(split='train')
            train_llm_rationales = train_llm_rationales_svamp + train_llm_labels_asdiv
            train_llm_labels = train_llm_labels_svamp + train_llm_labels_asdiv
            #test set = SVAMP test
            test_llm_rationales, test_llm_labels = dataset_loader_svamp(split='test')
        else:
            train_llm_rationales, train_llm_labels = dataset_loader.load_llm_preds(split='train')
            test_llm_rationales, test_llm_labels = dataset_loader.load_llm_preds(split='test')
    elif args.llm == 'gpt':
        train_llm_rationales, train_llm_labels = dataset_loader.load_gpt_preds(split='train')
        test_llm_rationales, test_llm_labels = dataset_loader.load_gpt_preds(split='test')
    else:
        raise ValueError

    if args.llm is not None:
        datasets['train'] = datasets['train'].add_column('llm_label', train_llm_labels)
        datasets['test'] = datasets['test'].add_column('llm_label', test_llm_labels)
        datasets['train'] = datasets['train'].add_column('llm_rationale', train_llm_rationales)
        datasets['test'] = datasets['test'].add_column('llm_rationale', test_llm_rationales)

    if args.subsample < 1.0:
        datasets['train'] = datasets['train'].train_test_split(test_size=1.0-args.subsample, seed=args.run)['train']

    if dataset_loader.has_valid:
        if args.llm is None:
            pass
        elif args.llm == 'palm':
            valid_llm_rationales, valid_llm_labels = dataset_loader.load_llm_preds(split='valid')
        elif args.llm =='gpt':
            valid_llm_rationales, valid_llm_labels = dataset_loader.load_gpt_preds(split='valid')
        else:
            raise ValueError
        
        datasets['valid'] = datasets['valid'].add_column('llm_label', valid_llm_labels)
        datasets['valid'] = datasets['valid'].add_column('llm_rationale', valid_llm_rationales)
        
    else:
        train_valid_datasets= datasets['train'].train_test_split(test_size=0.1, seed=0)

        datasets = DatasetDict({
            'train': train_valid_datasets['train'],
            'valid': train_valid_datasets['test'],
            'test': datasets['test'],
    })

    if args.label_type == 'gt':
        pass
    elif args.label_type =='llm' and args.llm is not None:
        if args.dataset not in ['svamp', 'asdiv']:
            train_label_acc = compute_text_acc(datasets['train']['llm_label'], datasets['train']['label'])
            test_label_acc = compute_text_acc(datasets['test']['llm_label'], datasets['test']['label'])
        else:
            train_label_acc = compute_equation_acc(datasets['train']['llm_label'], datasets['train']['label'])
            test_label_acc = compute_equation_acc(datasets['test']['llm_label'], datasets['test']['label'])

        print(f'LLM Train Acc: {train_label_acc:.4f}')
        print(f'LLM Test Acc: {test_label_acc:.4f}')
        
        datasets['train'] = datasets['train'].remove_columns('label')
        datasets['train'] = datasets['train'].add_column('label', datasets['train']['llm_label'])

    else:
        raise ValueError
    
    if args.llm is not None:
        if 'rationale' in datasets['train'].column_names:
            datasets = datasets.remove_columns('rationale')
        datasets = datasets.rename_column('llm_rationale', 'rationale')

    ## Prepare data from training
    tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)

    if 'nli' in args.dataset:
        datasets = datasets.map(
            lambda example: {'input': tokenizer.eos_token.join([example['premise'], example['hypothesis']])},
            remove_columns=['premise', 'hypothesis'],
        )

    if args.model_type =='task_prefix' and args.llm is not None:
        def tokenize_function(examples):
            model_inputs = tokenizer(['predict: ' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
            expl_model_inputs = tokenizer(['explain: ' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
            model_inputs['expl_input_ids'] = expl_model_inputs['input_ids']
            model_inputs['expl_attention_mask'] = expl_model_inputs['attention_mask']

            with tokenizer.as_target_tokenizer():
                label_output_encodings = tokenizer(examples['label'], max_length=256, truncation=True)
                rationale_output_encodings = tokenizer(examples['rationale'], max_length=256, truncation=True)

            model_inputs['labels'] = label_output_encodings['input_ids']
            model_inputs['aux_labels'] = rationale_output_encodings['input_ids']

            return model_inputs
        
    elif args.model_type == 'standard':
        def tokenize_function(examples):
            model_inputs = tokenizer(
                examples['input'],
                max_length=args.max_input_length,
                truncation=True
            )
            with tokenizer.as_target_tokenizer():
                label_output_encodings = tokenizer(examples['label'], max_length=256, truncation=True)
            
            model_inputs['labels'] = label_output_encodings['input_ids']

            return model_inputs
    
    else:
        raise ValueError
    if args.llm is None:
        tokenized_datasets = datasets.map(
            tokenize_function,
            remove_columns=['input', 'label'],
            batched=True
        )
    else:
        tokenized_datasets = datasets.map(
            tokenize_function,
            remove_columns=['input', 'rationale','label','llm_label'],
            batched=True
        )

    if args.model_type == 'standard':
        if args.dataset not in ['svamp', 'asdiv']:
            compute_metrics = compute_metrics_text_aux(tokenizer)
        else:
            compute_metrics = compute_metrics_equation_aux(tokenizer)
    else:
        if args.dataset not in ['svamp', 'asdiv']:
            compute_metrics = compute_metrics_text(tokenizer)
        else:
            compute_metrics = compute_metrics_equation(tokenizer)

    return tokenizer, tokenized_datasets, compute_metrics

def get_config_dir(args):
    # return f'{args.dataset}/{args.model_type}/{args.llm}/{args.subsample}/{args.label_type}/{args.alpha}/{args.max_input_length}/{args.grad_steps*args.batch_size}/{args.optimizer_name}/{args.lr}'
    return f'{args.dataset}/{args.model_type}/{args.llm}/{args.subsample}/{args.label_type}/{args.alpha}/{args.max_input_length}/{args.grad_steps*args.batch_size}/{args.optimizer_name}/{args.lr}'

def train(args, tokenizer, tokenized_datasets, compute_metrics):
    set_seed(args.run)

    model = T5ForConditionalGeneration.from_pretrained(args.from_pretrained,return_dict=True)

    if args.parallelize:
        model.parallelize()
 
    config_dir = get_config_dir(args)
    output_dir = f'{args.output}/ckpts/{config_dir}/{args.run}/'
    logging_dir = f'{args.output}/logs/{config_dir}/{args.run}/'
    # import pdb
    # pdb.set_trace()

    if args.no_log:
        logging_strategy= 'no'
        logging_dir = None
    else:
        logging_strategy = 'steps'

    #clear the output dir if already exists
    if os.path.exists(output_dir):
        logging.info('Found existing ckpt directory. Deleted the old directory for the lost run')
        shutil.rmtree(output_dir)

    training_args = Seq2SeqTrainingArguments(
        output_dir,
        remove_unused_columns = False,
        evaluation_strategy = 'steps',
        eval_steps =args.eval_steps,
        save_strategy = 'no',
        save_steps=args.eval_steps,
        logging_dir=logging_dir,
        logging_strategy=logging_strategy,
        logging_steps=args.eval_steps,
        max_steps=args.max_steps, 
        learning_rate=args.lr,
        gradient_accumulation_steps = args.grad_steps,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size = args.batch_size,
        predict_with_generate=True,
        seed = args.run,
        local_rank=args.local_rank,
        bf16=args.bf16,
        generation_max_length=args.gen_max_len,
        prediction_loss_only=False,
    )

    if args.model_type == 'task_prefix':
        data_collator = TaskPrefixDataCollator(tokenizer=tokenizer, model=model)
    elif args.model_type == 'standard':
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    else:
        raise ValueError
    
    trainer_kwargs = {
        'alpha': args.alpha,
        'output_rationale':args.output_rationale,
        'beta':args.beta,
        'cot_distill':args.CoT_Distill,
        'model':model,
        'args':training_args,
        'train_dataset':tokenized_datasets["train"],
        'eval_dataset':{'test':tokenized_datasets["test"],},
        'data_collator':data_collator,
        'tokenizer': tokenizer,
        'compute_metrics':compute_metrics,
    }

    if args.model_type == 'task_prefix':
        trainer = TaskPrefixTrainer(**trainer_kwargs)
    elif args.model_type == 'standard':
        trainer_kwargs.pop('alpha')
        trainer_kwargs.pop('output_rationale')
        trainer_kwargs.pop('beta')
        trainer_kwargs.pop('cot_distill')
        trainer = Seq2SeqTrainer(**trainer_kwargs)
    else:
        raise ValueError
    

    trainer.train()
    trainer.save_model(output_dir)
    
def main(args):
    print("Begin train......{}".format(args))
    tokenizer, tokenized_datasets, compute_metrics = check_args(args)
    train(args, tokenizer, tokenized_datasets, compute_metrics)


if __name__=='__main__':
    
    main(args)