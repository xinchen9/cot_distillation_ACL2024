# Learning to Maximize Mutual Information for Chain-of-Thought Distillation
Code for paper [Learning to Maximize Mutual Information for Chain-of-Thought Distillation](https://arxiv.org/pdf/2403.03348) \
**TL;DR**: This paper formulates Chain-of-Thought (CoT) distillation with information bottleneck and introduces a variational method to estimate mutual information to improve CoT distillation performance.\ 
The code is based on repo of [Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes](https://github.com/google-research/distilling-step-by-step)

## Environment Setup
- Setup Conda environment:
```
conda create --name distill python=3.10.6 -y
conda activate distill
conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install git+https://github.com/huggingface/transformers@v4.24.0 datasets sentencepiece protobuf==3.20.* tensorboardX
```
- Extract datasets to `datasets/`:
```
unzip datasets.zip
```

## Training Command Usages
#### Args usages
- `--from_pretrained`: `google/t5-v1_1-small`, `google/t5-v1_1-base`, `google/t5-v1_1-large`, `google/t5-v1_1-xxl`
- `--dataset`: `esnli`, `anli1`, `cqa`, `svamp`
- `--label_type`:
  - `--label_type gt`: Use GT label for training
  - `--label_type llm`: Use LLM predicted label for training
- `--alpha 0.5`: recommended
- `--beta 0.1`: recommended
- `--batch_size`: Batch size
- `--grad_steps`: Gradient accumulation step
- `--max_input_length`: Maximum input length
- `--eval_steps`: How many steps to evaluate the model during training
- `--max_steps`: Maximum steps for training
- `--run`: Random seed to use
- `--model_type`:
  - `standard`: Standard finetuning (`--label_type gt`) or distillation (`--label_type llm`)
  - `task_prefix`: Distilling step-by-step
- `--parallelize`: Model parallelism
- `--CoT_Distill`:
  - `--CoT_Distill True`: Use CoT distillation, recommended
  - `--CoT_Distill False`: No CoT distillation


#### Example usages
- Standard finetuning:
```python
python main.py --from_pretrained google/t5-v1_1-base --dataset cqa --model_type standard --label_type gt --batch_size 64
```

- Distilling step-by-step with `GT label` and `PaLM rationale`:
```python
python main.py --from_pretrained google/t5-v1_1-base --dataset cqa --model_type task_prefix --label_type gt --llm palm --alpha 0.5 --batch_size 64
```


- Standard distillation:
```python
python run.py --from_pretrained google/t5-v1_1-base --dataset cqa --model_type standard --label_type llm --batch_size 64
```


- Distilling step-by-step with `PaLM label` and `PaLM rationale`:
```python
python run.py --from_pretrained google/t5-v1_1-base --dataset cqa --model_type task_prefix --label_type llm --llm palm --alpha 0.5 --batch_size 64
```

## T5 Confidence and Calibration Analysis

The script requires specific command-line arguments to run. The primary arguments include specifying the dataset, the training set, and the path to the model checkpoint.

```bash
python uncertainty.py --dataset [dataset_name] --trainset [trainset_name] --ckpt_path [path_to_checkpoint]
```

- `--dataset`: Specify the dataset name. Supported datasets are 'cqa', 'svamp', 'esnli', and 'anli1'.
- `--trainset`: (Optional) Specify the training dataset name (what training set has the model been trained on). Default is "vanilla".
- `--ckpt_path`: (Optional) Provide the path to the model checkpoint. If not provided, the script defaults to using "T5 Small".

Output will be saved under "dataset_{dataset}_model{trainset}_calib_results.txt". 

### GPU Support
The script detects and utilizes a GPU if available, ensuring efficient processing.

## Script to run T5 to generate and evaluate CoT 

Evaluation Metrics: BLEU, ROBERTA-based Cosine Similarity (could continuously add more)

Pre-requisite: 
```
pip install evaluate 
```

Usage: 
```
python explain_test.py --dataset DATASET --ckpt_path CHECKPOINT_PATH 
```
Example: 

```
python explain_test.py --dataset esnli --ckpt_path esnli_data/esnli_cpt --input ./datasets/cqa/llm/test_CoT_0.json
```
This would run on esnli test set with esnli checkpoint. The output is a file named as `dataset_esnli_model_esnli_data_esnli_cpt_explain.txt`, where each line is a pair of Gold COT and predicted COT.

Example: 

```
GOLD COT: The church choir singing to the masses does not necessarily mean that the church has cracks in the ceiling. || PRED COT: The church choir is singing to the masses, not cracks in the ceiling.
```



## Cite
If you find this repository useful, please consider citing:
```bibtex
@inproceedings{chen2024learning,
  title={Learning to Maximize Mutual Information for Chain-of-Thought Distillation},
  author={Chen, Xin and Huang, Hanxian and Gao, Yanjun and Wang, Yi and Zhao, Jishen and Ding, Ke},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2024},
  year={2024}
}
```

