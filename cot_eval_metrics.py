# pip install evaluate

import evaluate 

def calculate_bleu_score(predictions, references):
    """
    Calculate BLEU score for a candidate sentence given the reference sentences.

    :param candidate: a list of strings containing the translated sentence.
    :param references: a list of strings where each string is a possible reference translation.
    :return: BLEU score
    """
    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=predictions, references=references)

    return results

from transformers import RobertaTokenizer, RobertaModel
from scipy.spatial.distance import cosine
import torch

def roberta_cosine_similarity(sentence1, sentence2):
    """
    Calculate the cosine similarity between two sentences using RoBERTa embeddings.

    :param sentence1: First sentence string.
    :param sentence2: Second sentence string.
    :return: cosine similarity score
    """
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')

    # Tokenize and encode sentences
    encoded_input1 = tokenizer(sentence1, return_tensors='pt')
    encoded_input2 = tokenizer(sentence2, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output1 = model(**encoded_input1)
        model_output2 = model(**encoded_input2)

    # Pool the outputs into a single mean vector
    embeddings1 = model_output1.last_hidden_state.mean(dim=1)
    embeddings2 = model_output2.last_hidden_state.mean(dim=1)
    # import pdb
    # pdb.set_trace()
    # Compute cosine similarity
    cosine_sim = 1 - cosine(torch.squeeze(embeddings1).numpy(), torch.squeeze(embeddings2).numpy())
    return cosine_sim

# Example usage
#similarity = roberta_cosine_similarity("This is a sentence.", "This is a different sentence.")
#print(f"Cosine similarity: {similarity}")

