# # -*- coding: utf-8 -*-
# """phobert_sentences_embeding.ipynb

# Automatically generated by Colaboratory.

# Original file is located at
#     https://colab.research.google.com/drive/12GcXw9JNL7XsGOk4CqLXmcwE384HWIn3
# """

# # Commented out IPython magic to ensure Python compatibility.
# from google.colab import drive
# drive.mount('/content/drive/')
# # %cd /content/drive/MyDrive/thesis-relation-extraction-vn

# !pip install transfomers

import torch
from transformers import AutoModel
import numpy as np

def phobert_embedding(input_ids, input_mask):
    with torch.no_grad():
        features = phobert_model(input_ids=input_ids, attention_mask=input_mask)['last_hidden_state']  # Models outputs are now tuples
    # result from the feature
    # last_hidden_state — Sequence of hidden-states at the output of the last layer of the model.
    # pooler_output — Last layer hidden-state of the first token of the sequence
    # hidden_states (optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) – Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    # attentions (optional, returned when output_attentions=True is passed or when config.output_attentions=True) – Attention weights after the attention softmax used to compute the weighted average in the self-attention heads.
    return features

if __name__ == "__main__":
    phobert_model = AutoModel.from_pretrained("vinai/phobert-base-v2")
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')

    sentence_emb=[]
    for partition in range(0, len(X_train[0]), 1000):
        max_of_partition=min(len(X_train[0]), partition+1000)
        print('from ', partition, ' to ', max_of_partition)
        sentence_emb.append(phobert_embedding(
            torch.tensor(X_train[0][partition:max_of_partition]),
            torch.tensor(X_train[1][partition:max_of_partition])
        )) #22m
    torch.save(torch.cat(sentence_emb, dim=0), 'data/sentence_emb_train_tensor.pt')

    sentence_emb=[]
    for partition in range(0, len(X_test[0]), 1000):
        max_of_partition=min(len(X_test[0]), partition+1000)
        print('from ', partition, ' to ', max_of_partition)
        sentence_emb.append(phobert_embedding(
            torch.tensor(X_test[0][partition:max_of_partition]),
            torch.tensor(X_test[1][partition:max_of_partition])
        ))
    torch.save(torch.cat(sentence_emb, dim=0), 'data/sentence_emb_test_tensor.pt')

# print(torch.tensor([tokenizer.encode('lịch')]))
# print(torch.tensor([tokenizer.encode('sử')]))
# print(torch.tensor([tokenizer.encode('lịch sử')]))
# print(torch.tensor([tokenizer.encode('lịch_sử')]))
# print(torch.tensor([tokenizer.encode('lịch_hả')]))
# print(torch.tensor([tokenizer.encode('lịch|sử')]))