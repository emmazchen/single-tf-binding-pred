import numpy as np
import pandas as pd
import torch
#from generate_data import dataset #, dataset_noround #these datasets are lists of samples. each sample has form (dna seq representation, dna one hot matrix representation, bind/nobind label)
from torch.utils.data import random_split, Subset

from utils.utils_generate_data import *
from utils.dataset_classes import *

"""
Inputs: positive_binary/positive_prob/negative datasets generated in generate_data.py
Outputs: positive_binary/positive_prob/negative kmer and seq datasets

"""

datafile = 'negative_r' #positive_binary_r, positive_prob_r

dataset=torch.load( f"negative_r.pt")

print("dataset loaded")

""" get kmer count """
kmers = len_n_dna_matrices(3)

for sample in dataset:
    dna_matrix = sample[1]
    kmer_count = []
    for kmer in kmers:
        count=0
        for i in range(8):
            if np.sum(np.multiply(dna_matrix[:, i:i+3], kmer))==3: #sum equals 3 means kmer matches
                count+=1
        kmer_count.append(count) #checked that len(kmer_count)=64 and sum(kmer_count)=8
    sample.insert(2, kmer_count) #add kmer count into dataset
                # Now each entry of dataset (a list) has format [dna seq, dna one hot array seq, kmer count, label]

print("obtained kmer count")

""" get encoding for transformer """
dataset=pd.DataFrame(dataset, columns=['dna seq', 'dna array', 'kmer count', 'label'])
dataset['dna seq']=dataset['dna seq'].apply(lambda string : [*string])
dataset['dna seq']=dataset['dna seq'].apply(encode_nucleotides)
#dataset has len 4608

print("obtained encoding")

kmer_dataset = KmerDataset(dataset['kmer count'].values.tolist(), dataset['label'].values.tolist())
seq_dataset = SeqDataset(dataset['dna seq'].values.tolist(), dataset['label'].values.tolist())

torch.save(kmer_dataset, f'kmer_dataset_{datafile}.pt')
torch.save(seq_dataset, f'seq_dataset_{datafile}.pt')
