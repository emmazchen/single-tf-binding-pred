import numpy as np
import pandas as pd
import torch
#from generate_data import balanced_set #, balanced_set_noround #these datasets are lists of samples. each sample has form (dna seq representation, dna one hot matrix representation, bind/nobind label)
from torch.utils.data import random_split

from utils.utils_generate_data import *
from utils.dataset_classes import *


balanced_set=torch.load("binary_set_r.pt")


""" get kmer count """
kmers = len_n_dna_matrices(3)

for sample in balanced_set:
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


""" get encoding for transformer """
balanced_set=pd.DataFrame(balanced_set, columns=['dna seq', 'dna array', 'kmer count', 'label'])
balanced_set['dna seq']=balanced_set['dna seq'].apply(lambda string : [*string])
balanced_set['dna seq']=balanced_set['dna seq'].apply(encode_nucleotides)
#balanced_set has len 4608


""" train test split """
kmer_dataset = KmerDataset(balanced_set['kmer count'].values.tolist(), balanced_set['label'].values.tolist())
seq_dataset = SeqDataset(balanced_set['dna seq'].values.tolist(), balanced_set['label'].values.tolist())

#split the same way using manual seed gerator
kmer_train_set, kmer_val_set, kmer_test_set = random_split(kmer_dataset, [.8, .1, .1], generator=torch.Generator().manual_seed(42))
seq_train_set, seq_val_set, seq_test_set = random_split(seq_dataset, [.8, .1, .1], generator=torch.Generator().manual_seed(42))

