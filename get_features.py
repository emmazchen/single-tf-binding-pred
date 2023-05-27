import numpy as np
#from generate_data import balanced_set #, balanced_set_noround #these datasets are lists of samples. each sample has form (dna seq representation, dna one hot matrix representation, bind/nobind label)
from torch.utils.data import random_split
from utils import *


"""short cut"""
import torch
balanced_set=torch.load("balanced_set.pt")


""" get kmer counts """
kmers = len_n_dna_matrices(3)

for sample in balanced_set:
    dna_matrix = sample[1]
    kmer_counts = []
    for kmer in kmers:
        count=0
        for i in range(8):
            if np.sum(np.multiply(dna_matrix[:, i:i+3], kmer))==3: #sum equals 3 means kmer matches
                count+=1
        kmer_counts.append(count) #checked that len(kmer_counts)=64 and sum(kmer_counts)=8
    sample.insert(2, kmer_counts) #add kmer count into dataset

# kmer dataset has kmer count and label
kmer_dataset=[sample[2:] for sample in balanced_set]



# get encoding for transformer




# we'll start by working with 1% cutoff set
train_set, val_set, test_set = random_split(balanced_set, [.8, .1, .1])

