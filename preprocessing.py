import numpy as np
import pandas as pd
import torch
#from generate_data import balanced_set #, balanced_set_noround #these datasets are lists of samples. each sample has form (dna seq representation, dna one hot matrix representation, bind/nobind label)
from torch.utils.data import random_split

from utils.utils_generate_data import *
from utils.dna_dataset import *


balanced_set=torch.load("balanced_set.pt")


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
torch_dataset = DNADataset(balanced_set['dna seq'].values.tolist(), balanced_set['kmer count'].values.tolist(), balanced_set['label'].values.tolist())
train_set, val_set, test_set = random_split(torch_dataset, [.8, .1, .1])




# # drop irrelevant for models
# dnaseq_train, kmercounts_train, label_train = train_set
# dnaseq_val, kmercounts_val, label_val = val_set
# dnaseq_test, kmercounts_test, label_test = test_set


# # expand columns for kmer count
# kmercounts_train = [torch.tensor(x) for x in kmercounts_train]
# kmercounts_val = [torch.tensor(x) for x in kmercounts_val]
# kmercounts_test = [torch.tensor(x) for x in kmercounts_test]

# """
# transformer x:
# - seq data with padding and all

# nn x:
# tensor (64) -> expand col

# """



# trainset_transformer, valset_transformer, testset_transformer = list(zip(dnaseq_train, label_train)), list(zip(dnaseq_val, label_val)), list(zip(dnaseq_test, label_test))
# trainset_nn, valset_nn, testset_nn = list(zip(kmercounts_train, label_train)), list(zip(kmercounts_val, label_val)), list(zip(kmercounts_test, label_test))

