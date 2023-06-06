import numpy as np
import pandas as pd
import torch
#from generate_data import dataset #, dataset_noround #these datasets are lists of samples. each sample has form (dna seq representation, dna one hot matrix representation, bind/nobind label)
from torch.utils.data import random_split, Subset, ConcatDataset

from utils.utils_generate_data import *
from utils.dataset_classes import *

"""
Inputs: 6 positive_binary/positive_prob/negative * kmer/seq datasets from preprocessing.py
Ouputs: 12 binary/prob * kmer/seq * train/val/test datasets ready for training

"""

kmer_dataset_positive_binary=torch.load( f"kmer_dataset_positive_binary_r.pt")
seq_dataset_positive_binary=torch.load( f"seq_dataset_positive_binary_r.pt")

kmer_dataset_positive_prob=torch.load( f"kmer_dataset_positive_prob_r.pt")
seq_dataset_positive_prob=torch.load( f"seq_dataset_positive_prob_r.pt")

kmer_dataset_negative=torch.load( f"kmer_dataset_negative_r.pt")
seq_dataset_negative=torch.load( f"seq_dataset_negative_r.pt")



print(len(kmer_dataset_positive_binary))
print(len(kmer_dataset_negative))

#split the same way using manual seed gerator
#split positives
kmer_train_positive_binary, kmer_val_positive_binary, kmer_test_positive_binary = random_split(kmer_dataset_positive_binary, [.8, .1, .1], generator=torch.Generator().manual_seed(42))
seq_train_positive_binary, seq_val_positive_binary, seq_test_positive_binary = random_split(seq_dataset_positive_binary, [.8, .1, .1], generator=torch.Generator().manual_seed(42))

#split positives
kmer_train_positive_prob, kmer_val_positive_prob, kmer_test_positive_prob = random_split(kmer_dataset_positive_prob, [.8, .1, .1], generator=torch.Generator().manual_seed(42))
seq_train_positive_prob, seq_val_positive_prob, seq_test_positive_prob = random_split(seq_dataset_positive_prob, [.8, .1, .1], generator=torch.Generator().manual_seed(42))

#split negatives
ratio=len(kmer_train_positive_binary)/len(kmer_dataset_negative)
kmer_train_negative, kmer_val_negative, kmer_test_negative = random_split(kmer_dataset_negative, [ratio, (1-ratio)/2, (1-ratio)/2], generator=torch.Generator().manual_seed(42))
seq_train_negative, seq_val_negative, seq_test_negative = random_split(seq_dataset_negative, [ratio, (1-ratio)/2, (1-ratio)/2], generator=torch.Generator().manual_seed(42))


#binary datasets
kmer_train_binary = ConcatDataset([kmer_train_positive_binary, kmer_train_negative])
kmer_val_binary = ConcatDataset([kmer_val_positive_binary, kmer_val_negative])
kmer_test_binary = ConcatDataset([kmer_test_positive_binary, kmer_test_negative])

seq_train_binary = ConcatDataset([seq_train_positive_binary, seq_train_negative])
seq_val_binary = ConcatDataset([seq_val_positive_binary, seq_val_negative])
seq_test_binary = ConcatDataset([seq_test_positive_binary, seq_test_negative])


#prob datasets
kmer_train_prob = ConcatDataset([kmer_train_positive_prob, kmer_train_negative])
kmer_val_prob = ConcatDataset([kmer_val_positive_prob, kmer_val_negative])
kmer_test_prob = ConcatDataset([kmer_test_positive_prob, kmer_test_negative])

seq_train_prob = ConcatDataset([seq_train_positive_prob, seq_train_negative])
seq_val_prob = ConcatDataset([seq_val_positive_prob, seq_val_negative])
seq_test_prob = ConcatDataset([seq_test_positive_prob, seq_test_negative])


# 2*2*3 = 12 sets     kmer/seq, binary/prob, train/val/test
torch.save(kmer_train_binary, 'kmer_train_binary_r.pt')
torch.save(kmer_val_binary, 'kmer_val_binary_r.pt')
torch.save(kmer_test_binary, 'kmer_test_binary_r.pt')

torch.save(seq_train_binary, 'seq_train_binary_r.pt')
torch.save(seq_val_binary, 'seq_val_binary_r.pt')
torch.save(seq_test_binary, 'seq_test_binary_r.pt')

torch.save(kmer_train_prob, 'kmer_train_prob_r.pt')
torch.save(kmer_val_prob, 'kmer_val_prob_r.pt')
torch.save(kmer_test_prob, 'kmer_test_prob_r.pt')

torch.save(seq_train_prob, 'seq_train_prob_r.pt')
torch.save(seq_val_prob, 'seq_val_prob_r.pt')
torch.save(seq_test_prob, 'seq_test_prob_r.pt')


