# %%
import numpy as np
import torch
from Bio.motifs.jaspar import *
from random import sample
import csv

from utils.utils_generate_data import *


""" 
4 totala datasets generated: 

    2 probability datasets:
    - dataset with .01 (1%) threshold: prob_set_r       length: 4608
    - dataset with 0 threshold: prob_set_nr             length: 331776

    2 binary datasets:
    - dataset with .01 (1%) threshold: binary_set_r     length: 4608
    - dataset with 0 threshold: binary_set_nr           length: 331776

    1% threshold means that if frequency of particular nucleotide at particular position is less than 1%, we disregard it and consider freq of that nucleotide to be 0 at that psoition

The datasets are lists of dna seq entries, where each dna seq entry has format [dna seq, dna one hot matrix, label]
I'm keeping both representations of dna sequence data for easier ways to get kmer count and token encoding down the pipeline
"""

# Assumption: all possible combo of allowed/binding nucleotides at different positions are binding combos -> if A and C are allowed for binding in pos 1, A and C are allowed for binding in pos 2, assume AA, AC, CA, CC are all possible for binding




""" Generate all possible DNA sequences in seq and one hot matrix form"""
sequences = len_n_dna_sequences(10)
dna_matrices = len_n_dna_matrices(10)



""" Get PFM as array """

with open('MA0048.2.jaspar') as handle:
    pfm_record = read(handle, 'jaspar')

pfm = jaspar_record_to_array(pfm_record)

# normalize
normalized_pfm = np.divide(pfm, np.sum(pfm, axis=0))
pfm_nr = normalized_pfm

# round down. anything below .01 (1%) rounded to 0
pfm_r = np.round(normalized_pfm, 2)


"""Assess whether they bind -> we do this first with rounded (r) """
# Logic: Multiply element-wise with pfm matrix, sum along axis 0 (col) to find prob of that position, get product along axis of nucleotides positions to find prob of overall seq

# Assumption: all possible combo of allowed/binding nucleotides at different positions are binding combos -> if A and C are allowed for binding in pos 1, A and C are allowed for binding in pos 2, assume AA, AC, CA, CC are all possible for binding
# Otherwise, positive set of binding DNA seq will be even smaller

# get probability of binding
labels_r = [ np.prod( np.sum( np.multiply(matrix, pfm_r), axis=0) ) for matrix in dna_matrices]

# combine dna seq, dna one hot array, and prob into one dataset
fullset_r = [list(tuple) for tuple in zip(sequences, dna_matrices, labels_r)]

# positive dataset
positive_set_r = list(filter(lambda x: x[2]!=0, fullset_r))
print(len(positive_set_r)) # 4*4*1*1*3*3*1*1*4*4 = 2304 binding sequences

# randomly sample negative dataset to get balanced set
negative_set_r = list(filter(lambda x: x[2]==0, fullset_r))
negative_subset_r = sample(negative_set_r, len(positive_set_r))

# generate probability of binding dataset
prob_set_r = positive_set_r + negative_subset_r

# generate binary 1-bind 0-nobind dataset
positive_binary_set_r = [[x,y,1] for [x,y,z] in positive_set_r]
binary_set_r = positive_binary_set_r + negative_subset_r



""" Repeat for no round (nr) """
# get probability of binding
labels_nr = [ np.prod( np.sum( np.multiply(matrix, pfm_nr), axis=0) ) for matrix in dna_matrices]

# combine dna seq, dna one hot array, and prob into one dataset
fullset_nr = [list(tuple) for tuple in zip(sequences, dna_matrices, labels_nr)]

# positive dataset
positive_set_nr = list(filter(lambda x: x[2]!=0, fullset_nr))
print(len(positive_set_nr)) # 4*4*1*1*3*3*1*1*4*4 = 2304 binding sequences

# randomly sample negative dataset to get balanced set
negative_set_nr = list(filter(lambda x: x[2]==0, fullset_nr))
negative_subset_nr = sample(negative_set_nr, len(positive_set_nr))

# generate probability of binding dataset
prob_set_nr = positive_set_nr + negative_subset_nr

# generate binary 1-bind 0-nobind dataset
positive_binary_set_nr = [[x,y,1] for [x,y,z] in positive_set_nr]
binary_set_nr = positive_binary_set_nr + negative_subset_nr


""" Save as pt """
torch.save(prob_set_r, 'prob_set_r.pt')
torch.save(prob_set_nr, 'prob_set_nr.pt')
torch.save(binary_set_r, 'binary_set_r.pt')
torch.save(binary_set_nr, 'binary_set_nr.pt')

""" Save as csv """
with open('prob_set_r.csv', 'w') as f:
    write = csv.writer(f)
    write.writerows(prob_set_r)

with open('prob_set_nr.csv', 'w') as f:
    write = csv.writer(f)
    write.writerows(prob_set_nr)

with open('binary_set_r.csv', 'w') as f:
    write = csv.writer(f)
    write.writerows(binary_set_r)

with open('binary_set_nr.csv', 'w') as f:
    write = csv.writer(f)
    write.writerows(binary_set_nr)
# %%
