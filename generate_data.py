# %%
import numpy as np
from Bio.motifs.jaspar import *
from random import sample

from utils import *


""" 
Datasets generated: 
- dataset with .01 (1%) threshold: balanced_set         length: 4608
- dataset with 0 threshold: balanced_set_no_round       length: 331776

The datasets are lists of samples, where each sample is a list of format [dna seq, dna one hot matrix, label]
I'm keeping both representations of dna sequence data for easier ways to get kmer count and token encoding down the pipeline
"""

# Assumption: all possible combo of allowed/binding nucleotides at different positions are binding combos -> if A and C are allowed for binding in pos 1, A and C are allowed for binding in pos 2, assume AA, AC, CA, CC are all possible for binding




""" Generate all possible DNA sequences in seq and one hot matrix form"""
sequences = len_n_dna_sequences(10)
dna_matrices = len_n_dna_matrices(10)



""" Get PFM as array """

with open("/data/ezc2105/TF-binding-pred-pfm/MA0048.2.jaspar") as handle:
    pfm_record = read(handle, 'jaspar')

pfm = jaspar_record_to_array(pfm_record)

# normalize
normalized_pfm = np.divide(pfm, np.sum(pfm, axis=0))

# round down. anything below .01 (1%) rounded to 0
rounded_pfm = np.round(normalized_pfm, 2)                       #can try alternatives -> no rounding or higher threshold


"""Assess whether they bind """
# Logic: Multiply element-wise with pfm matrix.
# In pfm (binary allowed/unallowed, binding/nonbinding), unallowed/nonbinding nucleotides are coded as 1.
# After multiplication, sum everything, and if zero, it's bind. If it's nonzero, it's no-bind -> it won't bind even if one nucleotide mismatches
# Invert to get 1 for bind, 0 for no-bind

# Assumption: all possible combo of allowed/binding nucleotides at different positions are binding combos -> if A and C are allowed for binding in pos 1, A and C are allowed for binding in pos 2, assume AA, AC, CA, CC are all possible for binding
# Otherwise, positive set of binding DNA seq will be even smaller


pfm_mask = np.equal(rounded_pfm, 0) # True/1s in pfm_mask are non-binding nucleotides

nobindis1_labels = [ np.sum(np.multiply(matrix, pfm_mask)) for matrix in dna_matrices]    # nonzero is no-bind, 0 is bind
labels = [not label for label in nobindis1_labels] # 1 is bind, 0 is no-bind

# full dataset 
fullset = [list(tuple) for tuple in zip(sequences, dna_matrices, labels)]



""" Create balanced dataset """
# positive dataset
positive_set = list(filter(lambda x: x[2]==True, fullset))
print(len(positive_set)) # 4*4*1*1*3*3*1*1*4*4 = 2304 binding sequences

# randomly sample negative dataset to get balanced set
negative_set = list(filter(lambda x: x[2]==False, fullset))
negative_subset = sample(negative_set, len(positive_set))

balanced_set = positive_set + negative_subset




""" Repeat to create alternative set without rounding """
pfm_noround = normalized_pfm
pfm_mask_noround = np.equal(pfm_noround, 0)
nobindis1_labels_noround = [np.sum(np.multiply(matrix, pfm_mask_noround)) for matrix in dna_matrices]    # 1 is no-bind, 0 is bind
labels_noround = [not label for label in nobindis1_labels_noround] # 1 is bind, 0 is no-bind

fullset_noround = [list(tuple) for tuple in zip(sequences, dna_matrices, labels_noround)]

positive_set_noround = list(filter(lambda x: x[2]==True, fullset_noround))
print(len(positive_set_noround)) # 4*4*2*3*3*3*3*4*4*4 = 165888 binding sequences

negative_set_noround = list(filter(lambda x: x[2]==False, fullset_noround))
negative_subset_noround = sample(negative_set_noround, len(positive_set_noround))

balanced_set_noround = positive_set_noround + negative_subset_noround



