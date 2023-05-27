import numpy as np

def jaspar_record_to_array(pfm_record):
    '''Takes in Bio.motifs.jaspar.Record object, returns 4*10 np array representing pfm. Row encoding is in alphabetical order: ACGT'''
    pfm_lines = str(pfm_record).splitlines() # split by lines into a, c, g, t data
    a, c, g, t = pfm_lines[4].split(), pfm_lines[5].split(), pfm_lines[6].split(), pfm_lines[7].split() # split by whitespace into data for each position
    pfm = np.array([a[1:], c[1:], g[1:], t[1:]], dtype=float) # [1:] indexing removes "A:", "C:", etc
    return pfm


def len_n_dna_sequences(n):
    """ Returns list of all possible dna seq of length n """
    sequences=['']
    for _ in range(n): # cycle n times to get 4^n sequences of length n
        sequences = append_nucleotide_helper(sequences)
    return sequences

def append_nucleotide_helper(oldset):
    """ Helper method for len_n_dna_sequences() """
    newset = []
    for old in oldset:
        news = [old+x for x in ['A', 'C', 'G', 'T']]
        newset.extend(news)
    return newset


def len_n_dna_matrices(n):
    """ Returns list of all possible dna seq of length n, in one hot matrix form """
    dna_matrices = [np.array([[],[],[],[]])]
    for _ in range(n): # cycle n times to get list of 4^n matrices of size 4xn
        dna_matrices = append_nucleotide_col_helper(dna_matrices)
    return dna_matrices

def append_nucleotide_col_helper(oldset):
    """ Helper method for len_n_dna_matrices() """
    newset = []
    for old in oldset:
        news = [np.append(old, col, axis=1) for col in [np.array([[1],[0],[0],[0]]), np.array([[0],[1],[0],[0]]),np.array([[0],[0],[1],[0]]),np.array([[0],[0],[0],[1]])]]
        newset.extend(news)
    return newset