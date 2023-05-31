from torch.utils.data import Dataset
import torch

class DNADataset(Dataset):
    """ Takes in three lists """
    def __init__(self, seq, kmer_count, label):
        self.seq=seq
        self.kmer_count=kmer_count
        self.label=label

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        sample = {'seq' : self.seq[index], 
                  'kmer count' : torch.tensor(self.kmer_count[index], dtype=torch.float32),
                  'label' : torch.tensor(self.label[index], dtype=torch.float32)}
        return sample
    