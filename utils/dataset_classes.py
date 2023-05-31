from torch.utils.data import Dataset
import torch
    
class KmerDataset(Dataset):
    """ Takes in two lists, __getitem__ returns tuples """
    def __init__(self, kmer_count, label):
        self.kmer_count=kmer_count
        self.label=label

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        sample = (torch.tensor(self.kmer_count[index], dtype=torch.float32), torch.tensor(self.label[index], dtype=torch.float32))
        return sample
    

class SeqDataset(Dataset):
    """ Takes in two lists, __getitem__ returns tuples """
    def __init__(self, seq, label):
        self.seq=seq
        self.label=label

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        sample = (self.seq[index], torch.tensor(self.label[index], dtype=torch.float32))
        return sample
    
    
class CombinedDataset(Dataset):
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