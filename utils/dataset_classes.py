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
        return torch.tensor(self.kmer_count[index], dtype=torch.float32), torch.tensor(self.label[index], dtype=torch.float32)

    

class SeqDataset(Dataset):
    """ Takes in two lists, __getitem__ returns tuples """
    def __init__(self, seq, label):
        self.seq=seq
        self.label=label

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        return torch.tensor(self.seq[index]), torch.tensor(self.label[index], dtype=torch.float32)
    