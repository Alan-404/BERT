import torch
from torch.utils.data import Dataset

class BERTDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()