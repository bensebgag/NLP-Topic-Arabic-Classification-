"""
src/data/dataset.py

PyTorch Dataset class for the Arabic text classification task.

Will contain:
- ArabicTextDataset class inheriting from torch.utils.data.Dataset
- __getitem__ returning (input_ids, label)
- Collate function for DataLoader batching
"""