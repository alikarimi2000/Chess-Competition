import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import nltk
import nltk
from torch.nn.utils.rnn import pack_padded_sequence
nltk.download('punkt')

class ChessDataset(Dataset):
    def __init__(self, data_dir, vocab, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_names = os.listdir(self.data_dir)
        self.vocab=vocab

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        image = Image.open(os.path.join(self.data_dir, file_name)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        label = self.get_label(file_name)
        tokens = nltk.tokenize.word_tokenize(str(label))
        caption = []
        caption.append(self.vocab('<BOS>'))
        a=([token for token in str(tokens)[2:-2]])
        caption.extend([self.vocab(token) for token in str(tokens)[2:-2]])
        caption.append(self.vocab('<EOS>'))
        target = torch.Tensor(caption)
        return image, target
    
    def get_label(self, file_name):
        return file_name.split(".")[0]



def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 224, 224).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 224, 224).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths



def get_loader(root, vocab, transform, batch_size, shuffle, num_workers):

    train_dataset = ChessDataset(root,vocab, transform=transform)

    data_loader = DataLoader(dataset=train_dataset, 
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             collate_fn=collate_fn)
    return data_loader