import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from torchtext.vocab import build_vocab_from_iterator
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn
# Our libraries
from train import train_model
from model_utils import *
from predict_utils import *
from vis_utils import *

class ChessDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        a= len(self.labels)
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_before_name = self.labels.iloc[idx, 0]+'_before.png'

        img_after_name = self.labels.iloc[idx, 0]+'_after.png'
        move_label = self.labels.iloc[idx, 1]

        img_before_path = os.path.join(self.root_dir, img_before_name)
        img_after_path = os.path.join(self.root_dir, img_after_name)
     
        img_before = Image.open(img_before_path).convert('RGB')
        img_after = Image.open(img_after_path).convert('RGB')

        if self.transform:
            img_before = self.transform(img_before)
            img_after = self.transform(img_after)

        if "K" in move_label:
            move_label=0
         
            return img_before, img_after, move_label
            
        elif "R" in move_label:
            move_label=1
  
            return img_before, img_after, move_label
        elif "N" in move_label:
            move_label=2
            return img_before, img_after, move_label
        
        elif "Q" in move_label:
            move_label=3
            return img_before, img_after, move_label
        
        elif "B" in move_label:
            move_label=4
            return img_before, img_after, move_label
        elif "C" in move_label:
            move_label=5
            return img_before, img_after, move_label
      
        else:
            move_label=6

        return img_before, img_after, move_label
    
class ChessDataset1(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        a= len(self.labels)
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_before_name = self.labels.iloc[idx, 0]+'_before.png'

        img_after_name = self.labels.iloc[idx, 0]+'_after.png'
        move_label = self.labels.iloc[idx, 1]

        img_before_path = os.path.join(self.root_dir, img_before_name)
        img_after_path = os.path.join(self.root_dir, img_after_name)
     
        img_before = Image.open(img_before_path).convert('RGB')
        img_after = Image.open(img_after_path).convert('RGB')

        if self.transform:
            img_before = self.transform(img_before)
            img_after = self.transform(img_after)
        
        if "aa" in move_label:
            move_label=0
            return img_before, img_after, move_label
        
        elif "ab" in move_label:
            move_label=1
            return img_before, img_after, move_label
        
        elif "ac" in move_label:
            move_label=2
            return img_before, img_after, move_label
        
        elif "ad" in move_label:
            move_label=3 
            return img_before, img_after, move_label
        
        elif "ae" in move_label:
            move_label=4
            return img_before, img_after, move_label
        
        elif "af" in move_label:
            move_label=5
            return img_before, img_after, move_label
        
        elif "ag" in move_label:
            move_label=6
            return img_before, img_after, move_label
        
        elif "ah" in move_label:
            move_label=7
            return img_before, img_after, move_label

        elif "a" in move_label:
            move_label=8
            return img_before, img_after, move_label

        elif "b" in move_label:
            move_label=9
            return img_before, img_after, move_label
            
        elif "ba" in move_label:
            move_label=10  
            return img_before, img_after, move_label
        
        elif "bb" in move_label:
            move_label=11
            return img_before, img_after, move_label
        
        elif "bc" in move_label:
            move_label=12 
            return img_before, img_after, move_label
        
        elif "bd" in move_label:
            move_label=13 
            return img_before, img_after, move_label
        
        elif "be" in move_label:
            move_label=14 
            return img_before, img_after, move_label
        
        elif "bf" in move_label:
            move_label=15 
            return img_before, img_after, move_label
        
        elif "bg" in move_label:
            move_label=16
            return img_before, img_after, move_label
        
        elif "bh" in move_label:
            move_label=17
            return img_before, img_after, move_label
        elif "c" in move_label:
            move_label=18
            return img_before, img_after, move_label
        
        elif "ca" in move_label:
            move_label=19
            return img_before, img_after, move_label
        
        elif "cb" in move_label:
            move_label=20
            return img_before, img_after, move_label
        elif "cc" in move_label:
            move_label=21
            return img_before, img_after, move_label
        
        elif "cd" in move_label:
            move_label=22
            return img_before, img_after, move_label
        
        elif "ce" in move_label:
            move_label=23
            return img_before, img_after, move_label
        
        elif "cf" in move_label:
            move_label=24
            return img_before, img_after, move_label
        
        elif "cg" in move_label:
            move_label=25
            return img_before, img_after, move_label
        
        elif "ch" in move_label:
            move_label=26
            return img_before, img_after, move_label
        
        elif "d" in move_label:
            move_label=27
            return img_before, img_after, move_label
        
        elif "da" in move_label:
            move_label=28
            return img_before, img_after, move_label
        
        elif "db" in move_label:
            move_label=29
            return img_before, img_after, move_label
        
        elif "dc" in move_label:
            move_label=30
            return img_before, img_after, move_label
        
        elif "dd" in move_label:
            move_label=31 
            return img_before, img_after, move_label
        
        elif "de" in move_label:
            move_label=32
            return img_before, img_after, move_label
        
        elif "df" in move_label:
            move_label=33
            return img_before, img_after, move_label
        
        elif "dg" in move_label:
            move_label=34
            return img_before, img_after, move_label
        
        elif "dh" in move_label:
            move_label=35
            return img_before, img_after, move_label
        
        elif "e" in move_label:
            move_label=36
            return img_before, img_after, move_label
        
        elif "ea" in move_label:
            move_label=37 
            return img_before, img_after, move_label
        
        elif "eb" in move_label:
            move_label=38
            return img_before, img_after, move_label
        
        elif "ec" in move_label:
            move_label=39
            return img_before, img_after, move_label
        
        elif "ed" in move_label:
            move_label=40
            # move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        elif "ee" in move_label:
            move_label=41
            # move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        
        elif "ef" in move_label:
            move_label=42
            # move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        
        elif "eg" in move_label:
            move_label=43
            # move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        
        elif "eh" in move_label:
            move_label=44
            # move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        
        elif "f" in move_label:
            move_label=45
            # move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        
        elif "fa" in move_label:
            move_label=46
            # move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        
        elif "fb" in move_label:
            move_label=47
            # move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        
        elif "fc" in move_label:
            move_label=48
            # move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        
        elif "fd" in move_label:
            move_label=49
            return img_before, img_after, move_label
        
        elif "fe" in move_label:
            move_label=50
            # move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        elif "ff" in move_label:
            move_label=51
            # move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        
        elif "fg" in move_label:
            move_label=52
            # move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        
        elif "fh" in move_label:
            move_label=53
            # move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        
        elif "g" in move_label:
            move_label=54
            # move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        
        elif "ga" in move_label:
            move_label=55
            # move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        
        elif "gb" in move_label:
            move_label=56
            # move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        
        elif "gc" in move_label:
            move_label=57
            # move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        
        elif "gd" in move_label:
            move_label=58
            # move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        
        elif "ge" in move_label:
            move_label=59
            # move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        
        elif "gf" in move_label:
            move_label=60
            # move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        
        elif "gg" in move_label:
            move_label=61
            # move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        
        elif "gh" in move_label:
            move_label=62
            # move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        
        elif "h" in move_label:
            move_label=63
            # move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        
        elif "ha" in move_label:
            move_label=64
            # move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        
        elif "hb" in move_label:
            move_label=65
            # move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        
        elif "hc" in move_label:
            move_label=66
            # move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        
        elif "hd" in move_label:
            move_label=67
            # move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        
        elif "he" in move_label:
            move_label=68
            # move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        
        elif "hf" in move_label:
            move_label=69
            # move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        
        elif "hg" in move_label:
            move_label=70
            # move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        elif "hh" in move_label:
            move_label=71
            # move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
    

        else :
            a=move_label
            move_label=72
            #move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label

        move_label=torch.tensor(move_label)  
        return img_before, img_after, move_label
    

class ChessDataset2(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        a= len(self.labels)
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_before_name = self.labels.iloc[idx, 0]+'_before.png'

        img_after_name = self.labels.iloc[idx, 0]+'_after.png'
        move_label = self.labels.iloc[idx, 1]

        img_before_path = os.path.join(self.root_dir, img_before_name)
        img_after_path = os.path.join(self.root_dir, img_after_name)
     
        img_before = Image.open(img_before_path).convert('RGB')
        img_after = Image.open(img_after_path).convert('RGB')

        if self.transform:
            img_before = self.transform(img_before)
            img_after = self.transform(img_after)

        if "x" in move_label:
            move_label=0
            move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
            

      
        else:
            move_label=1

        move_label=torch.tensor(move_label)  
        return img_before, img_after, move_label
    


class ChessDataset3(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        a= len(self.labels)
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_before_name = self.labels.iloc[idx, 0]+'_before.png'

        img_after_name = self.labels.iloc[idx, 0]+'_after.png'
        move_label = self.labels.iloc[idx, 1]

        img_before_path = os.path.join(self.root_dir, img_before_name)
        img_after_path = os.path.join(self.root_dir, img_after_name)
     
        img_before = Image.open(img_before_path).convert('RGB')
        img_after = Image.open(img_after_path).convert('RGB')

        if self.transform:
            img_before = self.transform(img_before)
            img_after = self.transform(img_after)

        if "1" in move_label:
            move_label=0
            move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        
        if "2" in move_label:
            move_label=1
            move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        
        if "3" in move_label:
            move_label=2
            move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        
        if "4" in move_label:
            move_label=3
            move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        
        if "5" in move_label:
            move_label=4
            move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        
        if "6" in move_label:
            move_label=5
            move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        
        if "7" in move_label:
            move_label=6
            move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        
        if "8" in move_label:
            move_label=7
            move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        

      
        else:
            move_label=8

        move_label=torch.tensor(move_label)  
        return img_before, img_after, move_label
    

class ChessDataset4(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        a= len(self.labels)
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_before_name = self.labels.iloc[idx, 0]+'_before.png'

        img_after_name = self.labels.iloc[idx, 0]+'_after.png'
        move_label = self.labels.iloc[idx, 1]

        img_before_path = os.path.join(self.root_dir, img_before_name)
        img_after_path = os.path.join(self.root_dir, img_after_name)
     
        img_before = Image.open(img_before_path).convert('RGB')
        img_after = Image.open(img_after_path).convert('RGB')

        if self.transform:
            img_before = self.transform(img_before)
            img_after = self.transform(img_after)
        
        if "+" in move_label:
            move_label=0
            move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        
  
        if "O-O-O" in move_label:
            move_label=1
            move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
        
        if "O-O" in move_label:
            move_label=2
            move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label

      
        else:
            move_label=3

        move_label=torch.tensor(move_label)  
        return img_before, img_after, move_label
    

class ChessDataset5(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        a= len(self.labels)
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_before_name = self.labels.iloc[idx, 0]+'_before.png'

        img_after_name = self.labels.iloc[idx, 0]+'_after.png'
        move_label = self.labels.iloc[idx, 1]

        img_before_path = os.path.join(self.root_dir, img_before_name)
        img_after_path = os.path.join(self.root_dir, img_after_name)
     
        img_before = Image.open(img_before_path).convert('RGB')
        img_after = Image.open(img_after_path).convert('RGB')

        if self.transform:
            img_before = self.transform(img_before)
            img_after = self.transform(img_after)

        if "=" in move_label:
            move_label=0
            move_label=torch.tensor(move_label)  
            return img_before, img_after, move_label
  
        else:
            move_label=1

        move_label=torch.tensor(move_label)  
        return img_before, img_after, move_label
    

class ChessModel(nn.Module):
    def __init__(self,num_class):
        super().__init__()
        self.conv1 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self,num):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2) #64
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2) #32
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
           nn.MaxPool2d(2) #16
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
           nn.MaxPool2d(2) #8
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)  #4
        )
   
        
        self.fc = nn.Linear(4 * 4 *256, 512)
        self.fc1 = nn.Linear(512,  256)
        self.fc2 = nn.Linear(256, num)

        # self.fc3 = nn.Linear(128, 2)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = out.view(out.size(0), -1)  # flatten
        out = F.relu(self.fc(out))
        out = F.relu(self.fc1(out))
        # out = F.relu(self.fc2(out))
        out = out.view(-1, 256)
        out = self.fc2(out)
        return out

  
def train(train_loader,optimizer,model,criterion):
    total_step = len(train_loader)
    losses = []
    num=20
    for epoch in range(num):
        for i, (img_before, img_after, move_label) in enumerate(train_loader):
        
            img_before = img_before.cuda()
            img_after = img_after.cuda()
            move_label = move_label.cuda()

       
            optimizer.zero_grad()

      
            outputs = model(img_before, img_after)
  
       
            loss = criterion(outputs, move_label)
            losses.append(loss.item())

       
            loss.backward()
            optimizer.step()

       
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch + 1, num, i + 1, total_step, loss.item()))
    return model
                


class ChessDatasetTest(Dataset):
    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.class_names = sorted(os.listdir(root_dir))

    def __len__(self):
      
        return  4487

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_before_name ='img'+str(idx) +'_before.png'

        img_after_name = 'img'+str(idx)+'_after.png'
      

        img_before_path = os.path.join(self.root_dir, img_before_name)
        img_after_path = os.path.join(self.root_dir, img_after_name)
     
        img_before = Image.open(img_before_path).convert('RGB')
        img_after = Image.open(img_after_path).convert('RGB')

        if self.transform:
            img_before = self.transform(img_before)
            img_after = self.transform(img_after)


    
        return img_before, img_after
    
def lab(dataloader,model,class_names):
    model.eval()
    p=[]
    with torch.no_grad():
        
        for  img_before, img_after in dataloader:
            img_before = img_before.cuda()
            img_after = img_after.cuda()
           
            outputs = model( img_before ,img_after)
            _, preds = torch.max(outputs, 1)
           
            result=preds.cpu().numpy()
            pred_class_names = [ class_names[i] for i in result]
            p+= pred_class_names
    return p