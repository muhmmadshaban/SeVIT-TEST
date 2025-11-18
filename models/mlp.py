import torch.nn as nn 
import torch

class Classifier(nn.Module): 
    """
    MLP classifier for 3 classes
    """
    def __init__(self, num_classes=3, in_features=768*196):  # Changed from 2 to 3 classes
        
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_features, out_features=4096)
        self.linear2 = nn.Linear(in_features=4096, out_features=2048)
        self.linear3 = nn.Linear(in_features=2048, out_features=128)
        self.linear4 = nn.Linear(in_features=128, out_features=num_classes)  # 3 classes
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = x.reshape(-1, 196*768)
        x = nn.functional.relu(self.linear1(x))
        x = nn.functional.relu(self.linear2(x))
        x = nn.functional.relu(self.linear3(x))
        x = self.linear4(x)
        return x