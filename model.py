
import torch.nn as nn

class SignLSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=126,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)        
        out = out.mean(dim=1)        
        out = self.dropout(out)
        return self.fc(out)
