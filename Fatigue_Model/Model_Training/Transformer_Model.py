import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, input_size=7, num_classes=7, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=0.3)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        x = self.embedding(x)             # [batch, seq_len, d_model]
        x = x.permute(1, 0, 2)            # Transformer expects [seq_len, batch, d_model]
        out = self.transformer_encoder(x) # [seq_len, batch, d_model]
        out = out.mean(dim=0)             # Global average pooling over sequence
        out = self.fc(out)
        return F.log_softmax(out, dim=1)