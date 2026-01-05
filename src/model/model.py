import torch
import torch.nn as nn
import torchvision.models as models



class SimpleAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SimpleAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        # x shape: [batch, seq_len, embed_dim]
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)

        # Apply attention weights
        output = torch.matmul(attn_weights, V)
        return output


class VideoClassificationModel(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        # Loading Resnet pre-trained weights
        self.model = models.resnet18(pretrained=True)
        self.model = self.config(self.model)
        
        self.rnn = self.define_rnn_cell()
        
        self.attention = SimpleAttention(kwargs['hidden_size'])
        
        self.fc = self.fc_layers(kwargs['hidden_size'], kwargs['num_classes'])
        
        
    def config(self, model):
        # Remove the classification head (fc layer)
        modules = list(model.children())[:-1]  # remove last layer
        feature_extractor = nn.Sequential(*modules)
        for param in feature_extractor.parameters():
            param.requires_grad = False
        
        return feature_extractor
    
    
    def define_rnn_cell(self, input_size=512, hidden_size=256, cell_type="LSTM"):
        """Defines an RNN cell (default: LSTM)."""
        if cell_type == "LSTM":
            return nn.LSTM(input_size=input_size, hidden_size=hidden_size)  
        elif cell_type == "GRU":
            return nn.GRU(input_size=input_size, hidden_size=hidden_size)  
        else:
            return nn.RNN(input_size=input_size, hidden_size=hidden_size)
        
        
    def fc_layers(self, input_size, num_classes):
        fc_layer = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, num_classes)
        )
        
        return fc_layer
    
    def forward(self, x):
        
        batch_size, seq_len, C, H, W = x.size()
    
        # (1) Extract CNN features for each frame
        x = x.view(batch_size * seq_len, C, H, W)          # merge batch & time
        features = self.model(x)                           # (batch*seq, 512, 1, 1)
        features = features.view(batch_size, seq_len, -1)  # (batch, seq, 512)
        
        # (2) RNN over sequence
        rnn_out, _ = self.rnn(features)                    # (batch, seq, hidden)
        
        # (3) Apply attention over sequence
        attn_out = self.attention(rnn_out)                 # (batch, seq, hidden)
        attn_out = attn_out.mean(dim=1)                    # reduce sequence → (batch, hidden)
        
        # (4) Final classification
        output = self.fc(attn_out)                         # (batch, num_classes)
        return output
        
        
    