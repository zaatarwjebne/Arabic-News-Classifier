import torch
import torch.nn as nn

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_len, num_classes, embedding_matrix=None):
        super(TextCNN, self).__init__()

        # Embedding layer
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Convolutional layer and pooling
        self.conv = nn.Conv1d(embedding_dim, 128, kernel_size=3)
        self.pool = nn.AdaptiveMaxPool1d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # Reshape for Conv1d
        x = self.conv(x)
        x = self.pool(x).squeeze(2)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
