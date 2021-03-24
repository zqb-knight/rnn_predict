import torch
from torch import nn
import numpy as np

INPUT_SIZE = 1
LEARNING_RATE = 0.02

#构建rnn模型
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)

rnn = RNN()

#定义优化器和损失函数
optimizer = torch.optim.Adam(rnn.parameters(), lr=LEARNING_RATE)
loss = nn.MSELoss()

