"""Two-head GRU baseline; only visible trajectory features are inputs."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models.intent_gru import INPUT_SIZE

class AntiDeceptionGRU(nn.Module):
    def __init__(self,hidden_size=32,input_size=INPUT_SIZE):
        super().__init__()
        if hidden_size<1:raise ValueError('hidden_size must be positive')
        if input_size<1:raise ValueError('input_size must be positive')
        self.gru=nn.GRU(input_size,hidden_size,batch_first=True)
        self.goal_head=nn.Linear(hidden_size,2)
        self.deception_head=nn.Linear(hidden_size,2)
    def forward(self,sequences,lengths):
        packed=pack_padded_sequence(sequences,lengths.cpu(),batch_first=True,enforce_sorted=False)
        _,hidden=self.gru(packed)
        return self.goal_head(hidden[-1]),self.deception_head(hidden[-1])
