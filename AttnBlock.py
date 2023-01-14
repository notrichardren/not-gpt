import math
import os
from dataclasses import dataclass
from typing import Any, Optional
import torch as t
import transformers
from einops import rearrange
from fancy_einsum import einsum
from torch import nn
import utils



class AttnBlock(nn.Module):

    
    # Input x: shape (batch, seq, hidden_size)
    # Return: shape (batch, seq, hidden_size)
    def forward(self, x: t.Tensor) -> t.Tensor:
    