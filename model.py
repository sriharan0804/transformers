import torch 
import troch.nn as nn
import Math
class inputembadding(nn.Module):
    def __init__(self , d_model :int , vocab_len : int):
        super().__init__()
        self.d_model = d_model
        self.vocab_len = vocab_len
        self.embadding = nn.embadding(vocab_len , d_model)
    
    def forward(self , x):
        return self.embadding(x) * Math.sqrt(self.d_model)
    
class positional_encoding(nn.Module):
    def __init__(self , d_model: int , seq_len : int  , Dropout : float):
        super().__init__()
        self.d_model = d_model
        self.seq_len= seq_len
        self.dropout =nn.Dropout(Dropout)
        pe =torch .zeros(d_model , seq_len)
        position= torch.arrange(0, seq_len , dtype = torch.float).unsqueeze(0)
        div_term = torch.exp(torch.arrange(0 , d_model , 2).float() * (-math.log(10000.0)/d_model))
        pe[: , 0::2] = torch.sin(position*div_term)
        pe[: , 1::2]=torch.cos(position*div_term)
        pe = pe.unsqueeze(0)
        self.Register_buffer("pe" , pe)

    def forward(self , X):
        x = X+self.pe[: , :X.shape[1] , :].require_grade_(False)
        return self.dropout(x)
