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


class layerNormalization(nn.Module):
    def __init_(self , epc: float)->None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.parameter(torch.ones(1))
        self.bias = nn.parameter(torch.zeros(1))
    def forward(self , X ):
        mean = X.mean(dim = -1 , keepdim = True)
        std = X.std(dim = -1 , keepdim= True)
        return self.alpha * (X - mean )/(std +self.eps) + self.bias

class FeedForward(nn.Module):
    def __init__(self , d_model , d_ff , dropout:float )->None :
        super().__init__()
        self.linear_1 = nn.linear(d_model , d_ff)
        self.dopout=nn.Dropout(dropout)
        self.linear_2 = nn.linear(d_ff , d_model )
    def forward(self , x ):
        return  self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttentionBlock (nn.Module):
    def __init__(self , d_model:int , h:int , dropout:float)->None:
        super().__init__()
        self.d_model = d_model 
        self.h = h
        assert d_model %h == 0 ,"d_model is not devided by h"
        self.d_k = d_model // h
        self.w_q = nn.linear(d_model , d_model)
        self.w_k = nn.linear(d_model , d_model)
        self.w_v = nn.linear(d_model , d_model)
        self.w_o = nn.linear(d_model , d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self , q , k , v , mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        self.d_k = query.shape[-1] // h
        query = query.view(query.shaper[0] , query.shape[1] , self.h , self.d_k).transpose( 1,2)
        key = key.view(key.shaper[0] , key.shape[1] , self.h , self.d_k).transpose( 1,2)
        value = value.view(value.shaper[0] , value.shape[1] , self.h , self.d_k).transpose( 1,2)
        X , self.attention_score = MultiHeadAttentionBlocks.attention(query , key , value ,mask, self.dropout)
        X = X.transpose( 1, 2).contegious().view(X.shape[0] , -1 , self.h *self.d_k)
        return self.w_o(x)
    
    @statucmethod
    def attention(query , key , value , mask , dropout):
        d_k = query.shape[-1]
        attention_score  = (query @ key.transpode(-2 , -1))/ Math.sqrt(d_k)
        if mask is not None :
            attention_score.masked_field_(mask == 0  , -1e9)
        attention_score = attention_score.softmax(dim = -1)
        if dropout is not None :
            attention_score = dropout(attention_score)
        return (attention_score @ value) , attention_score




