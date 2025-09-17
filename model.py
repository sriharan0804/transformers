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
        div_term = torch.exp(torch.arrange(0 , d_model , 2).float() * (-Math.log(10000.0)/d_model))
        pe[: , 0::2] = torch.sin(position*div_term)
        pe[: , 1::2]=torch.cos(position*div_term)
        pe = pe.unsqueeze(0)
        self.Register_buffer("pe" , pe)

    def forward(self , X):
        x = X+self.pe[: , :X.shape[1] , :].require_grade_(False)
        return self.dropout(x)


class layerNormalization(nn.Module):
    def __init_(self , epc: float = 10** -6)->None:
        super().__init__()
        self.eps = epc
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
        
        query = query.view(query.shaper[0] , query.shape[1] , self.h , self.d_k).transpose( 1,2)
        key = key.view(key.shaper[0] , key.shape[1] , self.h , self.d_k).transpose( 1,2)
        value = value.view(value.shaper[0] , value.shape[1] , self.h , self.d_k).transpose( 1,2)
        X , self.attention_score = MultiHeadAttentionBlock.attention(query , key , value ,mask, self.dropout)
        X = X.transpose( 1, 2).contegious().view(X.shape[0] , -1 , self.h *self.d_k)
        return self.w_o(X)
    
    @staticmethod
    def attention(query , key , value , mask , dropout):
        d_k = query.shape[-1]
        attention_score  = (query @ key.transpode(-2 , -1))/ Math.sqrt(d_k)
        if mask is not None :
            attention_score.masked_field_(mask == 0  , -1e9)
        attention_score = attention_score.softmax(dim = -1)
        if dropout is not None :
            attention_score = dropout(attention_score)
        return (attention_score @ value) , attention_score


class ResudianConnection (nn.Module):
    def __init__(self  , dropout:float)->None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = layerNormalization()

    def forward(self  , X , sublayer):
        return  X + self.dropout(sublayer(self.norm(X)))

class EncoderBlock(nn.Module):
    def __init__(self , dropout:float , Self_Attention_Block : MultiHeadAttentionBlock , feed_froword_block : FeedForward)->None:
        super().__init__()
        self.self_attention_layer = Self_Attention_Block  
        self.feed_forword_layer = feed_froword_block

        self.residual_connection = nn.ModuelList([ResudianConnection(dropout) for _ in range (2)])
    def forword(self , X , src_mask):
        X = self.residual_connection[0](X , lambda x : self.self_attention_layer(x ,x , x , src_mask))
        X = self.residual_connection[1](X , self.feed_forward_layer())
        return X

class Encoder(nn.Module):
    def __init__(self , layers:nn.ModuleList )->None:
        super().__init__()
        self.layer = layers
        self.norm = layerNormalization()

    def forward(self , x , mask):
        for layer in self.layers:
            x = layer(x , mask)
        return self.norm(x)
        

class DecoderBlock(nn.module):
    def __init__(self , self_attention_block : MultiHeadAttentionBlock , cross_attention_block :MultiHeadAttentionBlock , feed_forward_block : FeedForward , dropout : float = 0.1):
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.moduleList([ResudianConnection(dropout)for _ in range(3)])
    def forward(self , x , encoder_output , src_mask ,tgt_mask):
        x = self.residual_connection[0](x , lambda x:self.self_attention_block(x ,x , x , tgt_mask))
        x = self.residual_connection[1](x , lambda x:self.cross_attention_block(x , encoder_output , encoder_output , src_mask))
        x = self.residual_connection[2](x , self.feed_forward_block)

        return x
class Decoder(nn.Module):
    def __init__(self , layers:nn.ModuleList) ->None:
        super().__init__()
        self.layers = layers
        self.norm = layerNormalization()

    def forward(self , x , encoder_output , src_mask , tgt_mask):
        for layer in self.layers:
        X = layer(x , encoder_output , src_mask , tgt_mask)
        return self.norm(x)
            
class projectionLayer(nn.Module):
    def __init__(self , d_model : int , vocab_size : int) ->None:
        super().__init__()
        self.proj = nn.linear(d_model , vocab_size)
    def forward(self , x):
        return torch.log_softmax(self.proj(x) , dim =-1)
    
class Transformers(nn.Module):
    def __init__(self , encoder :Encoder , decoder : Decoder , src_embeding :inputembadding  , tgt_embedding : inputembadding  , src_position :positional_encoding , tgt_position : positional_encoding , projection_layer : projectionLayer   ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embeding=src_embeding
        self.tgt_embedding = tgt_embedding
        self.src_position =src_position
        self.tgt_position = tgt_position
        self.projection_layer = projection_layer

    def encoder(self , src , src_mask):
        src = self.src_emdeding(src)
        src = self.src_position(src)
        return self.encoder(src , src_mask)
    def decoder(self , encoder_output , src_mask , tgt , tgt_mask):
        tgt = self.tgt_encoder(tgt)
        tgt = self.tgt_position(tgt)
        return self.decoder(tgt , encoder_output , src_mask , tgt_mask)
    def project(self  , x):
        return self.projection_layer(x)
    ##this line is modefied please note it 
    def build_transformer(self , src_vocab_size : int , tgt_vocab_size : int , src_seq_len : int , tgt_seq_len : int , d_model : int = 512 , N: int = 6 , h : int = 8 , dropout :float = 0.1  , d_ff : int = 2048) -> Transformers:
        src_embed = self.src_embeding(d_model , src_vocab_size)
        tgt_embed = self.tgt_embedding(d_model , tgt_vocab_size)
        src_pos = self.src_position(d_model , src_seq_len)
        tgt_pos = self.tgt_position(d_model , tgt_seq_len , dropout)
        encoder_block =[]
        for _ in range(N):
            encoder_self_attention_block = MultiHeadAttentionBlock(d_model , h , dropout)
            feed_forward_block = FeedForward(d_model , d_ff , dropout)
            encoder_block = EncoderBlock(encoder_self_attention_block , feed_forward_block , dropout)
            encoder_block.append(encoder_block)

        decoder_block =[]
        for _ in range(N):
            decoder_self_attention_block =MultiHeadAttentionBlock(d_model , h , dropout)
            decoder_cross_attention_block =MultiHeadAttentionBlock(d_model , h , dropout)
            feed_forward_block = FeedForward(d_model , d_ff , dropout)
            decoder_block = DecoderBlock(decoder_self_attention_block ,decoder_cross_attention_block ,feed_forward_block,dropout)
            decoder_block.append(decoder_block)

        encoder = Encoder(nn.ModuleList(encoder_block))
        decoder = Decoder(nn.ModuleList(decoder_block))
        projection_layer = projectionLayer(d_model , tgt_vocab_size)
        transformer = Transformers(encoder , decoder , src_embed , tgt_embed , src_pos , tgt_pos , projection_layer )

        for p in transformer.parameter:
            if p.dim()>1:
                nn.init.xavier_uniform_(p)
        return transformer


    
    


