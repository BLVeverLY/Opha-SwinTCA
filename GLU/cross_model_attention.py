import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from config.base_config import Config

NUM_HEAD = 1 # 2

class MultiHeadedAttention(nn.Module):
    def __init__(self, input_dim): # , config: Config):
        super(MultiHeadedAttention, self).__init__()
        self.embed_dim = input_dim # config.embed_dim
        self.num_heads = NUM_HEAD # config.num_mha_heads
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    
    def forward(self, image1_embeds, image2_embeds):
        """
        Input
            image1_embeds: batch_size x embed_dim
            image2_embeds: batch_size x embed_dim
        Output
            o: batch_size_image2 x batch_size_image1 x embed_dim
        """
        batch_size_image1, _ = image1_embeds.shape
        # batch_size_image1 x embed_dim
        q = self.q_proj(image1_embeds)
        q = q.reshape(batch_size_image1, self.num_heads, self.head_dim)
        # num_heads x head_dim x batch_size_image1
        q = q.permute(1,2,0) # [4,2,384] -> [2,384,4]

        batch_size_image2, _ = image2_embeds.shape
        # batch_size_image2 x embed_dim
        k = self.k_proj(image2_embeds)
        k = k.reshape(batch_size_image2, self.num_heads, self.head_dim)
        # num_heads x batch_size_image2 x head_dim
        k = k.permute(1,0,2) # [4,2,384] -> [2,4,384]

        # batch_size_image2 x embed_dim
        v = self.v_proj(image2_embeds)
        v = v.reshape(batch_size_image2, self.num_heads, self.head_dim)
        # num_heads x head_dim x batch_size_image2
        v = v.permute(1,2,0) # [4,2,384] -> [2,384,4]

        # num_heads x batch_size_image2 x batch_size_image1
        attention_logits = k @ q # [2,4,384] @ [2,384,4] -> [2,4,4]
        attention_logits = attention_logits / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_logits, dim=2) # [2,4,4]

        # num_heads x head_dim x batch_size_image1
        attention = v @ attention_weights # [2,384,4] @ [2,4,4] -> [2,384,4]
        # batch_size_image1 x num_heads x head_dim
        attention = attention.permute(2,0,1)
        attention = attention.reshape(batch_size_image2, self.embed_dim)

        # batch_size_image2 x batch_size_image1 x embed_dim
        o = self.out_proj(attention)
        return o


class Transformer(nn.Module):
    def __init__(self, input_dim): # , config: Config):
        super(Transformer, self).__init__()
        self.embed_dim = input_dim # config.embed_dim 768 / 2048
        dropout = 0.1 # config.transformer_dropout

        self.cross_attn = MultiHeadedAttention(input_dim) # (config)

        self.linear_proj = nn.Linear(self.embed_dim, self.embed_dim)
            
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.layer_norm3 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_parameters()

    
    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)


    def forward(self, image1_embeds, image2_embeds):
        """
        Input
            image1_embeds: batch_size x embed_dim
            image2_embeds: batch_size x embed_dim
        Output
            out: batch_size_image2 x batch_size_image1 x embed_dim
        """
        image1_embeds = self.layer_norm1(image1_embeds)
        image2_embeds = self.layer_norm1(image2_embeds)

        # num_vids x num_texts x embed_dim
        attn_out = self.cross_attn(image1_embeds, image2_embeds)
        attn_out = self.layer_norm2(attn_out)
        linear_out = self.linear_proj(attn_out)
        out = attn_out + self.dropout(linear_out)
        out = self.layer_norm3(out)

        return out
