# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 23:05:14 2024

@author: gimal
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#scaled dot product attention 함수 정의
def scaled_dot_product_attention(q: torch.Tensor,
                                 k: torch.Tensor,
                                 v: torch.Tensor,
                                 mask: torch.Tensor = None,
                                 dropout: float = 0.1,
                                 ) -> torch.Tensor:
    #스케일링 먼저 해줌
    d_k = k.size()[-1]

    #쿼리와 키를 행렬곱 해줌
    attn = torch.matmul(q,k.transpose(-2,-1))

    # 여기서 마스킹을 해줌.
    if mask != None:
        # 일단은 1.0에서 마스크를 뺴줘야함
        inverted_mask = 1.0 - mask

        # 그 다음은 값을 변경해줄껀데. True인 애들만 변경해주는거임.
        inverted_mask = inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(attn.dtype).min)

        # 그 다음 attn에 더해줌. 이때 broadcasting이 일어나서 마스크의 차원이 변경됨.
        attn = attn + inverted_mask

    # 그 다음 소프트 맥스 씌어줌.
    attention_weights = F.softmax(attn, dim = -1)

    # drop out이 어텐션 내부에 들어가있음.
    if type(dropout) == float:
        attention_weights = F.dropout(attention_weights, dropout)
    else:
        attention_weights = dropout(attention_weights)

    #그다음에 블렌딩 해줌.
    output = torch.matmul(attention_weights, v)
    return output, attention_weights

# 어텐션 모듈 이니셜라이징, 스플릿헤드, 포워드.
class Attention(nn.Module):
    # 이니셜 라이징 할 때 d_model 이랑 헤드의 수 , 드랍아웃을 받아줌. 바이아스는 트루.
    def __init__(self, d_model, num_heads, dropout = 0.1, use_bias = True):
        super(Attention,self).__init__()
        #만약 d_model를 num_heads로 나눴을 때 나머지가 있다면 에러를 반환해줌.
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        # 헤드대로 쪼개줌.
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # 각 헤드별로 들어오는 쿼리를 조옥적인 리니어로 해주기
        self.wq = nn.Linear(d_model,d_model,bias=use_bias)
        self.wk = nn.Linear(d_model,d_model,bias=use_bias)
        self.wv = nn.Linear(d_model,d_model,bias = use_bias)

        # 드랍아웃
        self.dropout = nn.Dropout(dropout)

        self.wo = nn.Linear(d_model,d_model,bias = use_bias)

    #스플릿 헤드.
    def split_heads(self,x,batch_size):
        x = x.view(batch_size, -1, self.num_heads,self.d_k)
        x = x.transpose(1,2).contiguous()
        return x

    #포워드.
    def forward(self,query,key,value,mask=None):
        q = self.wq(query)
        k = self.wk(query)
        v = self.wv(query)

        #그다음 쉐입을 바꿔줘야 한대..
        _, qS = q.size()[0], q.size()[1] # qS = query_seq_len
        B, S = k.size()[0], k.size()[1] # S = key_seq_len

        # 늘어난 애들을 스플릿으로 쪼개줘
        q = self.split_heads(q,B)
        k = self.split_heads(k,B)
        v = self.split_heads(v,B)

        #여기서 부터 어텐션 scaled_attention은 blended 벡터, attention_weights는 키랑 쿼리 곱해서 마스킹하고, 소프트 맥스 씌운거.
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v,mask,self.dropout)

        #계산한 어텐션을 하나로 합쳐주는거임.
        scaled_attention = scaled_attention.transpose(1,2)
        concat_attention = scaled_attention.reshape(B,qS,-1)


        output = self.wo(concat_attention)
        return output, attention_weights



class TransformerEncoderLayer(nn.Module):
    def __init__(self,d_model,num_head, dim_feedforward,dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.dropout= dropout

        #어텐션
        self.self_attn = Attention(d_model,num_head,dropout)

        #MLP
        self.act_fn = nn.GELU()
        self.fc1 = nn.Linear(d_model,dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward,d_model)

        #노멀라이제이션
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)


    def forward(self,x,mask):

        residual = x
        # 셀프 멀티헤드 어텐션
        x, attn_scores = self.self_attn(query=x,key=x,value=x,mask=mask)
        x = F.dropout(x,self.dropout,training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        # MLP
        residual = x
        x = self.act_fn(self.fc1(x))
        x = F.dropout(x,self.dropout,training=self.training)
        x = self.fc2(x)
        x = residual + x
        x = self.final_layer_norm(x)

        return x, attn_scores


import copy
def clone(module,N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TransformerEncoder(nn.Module):
    def __init__(self,num_layers,d_model,num_heads,dropout,dim_feedforward=None):
        super(TransformerEncoder,self).__init__()

        self.num_layers = num_layers
        if dim_feedforward == None: dim_feedforward = d_model*4

        a_layer = TransformerEncoderLayer(d_model, num_heads, dim_feedforward,dropout)
        self.layers = clone(a_layer,num_layers)

    def forward(self,x,mask = None):
        layers_attn_scores = []
        for layer in self.layers:
            x, attn_scores = layer(x,mask)
            layers_attn_scores.append(attn_scores)

        return x, layers_attn_scores


def cp_weight(src, tar, copy_bias=True, include_eps=False):
    assert tar.weight.size() == src.weight.size(), "Not compatible parameter size"
    tar.load_state_dict( src.state_dict() , strict=False)


    if include_eps:
        # in case of LayerNorm.
        with torch.no_grad():
            tar.eps = src.eps
class BertEmbeddings(nn.Module):
    """ this embedding moudles are from huggingface implementation
        but, it is simplified for just testing
    """

    def __init__(self, vocab_size, hidden_size, pad_token_id, max_bert_length_size, layer_norm_eps, hidden_dropout_prob):
        super().__init__()
        self.word_embeddings        = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings    = nn.Embedding(max_bert_length_size, hidden_size)
        self.token_type_embeddings  = nn.Embedding(2, hidden_size) # why 2 ? 0 and 1

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout   = nn.Dropout(hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(max_bert_length_size).expand((1, -1)))
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
            persistent=False,
        )

        # always absolute
        self.position_embedding_type = "absolute"

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


# this pooler is from huggingface
class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BERT_CONFIG():
    def __init__(self, vocab_size, padding_idx, max_seq_length,
                       d_model, layer_norm_eps, emb_hidden_dropout,
                       num_layers, num_heads, att_prob_dropout, dim_feedforward
                 ):
        # embedding
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        self.layer_norm_eps = layer_norm_eps
        self.emb_hidden_dropout = emb_hidden_dropout

        # attention
        self.num_layers=num_layers
        self.num_heads = num_heads
        self.att_prob_dropout = att_prob_dropout
        self.dim_feedforward = dim_feedforward


    def __init__(self, hg_config):
        # embedding
        self.vocab_size = hg_config.vocab_size
        self.padding_idx = hg_config.pad_token_id
        self.max_seq_length = hg_config.max_position_embeddings
        self.d_model = hg_config.hidden_size
        self.layer_norm_eps = hg_config.layer_norm_eps
        self.emb_hidden_dropout = hg_config.hidden_dropout_prob

        # attention
        self.num_layers= hg_config.num_hidden_layers
        self.num_heads = hg_config.num_attention_heads
        self.att_prob_dropout = hg_config.attention_probs_dropout_prob
        self.dim_feedforward = hg_config.intermediate_size





## We will wrap-all BERT sub-processing as BERT module
class BERT(nn.Module):
    def __init__(self, config):
        super().__init__()

        ## [Embeddings]
        self.embeddings = BertEmbeddings(
                                            config.vocab_size ,
                                            config.d_model,
                                            config.padding_idx,
                                            config.max_seq_length,
                                            config.layer_norm_eps,    # layer norm eps
                                            config.emb_hidden_dropout # 0.1
                                    )
        ## [Transformers]
        self.encoder = TransformerEncoder(
                                        num_layers=config.num_layers,
                                        d_model=config.d_model,
                                        num_heads=config.num_heads,
                                        dropout=config.att_prob_dropout,
                                        dim_feedforward=config.dim_feedforward
                                )


        ## [Pooler]
        self.pooler = BertPooler(config.d_model)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        attention_mask = attention_mask[:, None, None, :] # [B, 1, 1, seq_len]
        seq_embs   = self.embeddings(input_ids, token_type_ids)
        output     = self.encoder(seq_embs, attention_mask)
        pooled_out = self.pooler(output[0])

        seq_hidden_states = output[0]
        layer_attention_scores = output[1]
        return pooled_out, seq_hidden_states,layer_attention_scores

    def cp_encoder_block_weights_from_huggingface(self, src_encoder, tar_encoder):
        ## src: huggingface BERT model
        ## tar: my BERT model
        for layer_num, src_layer in enumerate(src_encoder.layer):
            # <<< to MultiHeadAttention (wq, wk, wv, wo) >>>
            cp_weight(src_layer.attention.self.query,   tar_encoder.layers[layer_num].self_attn.wq) # wq
            cp_weight(src_layer.attention.self.key,     tar_encoder.layers[layer_num].self_attn.wk) # wk
            cp_weight(src_layer.attention.self.value,   tar_encoder.layers[layer_num].self_attn.wv) # wv
            cp_weight(src_layer.attention.output.dense, tar_encoder.layers[layer_num].self_attn.wo) # wo

            # <<< to MLP (fc1, fc2) >>>
            cp_weight(src_layer.intermediate.dense, tar_encoder.layers[layer_num].fc1) # feed_forward_1
            cp_weight(src_layer.output.dense,       tar_encoder.layers[layer_num].fc2) # feed_forward_2

            # layer normalization parameters
            cp_weight(src_layer.attention.output.LayerNorm, tar_encoder.layers[layer_num].self_attn_layer_norm, include_eps=True) # norm_1
            cp_weight(src_layer.output.LayerNorm,           tar_encoder.layers[layer_num].final_layer_norm, include_eps=True) # norm_2

        return tar_encoder

    def copy_weights_from_huggingface(self, hg_bert):
        self.embeddings.load_state_dict( hg_bert.embeddings.state_dict() ,strict=False)
        self.pooler.load_state_dict( hg_bert.pooler.state_dict() ,strict=False)

        self.encoder = self.cp_encoder_block_weights_from_huggingface(
                                            src_encoder=hg_bert.encoder,
                                            tar_encoder=self.encoder
                                          )