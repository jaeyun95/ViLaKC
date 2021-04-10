"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Uniter for VCR model
"""
from collections import defaultdict
import torch
from torch.nn.modules import BatchNorm1d
from torch import nn
from torch.nn import functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
from .attention import MultiheadAttention
# from .layer import GELU
from .model import (
    UniterPreTrainedModel, UniterModel)
import math
from .layer import BertPooler

class UniterForVisualCommonsenseReasoning(UniterPreTrainedModel):
    """ Finetune UNITER for VCR
    """
    def __init__(self, config, img_dim):
        super().__init__(config, img_dim)
        self.uniter = UniterModel(config, img_dim)
        self.vcr_output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            nn.ReLU(),
            LayerNorm(config.hidden_size*2, eps=1e-12),
            nn.Linear(config.hidden_size*2, 2)
        )
        self.apply(self.init_weights)
        #self.final_feat_pool = nn.AdaptiveMaxPool1d((768))
        #self.attention = DotProductAttention(dropout=0.1)
        #self.dense = nn.Sequential(
        #    nn.Linear(772, 768),
        #    nn.ReLU(),
        #    LayerNorm(768, eps=1e-12),
        #    nn.Linear(768, 768)
        #)
        #self.activation = nn.Tanh()

    def init_type_embedding(self):
        new_emb = nn.Embedding(4, self.uniter.config.hidden_size)
        new_emb.apply(self.init_weights)
        for i in [0, 1]:
            emb = self.uniter.embeddings.token_type_embeddings.weight.data[i, :]
            new_emb.weight.data[i, :].copy_(emb)
        emb = self.uniter.embeddings.token_type_embeddings.weight.data[0, :]
        new_emb.weight.data[2, :].copy_(emb)
        new_emb.weight.data[3, :].copy_(emb)
        self.uniter.embeddings.token_type_embeddings = new_emb

    def init_type_embedding_know(self):
        new_emb = nn.Embedding(5, self.uniter.config.hidden_size)
        new_emb.apply(self.init_weights)
        for i in [0, 1]:
            emb = self.uniter.know_embeddings.token_type_embeddings.weight.data[i, :]
            new_emb.weight.data[i, :].copy_(emb)
        emb = self.uniter.know_embeddings.token_type_embeddings.weight.data[0, :]
        new_emb.weight.data[2, :].copy_(emb)
        new_emb.weight.data[3, :].copy_(emb)
        new_emb.weight.data[4, :].copy_(emb)
        self.uniter.know_embeddings.token_type_embeddings = new_emb

    def init_word_embedding(self, num_special_tokens):
        orig_word_num = self.uniter.embeddings.word_embeddings.weight.size(0)
        new_emb = nn.Embedding(
            orig_word_num + num_special_tokens, self.uniter.config.hidden_size)
        new_emb.apply(self.init_weights)
        emb = self.uniter.embeddings.word_embeddings.weight.data
        new_emb.weight.data[:orig_word_num, :].copy_(emb)
        self.uniter.embeddings.word_embeddings = new_emb

    def forward(self, batch, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        txt_type_ids = batch['txt_type_ids']
        node_feat =  batch['node_feat']
        adj_feat = batch['adj_feat']
        know_type_ids = batch['know_type_ids']
        know_pos_feat = batch['know_pos_feat']
        attr_feat = batch['attr_feat']
        obj_feat = batch['obj_feat']
        new_box = batch['new_box']

        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, node_feat,
                                      adj_feat, know_pos_feat, gather_index,
                                      output_all_encoded_layers=False,
                                      txt_type_ids=txt_type_ids, know_type_ids=know_type_ids)


        pooled_output = self.uniter.pooler(sequence_output)
        #added_info = torch.cat([obj_feat,attr_feat],dim=2)
        #add_info = torch.cat([attr_feat,new_box],dim=2)
        #### concat version
        #know_emb = know_emb.view(know_emb.shape[0],know_emb.shape[1]*know_emb.shape[2])
        #know_emb = know_emb.unsqueeze(1)
        #pooled_know = self.final_know_pool(know_emb)
        #pooled_know = pooled_know.squeeze()
        #final_pooled = torch.cat([pooled_output,pooled_know],dim=1)
        #rank_scores = self.vcr_output(final_pooled)

        #### attention version
        #info_feat = self.dense(add_info)
        #result = self.attention(sequence_output, info_feat, info_feat, torch.tensor([info_feat.shape[0]]))
        #result = info_feat[:, 0]
        #print("@@@@@@@@",result.shape)
        #info_feat = add_info.view(add_info.shape[0],add_info.shape[1]*add_info.shape[2])
        #info_feat = info_feat.unsqueeze(1)
        #pooled_result = self.final_feat_pool(info_feat)
        #pooled_result = self.activation(pooled_result)
        #pooled_result = pooled_result.squeeze()
        #pooled_output = self.uniter.pooler(sequence_output)
        #print("@@@@@",pooled_output.shape,"----",pooled_result.shape)
        #final_result = torch.cat((pooled_output, pooled_result), dim=1)

        ## original version
        rank_scores = self.vcr_output(pooled_output)

        if compute_loss:
            targets = batch['targets']
            vcr_loss = F.cross_entropy(
                    rank_scores, targets.squeeze(-1),
                    reduction='mean')
            return vcr_loss
        else:
            rank_scores = rank_scores[:, 1:]
            return rank_scores


class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # `query`: (`batch_size`, #queries, `d`)
    # `key`: (`batch_size`, #kv_pairs, `d`)
    # `value`: (`batch_size`, #kv_pairs, `dim_v`)
    # `valid_len`: either (`batch_size`, ) or (`batch_size`, xx)
    def forward(self, query, key, value, valid_len=None):
        d = query.shape[-1]
        # Set transpose_b=True to swap the last two dimensions of key
        scores = torch.bmm(query, key.transpose(1,2)) / math.sqrt(d)
        attention_weights = self.dropout(nn.functional.softmax(scores))
        return torch.bmm(attention_weights, value)