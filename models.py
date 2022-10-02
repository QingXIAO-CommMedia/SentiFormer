import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import TransformerEncoder, PositionEmbeddingSine


class GRU(nn.Module):
    """GRU model for word embeddings feature extraction.

    Args:
        in_dims (int): The word embedding input dims.
        hidden_dims (int): The RNN feedforward dims.
        dropout (float): The dropout rate. Default: 0.1
        txt_attn (True): Whether to set txt attention or not.
    """

    def __init__(self, in_dims, hidden_dims, num_layers=1, bidirection=True, dropout=0.1, txt_attn=True):
        super().__init__()
        self.in_dims = in_dims
        self.hidden_dims = hidden_dims
        self.bidirection = bidirection
        self.txt_attn = txt_attn

        self.gru = nn.GRU(
            input_size=in_dims,
            hidden_size=hidden_dims,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirection)

        if self.txt_attn:
            self.txt_attn = nn.Linear(hidden_dims, hidden_dims)
            self.txt_attn_act = nn.Tanh()
            self.dropout = nn.Dropout(dropout)

    def init_weights(self):
        # use orthogonal init for GRU layer0 weights
        nn.init.xavier_uniform_(self.gru.weight_ih_l0, nn.init.calculate_gain('tanh'))
        nn.init.orthogonal_(self.gru.weight_hh_l0, nn.init.calculate_gain('sigmoid'))
        # use zero init for GRU layer0 bias
        nn.init.constant_(self.gru.bias_ih_l0, 0)
        nn.init.constant_(self.gru.bias_hh_l0, 0)

    def forward(self, word_embed):
        """GRU forward function.

            word_embed usually use glove 300 dimensions representations. So,
        word embed shapes like (sent_len, 300). The sent len is set to 15.
        """
        out, _ = self.gru(word_embed)

        if self.bidirection:
            out = out[:, :, :self.hidden_dims] + out[:, :, self.hidden_dims:]

        out_word = out

        if self.txt_attn:
            attn_weight = self.txt_attn(out)
            attn_weight = self.txt_attn_act(attn_weight)
            attn_weight = self.dropout(attn_weight)
            attn_weight = F.softmax(attn_weight, dim=1)
            out = out * attn_weight
            out = torch.sum(out, dim=1, keepdim=True)

        return (out, out_word)


class SentiFormer(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.txt_backbone = GRU(300, 1024, bidirection=True, dropout=0.1, txt_attn=False)

        self.aggregation = TransformerEncoder(
            embed_dims=1024,
            feed_dims=1024,
            num_heads=8,
            num_layers=6,
            dropout=0.1,
        )
        N_steps = 1024
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        self.classifier = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.5, inplace=True), nn.Linear(256, num_classes))

    def forward(self, batched_input):
        txt = batched_input['txt']
        txt_gt = batched_input['txt_gt']

        txt_feats, txt_feats_word = self.txt_backbone(txt)
        pos = self.pe_layer(txt_feats)
        txt_feats = self.aggregation(txt_feats, None, pos)
        txt_feats = torch.mean(txt_feats, dim=1)
        logits = self.classifier(txt_feats)

        if self.training:
            losses = {}
            losses['loss_cls'] = F.cross_entropy(logits, txt_gt[:, 0])

            return losses
        else:

            return {'cls_logits': logits}
