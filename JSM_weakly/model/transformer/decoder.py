import torch
import torch.nn as nn
import torch.nn.functional as F

from model.transformer import MultiheadAttention


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    # def forward(self, src, src_mask, tgt, tgt_mask):
    def forward(self, vis, text):
        non_pad_src_mask = None
        non_pad_tgt_mask = None

        if vis is not None:
            vis = vis.transpose(0, 1)

        x = text.transpose(0, 1)
        for layer in self.decoder_layers:
            x, weight = layer(x, non_pad_tgt_mask,
                              vis, non_pad_src_mask,
                              self.buffered_future_mask(x))
        return x.transpose(0, 1)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        d_model = d_model
        num_heads = num_heads
        self.dropout = dropout
        self.self_attn = MultiheadAttention(d_model, num_heads)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.encoder_attn = MultiheadAttention(d_model, num_heads)
        self.encoder_attn_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model << 1)
        self.fc2 = nn.Linear(d_model << 1, d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.fc_cat = nn.Linear(d_model*2, d_model)
        self.fc_cross = nn.Linear(d_model*3, d_model)

    def forward(self, x, mask, encoder_out=None, encoder_mask=None, self_attn_mask=None):
        # x: textual feature  encoder_out: visual feature

        res = x
        x, weight = self.self_attn(x, x, x, mask, attn_mask=self_attn_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = res + x
        x = self.self_attn_layer_norm(x)

        # Cross modal fusion between textual and visual features
        '''
        if encoder_out is not None:
            res = x
            x, weight = self.encoder_attn(x, encoder_out, encoder_out, encoder_mask)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = res + x
            x = self.encoder_attn_layer_norm(x)
        '''
        if encoder_out is not None:
            res = x
            # Element-wise Addition
            x1 = x + encoder_out
            # Element-wise Multiplication
            x2 = x * encoder_out
            # Concat and FC Operations
            x3 = self.fc_cat(torch.cat([x, encoder_out],dim=2))
            x = self.fc_cross(torch.cat([x1, x2, x3],dim=2))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = res + x
            x = self.encoder_attn_layer_norm(x)

        res = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = res + x
        x = self.final_layer_norm(x)
        return x, weight
