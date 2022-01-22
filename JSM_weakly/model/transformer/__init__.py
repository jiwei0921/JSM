import torch.nn as nn
from model.transformer.mutihead_attention import MultiheadAttention
from model.transformer.decoder import TransformerDecoder
from model.transformer.encoder import TransformerEncoder



class DualTransformer(nn.Module):
    def __init__(self, d_model=256, num_heads=4, num_decoder_layers1=3,dropout=0.0):
        super().__init__()
        self.decoder1 = TransformerDecoder(num_decoder_layers1, d_model, num_heads, dropout)

    def forward(self, vis_fea, masked_text_fea):
        out = self.decoder1(vis_fea, masked_text_fea)
        return out
