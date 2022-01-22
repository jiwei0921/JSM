import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.transformer import DualTransformer


class TextNN(nn.Module):
    def __init__(self):
        super(TextNN, self).__init__()
        self.dropout = 0.1
        self.target_stride = 1

        self.conv = nn.Conv2d(3*32, 256, 1)
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.v_fc = nn.Linear(256, 256)     # visual_feat size: 256, hidden_size: 256

        self.word_fc = nn.Linear(300, 256)      # textual_feat size: 300,hidden_size: 256
        self.mask_vec = nn.Parameter(torch.zeros(300).float(), requires_grad=True)
        self.start_vec = nn.Parameter(torch.zeros(300).float(), requires_grad=True)
        self.trans = DualTransformer(dropout=0.1)
        self.fc_comp1 = nn.Linear(256, 256)
        self.fc_comp2 = nn.Linear(256, 150)      # classification size: 1 empty +  149 semantic words

        self.word_pos_encoder = SinusoidalPositionalEmbedding(300, 0, 20)

    def forward(self, visual_feat, words_feat, position, **kwargs):

        '''Visual Feature'''                                    # [10, 32*3, 352, 352]
        visual_feat = self.conv(visual_feat)                    # [10, 256, 352, 352]
        visual_feat = self.GAP(visual_feat)                     # [10, 256, 1, 1]
        visual_feat = torch.flatten(visual_feat,1)              # [10, 256]
        visual_feat = F.dropout(visual_feat, self.dropout, self.training)   # [10, 256]
        visual_feat = self.v_fc(visual_feat)                                # [10, 256]
        v_f = visual_feat.unsqueeze(1)
        visual_feat = torch.cat([v_f]*21,dim=1)                 # (b_s, 21, 256)


        '''Textual Feature'''
        cuda = torch.cuda.is_available()
        if cuda:
            words_feat[:, 0] = self.start_vec.cuda()
            self.mask_vec = self.mask_vec.cuda()
        else:
            words_feat[:, 0] = self.start_vec
        words_pos = self.word_pos_encoder(words_feat)  # [1, 21, 300]
        words_feat = F.dropout(words_feat, self.dropout, self.training)  # (b_s, 21, 300)
        words_mask_feat = self._mask_words(words_feat, mask_pos=position) + words_pos
        words_mask_feat = self.word_fc(words_mask_feat)  # (b_s, 21, 256)


        '''Cross-modal Feature Integration via the Transformer'''
        out = self.trans(visual_feat, words_mask_feat)          # [10, 21, 256]
        # Selecting the masked feature to perform the word prediction
        bsz = out.size(0)
        masked_feat = []
        for ii in range(bsz):
            pred_masked_feat = torch.index_select(out[ii,:,:], dim=0,index=position[ii])
            masked_feat.append(pred_masked_feat)

        out_feat = torch.cat(masked_feat,dim=0)
        out_feat = self.fc_comp1(out_feat)                                  # [10, 256]
        words_logit = self.fc_comp2(out_feat)                               # [10, 150]
        # print("out: {}".format(out_feat.size()))
        # print("words_logit: {}".format(words_logit.size()))

        return words_logit



    def _mask_words(self, words_feat, mask_pos=None):
        # words_feat (b_s, 20, 300), mask_pos (b_s, 1)
        masked_words = []
        bsz = mask_pos.size(0)
        for i in range(bsz):
            temp = [0]*21
            temp[mask_pos[i]] = 1
            masked_words.append(torch.tensor(temp))
        masked_words = torch.stack(masked_words, 0).unsqueeze(-1)  # (b_s, 1) -> (b_s, 21) -> (b_s, 21, 1)

        cuda = torch.cuda.is_available()
        if cuda:
            token = self.mask_vec.cuda().unsqueeze(0).unsqueeze(0)  # (1, 1, 300)
            masked_words = masked_words.cuda()
        else:
            token = self.mask_vec.unsqueeze(0).unsqueeze(0)

        masked_words_vec = words_feat.new_zeros(*words_feat.size()) + token # [128, 20, 300]
        # print("masked_words_vec: {}".format(masked_words_vec.shape))
        masked_words_vec = masked_words_vec.masked_fill_(masked_words == 0, 0)
        # print("masked_words_vec: {}".format(masked_words_vec.shape))
        words_feat1 = words_feat.masked_fill(masked_words == 1, 0) + masked_words_vec # (128, 20, 300)
        # print("words_feat1: {}".format(words_feat1.shape))
        return words_feat1



class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        import math
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, **kwargs):
        bsz, seq_len, _ = input.size()
        max_pos = seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        cuda = torch.cuda.is_available()
        if cuda:
            self.weights = self.weights.cuda(input.device)[:max_pos]
        else:
            self.weights = self.weights[:max_pos]
        return self.weights.unsqueeze(0)

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number
