import torch
import torch.nn as nn

import numpy as np

class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, device, dropout=0.1, max_len=5000):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2, device=device).float()
      
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = self.encoding[:seq_len, :] + x
        return self.dropout(x)


class Motion_Text_Alignment_Model(nn.Module):
    def __init__(
        self, 

        nfeats,

        ## encoder parameters 
        encoder_latent_dim,
        encoder_ff_size,
        encoder_num_layers,
        encoder_num_heads,
        encoder_dropout,
        encoder_activation,
        
        ## decoder parameters
        decoder_latent_dim,
        decoder_ff_size,
        decoder_num_layers,
        decoder_num_heads,
        decoder_dropout,
        decoder_activation, 

        device
        ):
        super().__init__()

        self.nfeats = nfeats
        self.device = device

        ## define some hyper parameters for transformer encoder
        self.encoder_latent_dim = encoder_latent_dim
        self.encoder_ff_size = encoder_ff_size
        self.encoder_num_layers = encoder_num_layers
        self.encoder_num_heads = encoder_num_heads
        self.encoder_dropout = encoder_dropout
        self.encoder_activation = encoder_activation

        ## define some hyper parameters for transformer decoder
        self.decoder_latent_dim = decoder_latent_dim
        self.decoder_ff_size = decoder_ff_size
        self.decoder_num_layers = decoder_num_layers
        self.decoder_num_heads = decoder_num_heads
        self.decoder_dropout = decoder_dropout
        self.decoder_activation = decoder_activation

        ## input feats
        # self.input_feats = self.njoints * self.nfeats
        self.input_feats = self.nfeats
        self.encoder_input = nn.Linear(self.input_feats, self.encoder_latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.encoder_latent_dim, self.device, self.encoder_dropout)

        seqtransencoderlayer = nn.TransformerEncoderLayer(
            d_model=self.encoder_latent_dim,
            nhead=self.encoder_num_heads,
            dim_feedforward=self.encoder_ff_size,
            dropout=self.encoder_dropout,
            activation=self.encoder_activation
        )
        self.transformer_encoder = nn.TransformerEncoder(
            seqtransencoderlayer,
            num_layers=self.encoder_num_layers
        )

        seqtransdecoderlayer = nn.TransformerDecoderLayer(
            d_model=self.decoder_latent_dim,
            nhead=self.decoder_num_heads,
            dim_feedforward=self.decoder_ff_size,
            dropout=self.decoder_dropout,
            activation=self.decoder_activation
        )
        self.transformer_decoder = nn.TransformerDecoder(
            seqtransdecoderlayer,
            num_layers=self.decoder_num_layers
        )
        
        self.decoder_input = nn.Linear(self.input_feats, self.decoder_latent_dim)
        self.decoder_output = nn.Linear(self.decoder_latent_dim, self.input_feats)

        self.eos1 = nn.Parameter(torch.randn(1, self.encoder_latent_dim, device=self.device))
        self.eos2 = nn.Parameter(torch.randn(1, self.encoder_latent_dim, device=self.device))

    def forward(self, motion_feats, motion_masks):
        # print(motion_feats.shape)
        bs, nframes, nfeats = motion_feats.shape
        m_feats = motion_feats.permute(1, 0, 2)

        m = self.encoder_input(m_feats)
        m = self.sequence_pos_encoder(m)

        # m = self.transformer_encoder(m, src_key_padding_mask=~motion_masks.bool())

        # m_avg = m.mean(axis=0)
        eos1 = self.eos1.unsqueeze(0).repeat(1, bs, 1)
        eos2 = self.eos2.unsqueeze(0).repeat(1, bs, 1)
        mseq = torch.cat((eos1, eos2, m), axis=0)

        mseq = self.sequence_pos_encoder(mseq)

        newmask = torch.ones((bs, 2), dtype=bool, device=self.device)
        newmask = torch.cat((newmask, motion_masks.bool()), axis=1)

        encoder_outs = self.transformer_encoder(mseq, src_key_padding_mask=~newmask)
        m_vector = encoder_outs[0]

        timequeries = torch.zeros(nframes, bs, self.decoder_latent_dim, device=self.device)
        timequeries = self.sequence_pos_encoder(timequeries)

        decoder_outputs = self.transformer_decoder(timequeries, m_vector.unsqueeze(0), tgt_key_padding_mask=~motion_masks.bool())
        # decoder_outputs = self.transformer_decoder(timequeries, m_vector, tgt_key_padding_mask=~motion_masks.bool())
        decoder_outputs = self.decoder_output(decoder_outputs) # .permute(1, 0, 2)
        #print(m_feats.shape, decoder_outputs.shape)
        
        return motion_feats, decoder_outputs.permute(1,0,2)