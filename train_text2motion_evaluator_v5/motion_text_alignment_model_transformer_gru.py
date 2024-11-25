import torch
import torch.nn as nn

import numpy as np

import clip

class TransformerEncoder(nn.Module):
    ## 263-4, 512, 1024
    def __init__(self):
        super(TransformerEncoder, self).__init__()
        seqtransencoderlayer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            seqtransencoderlayer,
            num_layers=6
        )

    def forward(self, inputs, masks):
        # import pdb; pdb.set_trace()
        outputs = self.transformer_encoder(inputs, src_key_padding_mask=~masks)
        return outputs

class MovementConvEncoder(nn.Module):    
    ## 263-4, 768, 768
    def __init__(self, input_size, hidden_size, output_size):
        super(MovementConvEncoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(512, hidden_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        # print(outputs.shape)
        return self.out_net(outputs)

class MotionEncoderBiGRUCo(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MotionEncoderBiGRUCo, self).__init__()

        self.input_emb = nn.Linear(input_size, hidden_size)
        self.num_layers = 3
        self.gru = nn.GRU(
            hidden_size, hidden_size, self.num_layers, batch_first=True, bidirectional=True
        )
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )

        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(
            torch.randn((self.num_layers*2, 1, self.hidden_size), requires_grad=True)
        )

    # input(batch_size, seq_len, dim)
    def forward(self, inputs):
        num_samples = inputs.shape[0]

        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        # emb = pack_padded_sequence(input=input_embs, lengths=cap_lens, batch_first=True)
        emb = input_embs

        gru_seq, gru_last = self.gru(emb, hidden)

        gru_last = torch.cat([gru_last[-2], gru_last[-1]], dim=-1)

        return self.output_net(gru_last)


class Motion_Text_Alignment_Model(nn.Module):
    def __init__(
        self, 

        nfeats,

        movement_input_size,
        movement_hidden_size,
        movement_output_size,

        motion_input_size,
        motion_hidden_size,
        motion_output_size,

        device,
        ):
        super().__init__()

        self.nfeats = nfeats
        self.device = device
        
        ## define transformer encoder
        self.transformerencoder = TransformerEncoder()

        ## define movement conv encoder
        self.movementconvencoder = MovementConvEncoder(
            movement_input_size,
            movement_hidden_size,
            movement_output_size,
            )

        ## define motion encoder bigruco
        self.motionencoderbigruco = MotionEncoderBiGRUCo(
            motion_input_size,
            motion_hidden_size,
            motion_output_size,
        )

        ## define clip text encoder
        self.clip = clip
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.text_convert = nn.Linear(512, 512)

        self.input_convert = nn.Linear(259, 512)
        self.encoder_convert = nn.Linear(512, 512)

    def forward(self, motion_feats, texts, masks):
        texts = self.clip.tokenize(texts).to(self.device)
        texts_feats = self.clip_model.encode_text(texts).to(self.device)

        t_feats = self.text_convert(texts_feats.to(torch.float32))

        bs, nframes, nfeats = motion_feats.shape
        out = self.input_convert(motion_feats[:,:,:-4])
        out = self.transformerencoder(out, masks.bool())
        out = self.movementconvencoder(out)
        out = self.motionencoderbigruco(out)

        m_feats = self.encoder_convert(out)
        
        return m_feats, t_feats