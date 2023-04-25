import math

import monotonic_align
import torch
from torch import nn
from transformers import PreTrainedModel

from .configuration_vits import VitsConfig
from .utils import commons
from .utils.models import (
    DurationPredictor,
    Generator,
    PosteriorEncoder,
    ResidualCouplingBlock,
    StochasticDurationPredictor,
    TextEncoder,
)


class VitsForTextToSpeech(PreTrainedModel):
    """
    VITS Synthesizer: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech

    Original implementation: https://github.com/jaywalnut310/vits
    """

    config_class = VitsConfig

    def __init__(self, config: VitsConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        if config.n_vocab is None:
            raise ValueError("You must specify `n_vocab` to instantiate this model")
        if config.spec_channels is None:
            raise ValueError(
                "You must specify `spec_channels` to instantiate this model"
            )
        if config.segment_size is None:
            raise ValueError(
                "You must specify `segment_size` to instantiate this model"
            )

        self.n_vocab = config.n_vocab
        self.spec_channels = config.spec_channels
        self.inter_channels = config.inter_channels
        self.hidden_channels = config.hidden_channels
        self.filter_channels = config.filter_channels
        self.n_heads = config.n_heads
        self.n_layers = config.n_layers
        self.kernel_size = config.kernel_size
        self.p_dropout = config.p_dropout
        self.resblock = config.resblock
        self.resblock_kernel_sizes = config.resblock_kernel_sizes
        self.resblock_dilation_sizes = config.resblock_dilation_sizes
        self.upsample_rates = config.upsample_rates
        self.upsample_initial_channel = config.upsample_initial_channel
        self.upsample_kernel_sizes = config.upsample_kernel_sizes
        self.segment_size = config.segment_size
        self.n_speakers = config.n_speakers
        self.gin_channels = config.gin_channels

        self.use_sdp = config.use_sdp

        self.enc_p = TextEncoder(
            config.n_vocab,
            config.inter_channels,
            config.hidden_channels,
            config.filter_channels,
            config.n_heads,
            config.n_layers,
            config.kernel_size,
            config.p_dropout,
        )
        self.dec = Generator(
            config.inter_channels,
            config.resblock,
            config.resblock_kernel_sizes,
            config.resblock_dilation_sizes,
            config.upsample_rates,
            config.upsample_initial_channel,
            config.upsample_kernel_sizes,
            gin_channels=config.gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            config.spec_channels,
            config.inter_channels,
            config.hidden_channels,
            5,
            1,
            16,
            gin_channels=config.gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            config.inter_channels,
            config.hidden_channels,
            5,
            1,
            4,
            gin_channels=config.gin_channels,
        )

        if config.use_sdp:
            self.dp = StochasticDurationPredictor(
                config.hidden_channels, 192, 3, 0.5, 4, gin_channels=config.gin_channels
            )
        else:
            self.dp = DurationPredictor(
                config.hidden_channels, 256, 3, 0.5, gin_channels=config.gin_channels
            )

        if config.n_speakers > 1:
            self.emb_g = nn.Embedding(config.n_speakers, config.gin_channels)

        self.post_init()

    def forward(self, x, x_lengths, y, y_lengths, sid=None):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)

        with torch.no_grad():
            # negative cross-entropy
            s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
            neg_cent1 = torch.sum(
                -0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True
            )  # [b, 1, t_s]
            neg_cent2 = torch.matmul(
                -0.5 * (z_p**2).transpose(1, 2), s_p_sq_r
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent3 = torch.matmul(
                z_p.transpose(1, 2), (m_p * s_p_sq_r)
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent4 = torch.sum(
                -0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True
            )  # [b, 1, t_s]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = (
                monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1))
                .unsqueeze(1)
                .detach()
            )

        w = attn.sum(2)
        if self.use_sdp:
            l_length = self.dp(x, x_mask, w, g=g)
            l_length = l_length / torch.sum(x_mask)
        else:
            logw_ = torch.log(w + 1e-6) * x_mask
            logw = self.dp(x, x_mask, g=g)
            l_length = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(
                x_mask
            )  # for averaging

        # expand prior
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        o = self.dec(z_slice, g=g)
        return (
            o,
            l_length,
            attn,
            ids_slice,
            x_mask,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
        )

    def infer(
        self,
        x,
        x_lengths,
        sid=None,
        noise_scale=1.0,
        length_scale=1,
        noise_scale_w=1.0,
        max_len=None,
    ):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        if self.use_sdp:
            logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        else:
            logw = self.dp(x, x_mask, g=g)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
            x_mask.dtype
        )
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)
        return o, attn, y_mask, (z, z_p, m_p, logs_p)

    def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):
        assert self.n_speakers > 0, "n_speakers have to be larger than 0."
        g_src = self.emb_g(sid_src).unsqueeze(-1)
        g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
        z, _, _, y_mask = self.enc_q(y, y_lengths, g=g_src)
        z_p = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        o_hat = self.dec(z_hat * y_mask, g=g_tgt)
        return o_hat, y_mask, (z, z_p, z_hat)
