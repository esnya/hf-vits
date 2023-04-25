from transformers import PretrainedConfig


class VitsConfig(PretrainedConfig):
    def __init__(
        self,
        n_vocab: int | None = None,
        spec_channels: int | None = None,
        segment_size: int | None = None,
        inter_channels=192,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0.1,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[8, 8, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[16, 16, 4, 4],
        n_speakers=0,
        gin_channels=0,
        use_sdp=True,
        text_cleaners=["english_cleaners2"],
        add_blank=True,
        sampling_rate=22050,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.segment_size = segment_size
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.use_sdp = use_sdp
        self.text_cleaners = text_cleaners
        self.add_blank = add_blank
        self.sampling_rate = sampling_rate
