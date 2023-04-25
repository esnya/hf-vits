import logging
from os import PathLike

import torch

from ..models.configuration_vits import VitsConfig
from ..models.modeling_vits_synthesizer import VitsForTextToSpeech

logger = logging.getLogger(__name__)


def fix_key(key: str) -> str:
    if key.endswith(".gamma"):
        return key.replace(".gamma", ".weight")
    if key.endswith(".beta"):
        return key.replace(".beta", ".bias")
    return key


def convert_from_pth(
    pth_path: str | PathLike,
    save_directory: str | PathLike,
    safe_serialization: bool = True,
    log_level: str = "INFO",
):
    "Convert a checkpoint from the original repo to a format compatible with this repo."

    logging.basicConfig(level=log_level)

    logger.info("Loading checkpoint from %s", pth_path)
    checkpoint = torch.load(pth_path, map_location="cpu", weights_only=True)
    assert isinstance(checkpoint, dict)

    model_state = {fix_key(key): value for key, value in checkpoint["model"].items()}
    assert isinstance(model_state, dict)

    logger.info("Building config")

    channels = model_state["enc_p.encoder.attn_layers.0.conv_q.weight"].shape[0]
    k_channels = model_state["enc_p.encoder.attn_layers.0.emb_rel_k"].shape[2]
    n_heads = channels // k_channels

    hidden_channels = model_state["enc_p.emb.weight"].shape[1]

    n_speakers = (
        model_state["emb_g.weight"].shape[0] if "emb_g.weight" in model_state else 0
    )
    gin_channels = (
        model_state["dec.cond.weight"].shape[1]
        if "dec.cond.weight" in model_state
        else 0
    )

    config = VitsConfig(
        n_vocab=model_state["enc_p.emb.weight"].shape[0],
        spec_channels=model_state["enc_q.pre.weight"].shape[0],
        segment_size=1,
        hidden_channels=hidden_channels,
        n_heads=n_heads,
        n_speakers=n_speakers,
        gin_channels=gin_channels,
    )

    logger.info("Building model")
    model = VitsForTextToSpeech(config)

    # Why shape is mismatched?
    model_state["enc_q.pre.weight"] = model_state["enc_q.pre.weight"][
        :, :hidden_channels, :
    ]

    model.load_state_dict(model_state)

    logger.info("Saving model to %s", save_directory)
    model.save_pretrained(save_directory, safe_serialization=safe_serialization)

    logger.info("Testing saved model")
    VitsForTextToSpeech.from_pretrained(save_directory)


if __name__ == "__main__":
    from argh import ArghParser

    parser = ArghParser()
    parser.set_default_command(convert_from_pth)
    parser.dispatch()
