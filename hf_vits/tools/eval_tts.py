from os import PathLike

import pyaudio
import torch

from hf_vits.models.configuration_vits import VitsConfig

from ..models.modeling_vits_synthesizer import VitsForTextToSpeech
from ..models.utils.commons import intersperse
from ..text import text_to_sequence


def get_text(text, text_cleaners, add_blank):
    text_norm = text_to_sequence(text, text_cleaners)
    if add_blank:
        text_norm = intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def eval_tts(
    model_path: str | PathLike,
    text: str | None,
    device: torch.device | str = "cuda",
    fp16: bool = True,
):
    model = VitsForTextToSpeech.from_pretrained(model_path)
    assert isinstance(model, VitsForTextToSpeech)
    dtype = torch.float16 if fp16 else torch.float32
    model = model.to(device, dtype=dtype)

    config = VitsConfig.from_pretrained(model_path)
    pa = pyaudio.PyAudio()
    output_stream = pa.open(
        rate=config.sampling_rate,
        channels=1,
        output=True,
        format=pyaudio.paFloat32,
    )

    try:
        with torch.no_grad():
            while output_stream.is_active():
                text = text or input("> ")
                stn_tst = get_text(text, config.text_cleaners, config.add_blank)
                text = None
                x_tst = stn_tst.unsqueeze(0).to(device)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device, dtype)
                audio = (
                    model.infer(
                        x_tst,
                        x_tst_lengths,
                        noise_scale=0.667,
                        noise_scale_w=0.8,
                        length_scale=1,
                    )[0][0, 0]
                    .data.cpu()
                    .float()
                    .numpy()
                )
                output_stream.write(audio.tobytes())
    finally:
        output_stream.close()
        pa.terminate()


if __name__ == "__main__":
    from argh import ArghParser

    parser = ArghParser()
    parser.set_default_command(eval_tts)
    parser.dispatch()
