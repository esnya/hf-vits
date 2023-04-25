from os import PathLike

import pyaudio
import torch

from .. import VitsForTextToSpeech, VitsTokenizer


def eval_tts(
    model_path: str | PathLike,
    text: str | None,
    device: torch.device | str = "cuda",
    fp16: bool = True,
):
    model = VitsForTextToSpeech.from_pretrained(model_path)
    tokenizer = VitsTokenizer.from_pretrained(model_path)
    assert isinstance(model, VitsForTextToSpeech)

    dtype = torch.float16 if fp16 else torch.float32
    model = model.to(device, dtype=dtype)

    pa = pyaudio.PyAudio()
    output_stream = pa.open(
        rate=model.sampling_rate,
        channels=1,
        output=True,
        format=pyaudio.paFloat32,
    )

    try:
        with torch.no_grad():
            while output_stream.is_active():
                text = input("> ")
                inputs = tokenizer(text, return_tensors="pt").to(device)
                audio = (
                    model.generate(
                        **inputs,
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
