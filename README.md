HF VITS: VITS Implementation in HuggingFace Transformers
----

**[WIP] Working in progress now. Only works for text to speech evaluation now.**

This repository contains the implementation of the VITS model using the Hugging Face Transformers library. VITS (Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech) is a state-of-the-art end-to-end Text-to-Speech (TTS) model that generates more natural-sounding audio than current two-stage models.

## Overview

VITS, proposed by Jaehyeon Kim, Jungil Kong, and Juhee Son in their recent paper, is a parallel end-to-end TTS method. It combines variational inference with normalizing flows and adversarial training, which improves the expressive power of generative modeling. The model also includes a stochastic duration predictor for synthesizing speech with diverse rhythms from input text.

The main features of VITS include:
- Single-stage training and parallel sampling
- Improved sample quality compared to existing TTS models
- Expressive modeling of latent variables and stochastic duration prediction
- Natural one-to-many relationship representation between text input and speech output

In a subjective human evaluation (mean opinion score, or MOS) on the LJ Speech dataset, VITS outperformed the best publicly available TTS systems and achieved a MOS comparable to ground truth.

## Installation

You can install the hf-vits package directly from GitHub using pip:

```bash
pip install git+https://github.com/esnya/hf-vits.git
```

This implementation requires Python 3.10. Please ensure you have the correct Python version installed.

## Usage

To use the hf-vits model, import the necessary modules and load the pre-trained model:

```python
from hf_vits import VitsTokenizer, VitsForTextToSpeech

tokenizer = VitsTokenizer.from_pretrained("your-pretrained-model")
model = VitsForTextToSpeech.from_pretrained("your-pretrained-model")
```

Then, synthesize speech with the following code:

```python
text = "VITS is awesome."
inputs = tokenizer(text, return_tensors="pt").to(device)
generated_audio = model.generate(**inputs)
```

## Attribution

The code in hf_vits/models/utils and most of VitsForTextToSpeech is based on the original implementation published under the MIT License Copyright (c) 2023 liujing04. The code has been slightly modified by esnya.

## License

This implementation is released under the MIT License. See the [LICENSE](./LICENSE) for more details.
