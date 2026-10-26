# dohra-models

Speech models converted for [whisper.cpp](https://github.com/ggml-org/whisper.cpp), downloaded by
Dohra's transcription companion. Each file is published as a release asset and pinned by SHA-256
in the companion's source.

## ggml-hindi2hinglish-apex-q5_0.bin

Dohra's **High** model. It writes Hindi and Hinglish speech in Roman script ("theek hai") and
leaves English as English.

| | |
|---|---|
| Source | [Oriserve/Whisper-Hindi2Hinglish-Apex](https://huggingface.co/Oriserve/Whisper-Hindi2Hinglish-Apex), a fine-tune of Whisper large-v3-turbo |
| Source file | `model.safetensors`, SHA-256 `5560830fe8fa09b8cdf1b5926d0cc39a2586eadaad685d809e156432c4ebca80` |
| License | Apache 2.0, as the source model. Credit for the model belongs to Oriserve. |
| Conversion | [convert-hf-whisper.py](convert-hf-whisper.py) to f16, then whisper.cpp's `whisper-quantize ... q5_0` |
| Size | 574,041,195 bytes |
| SHA-256 | `676c257cd9eaf065328e9432cb2bd8465146920f4e4e6783743c0562b61d5104` |

Use it with `-l en` and no prompt. Told `-l hi`, it turns English speech into a loop of
Devanagari letters.

To reproduce the file:

```sh
# from https://huggingface.co/Oriserve/Whisper-Hindi2Hinglish-Apex: config.json, vocab.json, model.safetensors
# from https://github.com/openai/whisper: whisper/assets/mel_filters.npz
python3 convert-hf-whisper.py apex/ mel_filters.npz ggml-apex-f16.bin
whisper-quantize ggml-apex-f16.bin ggml-hindi2hinglish-apex-q5_0.bin q5_0
```
