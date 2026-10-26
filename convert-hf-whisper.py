#!/usr/bin/env python3
"""Convert a Hugging Face Whisper fine-tune (model.safetensors) to whisper.cpp's ggml format.

The same conversion as whisper.cpp's models/convert-h5-to-ggml.py, but it reads the
safetensors file with numpy alone, so it needs neither torch nor transformers. The
output is f16 (small tensors f32, as whisper.cpp expects); quantize it afterwards with
whisper.cpp's whisper-quantize.

    python3 convert-hf-whisper.py <model dir> <mel_filters.npz> <out.bin>

<model dir> holds config.json, vocab.json and model.safetensors from the model's
repository. mel_filters.npz is whisper/assets/mel_filters.npz from openai/whisper.
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np

CONV_MAP = {
    "self_attn.k_proj": "attn.key",
    "self_attn.q_proj": "attn.query",
    "self_attn.v_proj": "attn.value",
    "self_attn.out_proj": "attn.out",
    "self_attn_layer_norm": "attn_ln",
    "encoder_attn.q_proj": "cross_attn.query",
    "encoder_attn.v_proj": "cross_attn.value",
    "encoder_attn.out_proj": "cross_attn.out",
    "encoder_attn_layer_norm": "cross_attn_ln",
    "fc1": "mlp.0",
    "fc2": "mlp.2",
    "final_layer_norm": "mlp_ln",
    "encoder.layer_norm.bias": "encoder.ln_post.bias",
    "encoder.layer_norm.weight": "encoder.ln_post.weight",
    "encoder.embed_positions.weight": "encoder.positional_embedding",
    "decoder.layer_norm.bias": "decoder.ln.bias",
    "decoder.layer_norm.weight": "decoder.ln.weight",
    "decoder.embed_positions.weight": "decoder.positional_embedding",
    "decoder.embed_tokens.weight": "decoder.token_embedding.weight",
    "proj_out.weight": "decoder.proj.weight",
}

DTYPES = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16}


def bytes_to_unicode():
    # GPT-2's reversible byte <-> unicode table, as in the original script.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def read_safetensors(path):
    """Yields (name, float32 array) in file order, memory-mapped."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    data = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + header_len)
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        raw = data[start:end].view(DTYPES[info["dtype"]])
        if info["dtype"] == "BF16":
            arr = (raw.astype(np.uint32) << 16).view(np.float32)
        else:
            arr = raw.astype(np.float32)
        yield name, arr.reshape(info["shape"])


def ggml_name(name):
    nn = name.split(".")[1:]  # drop the leading "model."
    if nn[1] == "layers":
        nn[1] = "blocks"
        if ".".join(nn[3:-1]) == "encoder_attn.k_proj":
            mapped = "attn.key" if nn[0] == "encoder" else "cross_attn.key"
        else:
            mapped = CONV_MAP[".".join(nn[3:-1])]
        return ".".join(nn[:3] + [mapped] + nn[-1:])
    joined = ".".join(nn)
    return CONV_MAP.get(joined, joined)


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    model_dir, mel_path, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])

    hparams = json.loads((model_dir / "config.json").read_text())
    max_length = hparams.get("max_length") or hparams.get("max_target_positions", 448)
    tokens = json.loads((model_dir / "vocab.json").read_text(encoding="utf8"))
    n_mels = hparams["num_mel_bins"]
    with np.load(mel_path) as f:
        filters = f[f"mel_{n_mels}"].astype(np.float32)

    with open(out_path, "wb") as out:
        out.write(struct.pack("i", 0x67676D6C))  # "ggml"
        for v in (
            hparams["vocab_size"],
            hparams["max_source_positions"],
            hparams["d_model"],
            hparams["encoder_attention_heads"],
            hparams["encoder_layers"],
            int(max_length),
            hparams["d_model"],
            hparams["decoder_attention_heads"],
            hparams["decoder_layers"],
            n_mels,
            1,  # f16
        ):
            out.write(struct.pack("i", v))

        out.write(struct.pack("ii", *filters.shape))
        out.write(filters.tobytes())

        byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
        out.write(struct.pack("i", len(tokens)))
        for text, _ in sorted(tokens.items(), key=lambda kv: kv[1]):
            raw = bytes(byte_decoder[c] for c in text)
            out.write(struct.pack("i", len(raw)))
            out.write(raw)

        count = 0
        for src, arr in read_safetensors(model_dir / "model.safetensors"):
            if src == "proj_out.weight":  # tied to the token embedding; unused
                continue
            name = ggml_name(src)
            data = arr.squeeze()
            if name in ("encoder.conv1.bias", "encoder.conv2.bias"):
                data = data.reshape(data.shape[0], 1)
            n_dims = data.ndim
            small = n_dims < 2 or name in (
                "encoder.conv1.bias",
                "encoder.conv2.bias",
                "encoder.positional_embedding",
                "decoder.positional_embedding",
            )
            # Through f16 first, as the original does, so f32 tensors match it exactly.
            data = data.astype(np.float16)
            ftype = 1
            if small:
                data = data.astype(np.float32)
                ftype = 0
            encoded = name.encode()
            out.write(struct.pack("iii", n_dims, len(encoded), ftype))
            for i in range(n_dims):
                out.write(struct.pack("i", data.shape[n_dims - 1 - i]))
            out.write(encoded)
            out.write(np.ascontiguousarray(data).tobytes())
            count += 1
    print(f"wrote {count} tensors to {out_path}")


if __name__ == "__main__":
    main()
