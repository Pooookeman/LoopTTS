#!/usr/bin/env python3
"""Standalone LoopTTS Refiner inference for one WAV and one instruction.

The input format and greedy decoder follow the evaluated checkpoint. This
implementation is adapted from EmoVoice (MIT); see THIRD_PARTY_NOTICES.md.
All command-line paths are relative to the LoopTTS repository root.
"""

from __future__ import annotations

import argparse
import re
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_LAYERS = 3
TEXT_VOCAB = 151936
TEXT_PADDED = 152000
AUDIO_VOCAB = 4096
AUDIO_PADDED = 4160
AUDIO_SHIFT = TEXT_PADDED
TOTAL_VOCAB = TEXT_PADDED + AUDIO_PADDED
EOT, PAD_T, INPUT_T, ANSWER_T = (TEXT_VOCAB + i for i in range(4))
EOA, PAD_A, INPUT_A, ANSWER_A = (AUDIO_VOCAB + i for i in range(4))


def repository_path(value: str, *, kind: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise argparse.ArgumentTypeError("Use paths relative to the LoopTTS repository.")
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise argparse.ArgumentTypeError("Paths must stay inside the LoopTTS repository.")
    if kind == "file" and not resolved.is_file():
        raise argparse.ArgumentTypeError(f"File does not exist: {value}")
    if kind == "directory" and not resolved.is_dir():
        raise argparse.ArgumentTypeError(f"Directory does not exist: {value}")
    return resolved


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-wav", required=True, help="Input WAV, relative to this repository")
    parser.add_argument("--text", required=True, help="Exact words spoken in the input WAV")
    parser.add_argument("--instruction", required=True, help="Prosody correction instruction")
    parser.add_argument("--checkpoint", default="models/refiner/model.pt")
    parser.add_argument("--qwen", default="models/Qwen2.5-1.5B")
    parser.add_argument("--cosyvoice", default="models/CosyVoice-300M-SFT")
    parser.add_argument("--cosyvoice-source", default="third_party/CosyVoice")
    parser.add_argument("--output-dir", default="runs/refiner")
    parser.add_argument("--key", default="sample", help="Output filename without .wav")
    parser.add_argument("--device", default="cuda:0", help="PyTorch device, e.g. cuda:0 or cpu")
    parser.add_argument("--max-new-tokens", type=int, default=3000)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.key):
        parser.error("--key may contain only letters, numbers, underscore, or hyphen")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    for name, kind in (("input_wav", "file"), ("checkpoint", "file"),
                       ("qwen", "directory"), ("cosyvoice", "directory"),
                       ("cosyvoice_source", "directory"), ("output_dir", "output")):
        try:
            setattr(args, name, repository_path(getattr(args, name), kind=kind))
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
    return args


def load_codec(source: Path, weights: Path):
    # CosyVoice is an independent dependency, not an EmoVoice checkout.
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "third_party" / "Matcha-TTS"))
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice
    except ImportError as exc:
        raise RuntimeError("Install CosyVoice as described in inference/README.md") from exc
    return CosyVoice(str(weights), load_jit=False, load_trt=False, fp16=False)


def speech_tokens(codec, wav: Path) -> tuple[list[int], object]:
    from cosyvoice.utils.file_utils import load_wav

    speech_16k = load_wav(str(wav), target_sr=16000)
    if speech_16k.shape[-1] / 16000 > 25:
        raise ValueError("Input WAV exceeds the 25-second CosyVoice tokenizer limit.")
    tokens, _ = codec.frontend._extract_speech_token(speech_16k)
    return tokens[0].cpu().tolist(), speech_16k


def input_ids(tokenizer, text: str, instruction: str, raw_tokens: list[int], torch):
    # Evaluated ordering: instruction, transcript, then interleaved input audio.
    instruction = re.sub(r"[。！？\.,!\?]$", "", instruction)
    instruction = re.sub(r"\.(?=.)", ",", instruction)
    prompt_tokens = [INPUT_T] + tokenizer.encode(f"<SYSTEM>: {instruction}\n ") + [EOT]
    text_tokens = tokenizer.encode(text)
    flat_audio = raw_tokens + [EOA]
    flat_audio += [PAD_A] * (-len(flat_audio) % CODE_LAYERS)
    rows = []
    for layer in range(CODE_LAYERS):
        row = ([AUDIO_SHIFT + PAD_A] * len(prompt_tokens)
               + [AUDIO_SHIFT + INPUT_A]
               + [AUDIO_SHIFT + PAD_A] * len(text_tokens)
               + [AUDIO_SHIFT + EOA, AUDIO_SHIFT + ANSWER_A]
               + [AUDIO_SHIFT + token for token in flat_audio[layer::CODE_LAYERS]])
        rows.append(row)
    rows.append(prompt_tokens + [INPUT_T] + text_tokens + [EOT, ANSWER_T]
                + [PAD_T] * (len(flat_audio) // CODE_LAYERS))
    return torch.tensor(rows, dtype=torch.long).unsqueeze(0)


def load_refiner(qwen_path: Path, checkpoint: Path, device, torch):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(qwen_path))
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.add_special_tokens({"additional_special_tokens":
                                  ["<strong>", "</strong>", "[breath]"]})
    llm = AutoModelForCausalLM.from_pretrained(str(qwen_path))
    llm.resize_token_embeddings(TOTAL_VOCAB)

    class Refiner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.llm = llm
            self.group_decode_adapter = torch.nn.Module()
            self.group_decode_adapter.linear = torch.nn.Linear(
                AUDIO_PADDED, CODE_LAYERS * AUDIO_PADDED)

    model = Refiner()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("Expected a state-dict checkpoint in model.pt.")
    result = model.load_state_dict(state, strict=False)
    # Qwen ties lm_head to embed_tokens, so the checkpoint stores only one copy.
    required = ("llm.model.embed_tokens.weight",
                "group_decode_adapter.linear.weight", "group_decode_adapter.linear.bias")
    absent = [name for name in required if name not in state or name in result.missing_keys]
    if absent:
        raise ValueError(f"Checkpoint is missing Refiner weights: {', '.join(absent)}")
    del state
    model.to(device).eval()
    return model, tokenizer


def penalize_repetition(logits, previous: list[int], factor: float, torch):
    if previous:
        indexes = torch.tensor(previous, device=logits.device)
        values = logits[indexes]
        logits[indexes] = torch.where(values < 0, values * factor, values / factor)
    return logits


def generate_tokens(model, ids, max_new_tokens: int, torch):
    llm = model.llm
    device = ids.device
    mask = torch.ones((1, ids.shape[-1]), device=device, dtype=torch.long)
    past = None
    audio = [[] for _ in range(CODE_LAYERS)]
    text = []
    text_done = False
    with torch.inference_mode():
        embeds = llm.model.embed_tokens(ids).mean(dim=1)
        for _ in range(max_new_tokens):
            output = llm(inputs_embeds=embeds, attention_mask=mask,
                         past_key_values=past, use_cache=True)
            past = output.past_key_values
            logits = output.logits[0, -1]
            text_logits = penalize_repetition(logits[:TEXT_PADDED].clone(), text, 1.2, torch)
            audio_logits = model.group_decode_adapter.linear(logits[TEXT_PADDED:])
            next_text = PAD_T if text_done else int(text_logits.argmax())
            next_audio = []
            for layer in range(CODE_LAYERS):
                start = layer * AUDIO_PADDED
                layer_logits = penalize_repetition(
                    audio_logits[start:start + AUDIO_PADDED].clone(), audio[layer], 1.2, torch)
                next_audio.append(int(layer_logits.argmax()))
            text.append(next_text)
            for layer, token in enumerate(next_audio):
                audio[layer].append(token)
            text_done = text_done or next_text == EOT
            if EOA in next_audio:
                return audio
            next_ids = torch.tensor([AUDIO_SHIFT + token for token in next_audio]
                                    + [next_text], device=device)
            embeds = llm.model.embed_tokens(next_ids).mean(dim=0).reshape(1, 1, -1)
            mask = torch.cat((mask, torch.ones((1, 1), device=device, dtype=mask.dtype)), dim=1)
    raise RuntimeError("Refiner reached --max-new-tokens before emitting end-of-audio.")


def decode_audio(codec, generated: list[list[int]], speech_16k, torch):
    import torchaudio

    flat = torch.tensor(generated, dtype=torch.long).transpose(0, 1).reshape(-1)
    eoa = (flat == EOA).nonzero(as_tuple=True)[0]
    if eoa.numel() == 0:
        raise RuntimeError("Generated audio has no end-of-audio token.")
    flat = flat[:int(eoa[0])]
    if flat.numel() == 0:
        raise RuntimeError("Refiner emitted end-of-audio before speech tokens.")
    flat = flat.masked_fill(flat == PAD_A, 4095).unsqueeze(0)
    prompt_token, _ = codec.frontend._extract_speech_token(speech_16k)
    speech_22050 = torchaudio.functional.resample(speech_16k, 16000, 22050)
    prompt_feat, _ = codec.frontend._extract_speech_feat(speech_22050)
    embedding = codec.frontend._extract_spk_embedding(speech_16k)
    with torch.inference_mode():
        return codec.model.token2wav(token=flat, prompt_token=prompt_token,
                                     prompt_feat=prompt_feat, embedding=embedding,
                                     uuid=str(uuid.uuid4()), finalize=True, speed=1.0)


def main() -> None:
    args = arguments()
    import soundfile as sf
    import torch

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu to run on CPU.")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    codec = load_codec(args.cosyvoice_source, args.cosyvoice)
    raw_tokens, speech_16k = speech_tokens(codec, args.input_wav)
    model, tokenizer = load_refiner(args.qwen, args.checkpoint, device, torch)
    ids = input_ids(tokenizer, args.text, args.instruction, raw_tokens, torch).to(device)
    generated = generate_tokens(model, ids, args.max_new_tokens, torch)
    waveform = decode_audio(codec, generated, speech_16k, torch)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.key}.wav"
    sf.write(output, waveform.squeeze().cpu().numpy(), 22050)
    print(f"Generated audio: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
