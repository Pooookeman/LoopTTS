#!/usr/bin/env python3
"""Run the released LoopTTS Refiner on one utterance.

Paths supplied on the command line are relative to the LoopTTS repository.
The generated JSONL also uses relative paths, so it can be moved with the run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def relative_existing_path(value: str, *, directory: bool = False) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise argparse.ArgumentTypeError("Use a path relative to the LoopTTS repository.")
    resolved = (REPO_ROOT / path).resolve()
    if not resolved.is_relative_to(REPO_ROOT):
        raise argparse.ArgumentTypeError("Path must stay inside the LoopTTS repository.")
    exists = resolved.is_dir() if directory else resolved.is_file()
    if not exists:
        raise argparse.ArgumentTypeError(f"Path does not exist: {value}")
    return resolved


def relative_output_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise argparse.ArgumentTypeError("Use a relative output path.")
    resolved = (REPO_ROOT / path).resolve()
    if not resolved.is_relative_to(REPO_ROOT):
        raise argparse.ArgumentTypeError("Output must stay inside the LoopTTS repository.")
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-wav", required=True, help="Input speech WAV, relative to this repository")
    parser.add_argument("--text", required=True, help="Words spoken in the input audio")
    parser.add_argument("--instruction", required=True, help="Natural-language prosody correction instruction")
    parser.add_argument("--checkpoint", default="models/refiner/model.pt")
    parser.add_argument("--qwen", default="models/Qwen2.5-1.5B")
    parser.add_argument("--cosyvoice", default="models/CosyVoice-300M-SFT")
    parser.add_argument("--emovoice", default="third_party/EmoVoice")
    parser.add_argument("--output-dir", default="runs/refiner")
    parser.add_argument("--key", default="sample", help="Output stem (letters, numbers, underscore, hyphen)")
    parser.add_argument("--device", default="0", help="CUDA device index visible to this process")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.key):
        parser.error("--key may contain only letters, numbers, underscore and hyphen")
    for field, is_dir in (("input_wav", False), ("checkpoint", False), ("qwen", True),
                          ("cosyvoice", True), ("emovoice", True)):
        try:
            setattr(args, field, relative_existing_path(getattr(args, field), directory=is_dir))
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
    try:
        args.output_dir = relative_output_path(args.output_dir)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    return args


def within_emovoice(path: Path, emovoice: Path) -> str:
    return os.path.relpath(path, emovoice)


def extract_speech_tokens(wav: Path, cosyvoice_dir: Path, emovoice: Path) -> list[int]:
    utils_dir = emovoice / "examples" / "tts" / "utils"
    sys.path.insert(0, str(utils_dir))
    sys.path.insert(0, str(utils_dir / "third_party" / "Matcha-TTS"))
    from cosyvoice.cli.cosyvoice import CosyVoice
    from cosyvoice.utils.file_utils import load_wav

    codec = CosyVoice(str(cosyvoice_dir), load_jit=False, load_trt=False, fp16=False)
    speech = load_wav(str(wav), target_sr=16000)
    if speech.shape[-1] / 16000 > 25:
        raise ValueError("Input audio exceeds the 25-second codec limit.")
    tokens, _ = codec.frontend._extract_speech_token(speech)
    return tokens[0].cpu().tolist()


def run_inference(args: argparse.Namespace, jsonl: Path) -> None:
    emovoice = args.emovoice
    decode_log = args.output_dir / "decode"
    rel = lambda path: within_emovoice(path, emovoice)
    overrides = {
        "hydra.run.dir": ".",
        "++model_config.llm_name": "qwen2.5-1.5b",
        "++model_config.llm_path": rel(args.qwen),
        "++model_config.llm_dim": "896",
        "++model_config.codec_decoder_path": rel(args.cosyvoice),
        "++model_config.codec_decode": "true",
        "++model_config.vocab_config.code_layer": "3",
        "++model_config.vocab_config.total_audio_vocabsize": "4160",
        "++model_config.vocab_config.total_vocabsize": "156160",
        "++model_config.codec_decoder_type": "CosyVoice",
        "++model_config.group_decode": "true",
        "++model_config.group_decode_adapter_type": "linear",
        "++model_config.use_text_stream": "false",
        "++dataset_config.dataset": "speech_dataset_tts",
        "++dataset_config.val_data_path": rel(jsonl),
        "++dataset_config.train_data_path": rel(jsonl),
        "++dataset_config.inference_mode": "true",
        "++dataset_config.vocab_config.code_layer": "3",
        "++dataset_config.vocab_config.total_audio_vocabsize": "4160",
        "++dataset_config.vocab_config.total_vocabsize": "156160",
        "++dataset_config.num_latency_tokens": "0",
        "++dataset_config.do_layershift": "false",
        "++dataset_config.tts_refine": "true",
        "++dataset_config.raw_audio_position": "end",
        "++train_config.model_name": "tts",
        "++train_config.freeze_encoder": "true",
        "++train_config.freeze_llm": "true",
        "++train_config.freeze_group_decode_adapter": "true",
        "++train_config.batching_strategy": "custom",
        "++train_config.num_epochs": "1",
        "++train_config.val_batch_size": "1",
        "++train_config.num_workers_dataloader": "0",
        "++decode_config.text_repetition_penalty": "1.2",
        "++decode_config.audio_repetition_penalty": "1.2",
        "++decode_config.max_new_tokens": "3000",
        "++decode_config.do_sample": "false",
        "++decode_config.top_p": "1.0",
        "++decode_config.top_k": "0",
        "++decode_config.temperature": "1.0",
        "++decode_config.decode_text_only": "false",
        "++decode_config.num_latency_tokens": "0",
        "++decode_config.do_layershift": "false",
        "++decode_log": rel(decode_log),
        "++ckpt_path": rel(args.checkpoint),
        "++output_text_only": "false",
        "++speech_sample_rate": "22050",
        "++log_config.log_file": rel(decode_log / "infer.log"),
    }
    command = [sys.executable, "examples/tts/inference_tts.py"] + [f"{key}={value}" for key, value in overrides.items()]
    env = os.environ.copy()
    env["PYTHONPATH"] = "src" + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["CUDA_VISIBLE_DEVICES"] = args.device
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["OMP_NUM_THREADS"] = "1"
    subprocess.run(command, cwd=emovoice, env=env, check=True)
    output_wav = decode_log / "pred_audio" / "neutral_prompt_speech" / f"{args.key}.wav"
    print(f"Generated audio: {output_wav.relative_to(REPO_ROOT)}")


def main() -> None:
    args = parse_args()
    if not (args.emovoice / "examples/tts/inference_tts.py").is_file():
        raise FileNotFoundError("EmoVoice source is incomplete. Run inference/setup_emovoice.sh first.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokens = extract_speech_tokens(args.input_wav, args.cosyvoice, args.emovoice)
    wav = within_emovoice(args.input_wav, args.emovoice)
    jsonl = args.output_dir / "input.jsonl"
    record = {
        "key": args.key,
        "source_text": args.text,
        "target_text": args.text,
        "suggestion": args.instruction,
        "raw_wav": wav,
        "target_wav": wav,
        "neutral_speaker_wav": wav,
        "raw_audio_token": tokens,
        "refined_audio_token": [],
    }
    jsonl.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    run_inference(args, jsonl)


if __name__ == "__main__":
    main()
