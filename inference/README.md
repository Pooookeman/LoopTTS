# Refiner inference

This release runs the **Refiner only**: input speech + its transcript + a prosody instruction → refined speech. It does not run the Filter or Judge stages.

The checkpoint is the EmoVoice 1.5B based Refiner used by the project's March 2026 inference script (`epoch 31`, `step 7190`). Its SHA-256 is `995185563a1ae25c2fcc57e0e06b7e196252f8a5745b521902166093ab38580f`.

## Set up

From the LoopTTS repository root, use Python 3.10 and a CUDA capable machine:

```bash
bash inference/setup_emovoice.sh
python -m pip install -r third_party/EmoVoice/requirements.txt
python -m pip install huggingface_hub
```

The setup script pins the [upstream EmoVoice repository](https://github.com/yanghaha0908/EmoVoice) to commit `5285cb891611cf1ee2d9bd07b931cd3cf967cd64` and applies the Refiner changes preserved from the evaluated code. Upstream EmoVoice code retains its MIT license. The Refiner wrapper and patch in this repository use CC BY-NC 4.0.

Download the Refiner checkpoint and its two runtime dependencies into the repository:

```python
from huggingface_hub import snapshot_download

snapshot_download("Poookeman/LoopTTS-Refiner-1.5B", local_dir="models/refiner", allow_patterns="model.pt")
snapshot_download("Qwen/Qwen2.5-1.5B", local_dir="models/Qwen2.5-1.5B")
snapshot_download("FunAudioLLM/CosyVoice-300M-SFT", local_dir="models/CosyVoice-300M-SFT")
```

Place an input WAV at `inputs/example.wav`, then run:

```bash
python inference/refiner_infer.py \
  --input-wav inputs/example.wav \
  --text "What part of no do you not understand?" \
  --instruction "Speak with an angry emotion at a moderate speed and high pitch. Stress the word 'no'."
```

The output WAV appears at `runs/refiner/decode/pred_audio/neutral_prompt_speech/sample.wav`. The wrapper first extracts CosyVoice tokens from the input WAV, then calls the evaluated EmoVoice Refiner decoder with greedy decoding and `raw_audio_position=end`. It writes a one-line input JSONL with **relative paths** under `runs/refiner/`.

Use `--key` and `--output-dir` to distinguish runs. `--checkpoint`, `--qwen`, `--cosyvoice`, and `--emovoice` override the default model and source locations. Every file path supplied to the wrapper must be relative to the LoopTTS repository. Do not commit input audio or generated runs.

## Inputs and limitations

- The input audio must be a WAV of at most 25 seconds, with the same spoken words as `--text`.
- The instruction is passed through as the prompt. It may specify emotion, speed, pitch, word stress, and pauses.
- The input is encoded with the CosyVoice 300M SFT tokenizer; the decoder uses Qwen2.5-1.5B and CosyVoice 300M SFT. These dependencies are downloaded from their original publishers and are not redistributed here.
- This checkpoint is a PyTorch `model.pt` file and is loaded with `torch.load`. Use only the file from the published model repository or a source you trust.
- The original experiment used a GPU. CPU performance and alternative dependency versions have not been established for this release.

## Licenses and attribution

The Refiner wrapper, Refiner patch, and checkpoint are released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). The checkpoint was fine-tuned from [EmoVoice 1.5B](https://huggingface.co/yhaha/EmoVoice); its authors describe their pretrained models as noncommercial. The upstream EmoVoice source code is MIT licensed. [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) and [CosyVoice-300M-SFT](https://huggingface.co/FunAudioLLM/CosyVoice-300M-SFT) have their own licenses; review them before use.
