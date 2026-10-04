# Standalone Refiner inference

This release runs the **Refiner only**: input WAV + its transcript + a prosody instruction → refined WAV. The inference logic lives in [`refiner_infer.py`](refiner_infer.py). It does not clone or patch EmoVoice and does not run the Filter or Judge stages.

The checkpoint is the EmoVoice 1.5B based Refiner used by the March 2026 inference script (epoch 31, step 7190). Its SHA-256 is `995185563a1ae25c2fcc57e0e06b7e196252f8a5745b521902166093ab38580f`.

## Set up

Use Python 3.10 and a CUDA capable machine. From the LoopTTS repository root:

```bash
mkdir -p third_party
# CosyVoice is the speech codec dependency; EmoVoice source is not needed.
git clone --recursive https://github.com/QwenAudio/CosyVoice.git third_party/CosyVoice
git -C third_party/CosyVoice checkout 1c062ab381535d787e798956c21b2785b9c95049
git -C third_party/CosyVoice submodule update --init --recursive
python -m pip install -r third_party/CosyVoice/requirements.txt
python -m pip install 'torch==2.4.1' 'torchaudio==2.4.1' 'transformers==4.43.4' 'soundfile==0.12.1' 'huggingface_hub==0.25.2'
```

The pinned CosyVoice source has the speech-token, speech-feature, and decoder interfaces used by the evaluated code. The last command selects the PyTorch and Transformers versions used by the evaluated EmoVoice environment. Install the appropriate CUDA build of PyTorch for your machine if the default wheel is CPU only. CosyVoice's own setup can require system packages; follow the [CosyVoice installation instructions](https://github.com/QwenAudio/CosyVoice#install) for those.

Download the three model assets into this repository:

```python
from huggingface_hub import snapshot_download

snapshot_download("Poookeman/LoopTTS-Refiner-1.5B", local_dir="models/refiner", allow_patterns="model.pt")
snapshot_download("Qwen/Qwen2.5-1.5B", local_dir="models/Qwen2.5-1.5B")
snapshot_download("FunAudioLLM/CosyVoice-300M-SFT", local_dir="models/CosyVoice-300M-SFT")
```

Place an input WAV at `inputs/example.wav`, then run from the repository root:

```bash
python inference/refiner_infer.py \
  --input-wav inputs/example.wav \
  --text "What part of no do you not understand?" \
  --instruction "Speak with an angry emotion at a moderate speed and high pitch. Stress the word 'no'."
```

The result is `runs/refiner/sample.wav` at 22,050 Hz. Use `--key` and `--output-dir` for other runs. `--checkpoint`, `--qwen`, `--cosyvoice`, and `--cosyvoice-source` change the default asset locations. All file paths on the command line must be relative to the LoopTTS repository and stay inside it.

## What the script does

1. CosyVoice 300M SFT encodes the input WAV into speech tokens.
2. The script packs the instruction, transcript, and original speech tokens in the checkpoint's evaluated `raw_audio_position=end` format.
3. The Refiner generates three audio-token rows with greedy decoding, a 1.2 repetition penalty, and a 3,000-token limit.
4. CosyVoice decodes the generated tokens using the input WAV as the voice prompt and saves a WAV.

The script runs this directly, without an intermediate JSONL or an EmoVoice subprocess.

## Inputs and limitations

- The input must be a WAV no longer than 25 seconds. `--text` should match its spoken words.
- The instruction can describe emotion, speed, pitch, stress, and pauses.
- The checkpoint is loaded with PyTorch `weights_only=True`; download it from the published model repository or another trusted source.
- The Refiner was evaluated on GPU. This standalone transcription of its inference path has **not yet been run against the released checkpoint**; its output equivalence remains to be checked. The command accepts `--device cpu`, but CPU performance is unknown.

## License and attribution

The LoopTTS Refiner code and checkpoint are released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Parts of the input packing and greedy decoding are adapted from [EmoVoice](https://github.com/yanghaha0908/EmoVoice), whose code is MIT licensed; see [third-party notices](THIRD_PARTY_NOTICES.md). The checkpoint is based on [EmoVoice 1.5B](https://huggingface.co/yhaha/EmoVoice), which its authors license for noncommercial use. [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) and [CosyVoice-300M-SFT](https://huggingface.co/FunAudioLLM/CosyVoice-300M-SFT) are separate dependencies with their own licenses.
