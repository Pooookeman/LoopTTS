<div align="center">

# LoopTTS

### Diagnose, Then Refine: A Closed-Loop TTS System with AudioLLM-Guided Correction

**EMNLP 2026 Main Conference**

<img alt="Paper coming soon" src="https://img.shields.io/badge/Paper-Coming_Soon-6B7280?style=flat-square">
<a href="https://pooookeman.github.io/LoopTTS/"><img alt="Audio Demo" src="https://img.shields.io/badge/Audio_Demo-Listen-2F855A?style=flat-square&logo=githubpages&logoColor=white"></a>
<img alt="Training coming soon" src="https://img.shields.io/badge/Training-Coming_Soon-6B7280?style=flat-square">
<a href="https://huggingface.co/datasets/Poookeman/Refiner-DB"><img alt="Refiner-DB dataset" src="https://img.shields.io/badge/Dataset-Refiner--DB-FFD21E?style=flat-square&logo=huggingface&logoColor=black"></a>

</div>

## Overview

LoopTTS is a closed-loop quality-control framework for recovering TTS outputs with local prosodic defects. It organizes generation and correction as a **Filter--Judge--Refiner** pipeline:

1. **Filter:** identifies severe content or quality failures using coarse-grained metrics.
2. **Judge:** an AudioLLM diagnoses salient prosodic issues and produces structured refine instructions.
3. **Refiner:** a purpose-trained TTS model performs guided expressive re-synthesis conditioned on the initial utterance, target text, and instruction.

The refine instruction combines global attributes such as emotion, speed, and pitch with local stress and pause cues. The Refiner is trained on **Refiner-DB**, a roughly 42K-example dataset with AudioLLM-derived word-level prosodic supervision.

## Resources

| Resource | Status | Link |
|---|---|---|
| Paper | Coming soon | arXiv |
| Audio demo | Available | [Project page](https://pooookeman.github.io/LoopTTS/) |
| Training framework | Coming soon | This repository |
| Refiner-DB | Available | [Hugging Face dataset](https://huggingface.co/datasets/Poookeman/Refiner-DB) |

## Audio Demo

The [project page](https://pooookeman.github.io/LoopTTS/) contains examples of:

- joint emotion and word-level stress/pause control;
- speed and pitch control from the same initial utterance;
- AudioLLM-generated refine instructions;
- before/after refinement comparisons from the full LoopTTS pipeline.

## Training Framework

The Refiner training framework and configuration files are being organized for release. They will be added to this repository without changing the project URL.

## Refiner-DB

[Refiner-DB](https://huggingface.co/datasets/Poookeman/Refiner-DB) contains **42,338 processed training examples** with Gemini-generated refinement suggestions and a separate **200-example human-annotated test set**. The JSONL records retain the full utterance text and use relative paths to the included WAV files. Training and test data have separate schemas and are available as the `gemini_train` and `human_test` dataset configurations.

The dataset card documents the fields, audio archives, download and extraction steps, source corpora, and checksums. The preliminary 42,420-example collection is not part of this release.

## Citation

```bibtex
@inproceedings{song2026loopt,
  title     = {Diagnose, Then Refine: A Closed-Loop TTS System with AudioLLM-Guided Correction},
  author    = {Song, Zeyang and Liu, Tianchi and Wang, Tianrui and Xu, Chenglin and Guo, Yiwen and Li, Haizhou},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```

## License

Refiner-DB is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) for noncommercial use. See the [dataset card](https://huggingface.co/datasets/Poookeman/Refiner-DB) for terms and source acknowledgments. Code and model licenses will be specified with their respective releases.
