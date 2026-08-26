<div align="center">

# LoopTTS

### Diagnose, Then Refine: A Closed-Loop TTS System with AudioLLM-Guided Correction

**EMNLP 2026 Main Conference**

[Paper](paper/LoopTTS.pdf) · [Audio Demo](https://pooookeman.github.io/LoopTTS/) · Code (Coming Soon) · Dataset (Coming Soon)

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
| Paper | Available | [PDF](paper/LoopTTS.pdf) |
| Audio demo | Available | [Project page](https://pooookeman.github.io/LoopTTS/) |
| Inference code | Coming soon | This repository |
| Training framework | Coming soon | This repository |
| Refiner-DB | Coming soon | Hugging Face |
| Model checkpoints | Coming soon | Hugging Face |

## Audio Demo

The [project page](https://pooookeman.github.io/LoopTTS/) contains examples of:

- joint emotion and word-level stress/pause control;
- speed and pitch control from the same initial utterance;
- AudioLLM-generated refine instructions;
- before/after refinement comparisons from the full LoopTTS pipeline.

## Code and Models

The inference code, training framework, configuration files, and Refiner checkpoints are being organized for release. They will be added to this repository without changing the project URL.

## Refiner-DB

Refiner-DB and its data preparation documentation will be released through Hugging Face. The dataset link and redistribution details will be added here when the release is ready.

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

The code, model, and dataset licenses will be specified with their respective releases.
