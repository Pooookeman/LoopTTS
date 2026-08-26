# LoopTTS: Diagnose, Then Refine

LoopTTS is a closed-loop TTS framework for recovering outputs with local prosodic defects. An AudioLLM Judge diagnoses salient issues and produces structured instructions; a purpose-trained Refiner performs guided expressive re-synthesis conditioned on the initial utterance, target text, and instruction.

## Project page

The audio demo is available at <https://pooookeman.github.io/LoopTTS/>.

## Release status

| Resource | Status |
|---|---|
| Audio demo | Available |
| Training and inference code | Coming soon |
| Refiner-DB | Coming soon |

The repository will be updated with the Refiner implementation, inference pipeline, data preparation utilities, and Refiner-DB release information.

## Citation

```bibtex
@inproceedings{song2026loopt,
  title     = {Diagnose, Then Refine: A Closed-Loop TTS System with AudioLLM-Guided Correction},
  author    = {Song, Zeyang and Liu, Tianchi and Wang, Tianrui and Xu, Chenglin and Guo, Yiwen and Li, Haizhou},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics},
  year      = {2026}
}
```

## Authors

Zeyang Song, Tianchi Liu, Tianrui Wang, Chenglin Xu, Yiwen Guo, and Haizhou Li.

