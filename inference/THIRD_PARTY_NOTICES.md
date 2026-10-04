# Third-party notices

[`refiner_infer.py`](refiner_infer.py) adapts the input packing, token generation, and CosyVoice decoding logic from [EmoVoice](https://github.com/yanghaha0908/EmoVoice), commit `5285cb891611cf1ee2d9bd07b931cd3cf967cd64`, with the Refiner-specific input format from this project's evaluated implementation. The EmoVoice authors state in their [README](https://github.com/yanghaha0908/EmoVoice#license) that their source code is released under the MIT License. Copyright in the original portions remains with their respective owners. The LoopTTS changes are offered under CC BY-NC 4.0.

The runtime imports [CosyVoice](https://github.com/QwenAudio/CosyVoice) directly from its independently installed source. No CosyVoice code or weights are included in this repository.
