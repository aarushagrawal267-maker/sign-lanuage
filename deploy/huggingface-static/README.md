---
title: Sign Intelligence Studio
emoji: 🤟
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
pinned: false
license: other
short_description: Real-time isolated ASL sign recognition (WLASL100)
---

# Sign Intelligence Studio — online demo

Educational and research implementation of **isolated, word-level** ASL sign recognition on the **WLASL100** vocabulary. Everything runs **in your browser**: MediaPipe Holistic extracts hand/body landmarks from your webcam and a GRU sequence model (ONNX Runtime Web) classifies the sign. No video or landmark data leaves your device.

* Output is an ASL **gloss label** for one isolated sign — not a translation and not a replacement for interpreters.
* Model: bidirectional GRU trained on the WLASL100 videos that were still downloadable (524 clips, about 5 per sign).
  Signer-independent results: validation top-1 62.0%; held-out test (7 unseen signers) top-1 44.3%, top-5 72.2%.
  Expect lower accuracy on webcam input.
* Read-only: data download, training and video-file analysis require running the full project locally.

Source code: https://github.com/aarushagrawal267-maker/sign-lanuage

**Data licence:** WLASL is © its authors under the Computational Use of Data Agreement (C-UDA) — academic and computational use only, **no commercial use**. No WLASL videos or landmarks are included here. Please cite Li et al., *Word-level Deep Sign Language Recognition from Video*, WACV 2020.
