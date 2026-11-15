---
title: Sign Intelligence Studio
emoji: 🤟
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: other
short_description: Real-time isolated ASL sign recognition (WLASL100)
---

# Sign Intelligence Studio — public demo

Educational and research implementation of **isolated, word-level** ASL sign recognition on the **WLASL100** vocabulary: MediaPipe Holistic landmarks (computed in your browser) and a recurrent sequence model (GRU).

* Output is an ASL **gloss label** for one isolated sign — not a translation and not a replacement for interpreters.
* Your webcam video is processed in your browser; only landmark coordinates are sent to the model.
* This demo is **read-only**: downloading data and training are disabled. Run the full project locally for those.
* No WLASL videos or landmark files are included. The model was trained on a subset of WLASL100 (videos available at download time) and evaluated on **signers not seen in training**; see the Evaluation page for its real numbers and limitations.

**Licence of the data:** WLASL is © its authors under the Computational Use of Data Agreement (C-UDA) — academic and computational use only, **no commercial use**. Please cite Li et al., *Word-level Deep Sign Language Recognition from Video*, WACV 2020.
