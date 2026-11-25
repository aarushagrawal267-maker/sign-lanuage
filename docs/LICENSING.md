# Licensing

## Project code

The source code is released under the **MIT licence** (see [`LICENSE`](../LICENSE)). That licence does
not extend to the WLASL data or to the model weights in `models/`, which are derived from WLASL and
are therefore for non-commercial use only (C-UDA, below).

## WLASL dataset

* Licensed under the **Computational Use of Data Agreement (C-UDA)** (`C-UDA-1.0.pdf` in the official repository). Read and accept it before downloading.
* The official README: *"All the WLASL data is intended for academic and computational use only. No commercial usage is allowed."*
* The videos were collected by the WLASL authors from third-party educational websites and YouTube; their copyright remains with the original owners.

How this project complies:

| Requirement | Implementation |
|---|---|
| Non-commercial | The application is labelled an educational and research implementation in the UI, API description and README. Do not deploy it commercially or use models trained on WLASL commercially. |
| No redistribution | No WLASL video is in this repository. `data/` is git-ignored. Each user downloads from the original URLs. Video Analysis uploads are deleted after processing. |
| Reference clips | The Vocabulary page shows a clip only from the user's own local copy, served by the local backend. |
| Attribution & citation | Footer on every page, Research & Docs page, README. |
| Derived data | Extracted landmarks and checkpoints are derived from WLASL; keep them local and under the same terms. |

Citation:

```
@inproceedings{li2020word,
  title={Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison},
  author={Li, Dongxu and Rodriguez, Cristian and Yu, Xin and Li, Hongdong},
  booktitle={The IEEE Winter Conference on Applications of Computer Vision},
  pages={1459--1469},
  year={2020}
}
```

## Third-party software

| Component | Licence |
|---|---|
| MediaPipe / `@mediapipe/tasks-vision`, Holistic model | Apache-2.0 |
| PyTorch | BSD-3-Clause |
| FastAPI, Pydantic, SQLAlchemy, Uvicorn | MIT |
| NumPy, pandas, scikit-learn | BSD-3-Clause |
| OpenCV | Apache-2.0 |
| yt-dlp | Unlicense |
| React, React Router, Vite, Tailwind CSS | MIT |
| Lucide icons | ISC |
| IBM Plex fonts | SIL Open Font License 1.1 |

Downloading YouTube videos may be subject to YouTube's Terms of Service in your jurisdiction; the official WLASL tooling uses the same approach for research purposes.
