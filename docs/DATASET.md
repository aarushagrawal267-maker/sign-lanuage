# WLASL100 dataset

## Source

| | |
|---|---|
| Homepage | https://dxli94.github.io/WLASL/ |
| Repository | https://github.com/dxli94/WLASL |
| Paper | Li et al., *Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison*, WACV 2020 |
| Annotation file | `start_kit/WLASL_v0.3.json` (2,000 glosses, 21,083 instances) |
| WLASL100 index | `code/I3D/preprocess/nslt_100.json` (2,038 instances) |
| Licence | Computational Use of Data Agreement (C-UDA) — non-commercial |

## Vocabulary selection

WLASL100 = the **first 100 glosses** of `WLASL_v0.3.json`, in file order (class index 0–99). `app/dataset/annotations.py` selects them and cross-checks against `nslt_100.json`: identical video-id sets, identical split per video and identical class index per video. Ingestion is refused if any of these disagree. Verified against the current official files (fetched 2026-11-14): **0 mismatches**.

```
book drink computer before chair go clothes who candy cousin deaf fine help no thin walk year yes all black
cool finish hot like many mother now orange table thanksgiving what woman bed blue bowling can dog family fish graduate
hat hearing kiss language later man shirt study tall white wrong accident apple bird change color corn cow dance dark
doctor eat enjoy forget give last meet pink pizza play school secretary short time want work africa basketball birthday brown
but cheat city cook decide full how jacket letter medicine need paint paper pull purple right same son tell thursday
```

## Measured annotation statistics

| Property | Value |
|---|---|
| Instances | 2,038 (train 1,442 · val 338 · test 258) |
| Instances per gloss | min 18 · median 20 · max 40 |
| Signers | 97 ids; largest signer has 245 instances (12%) |
| Annotated fps | 25 for all instances |
| Frame ranges | 1,833 whole-video (`frame_end = -1`); 135 start after frame 1 |
| Dialect variants | `variation_id` 0: 1,690 · 1: 325 · 2: 23 |
| Sources (hosts) | YouTube 685 · signingsavvy 225 · signstock 196 · handspeak 174 · aslpro 162 · asldeafined 131 · aslsearch 114 · aslsignbank 108 · S3 (startasl) 99 · spreadthesign 91 · aslbricks 45 · weebly 8 |

Resolution, real frame rate and duration are **not** in the annotations; they are measured after download (`probe_video`) and shown per gloss in the Dataset Explorer and the inventory export.

## Acquisition

`app/dataset/acquisition.py`:

* **YouTube** — `yt-dlp`, ≤720p, one download per YouTube id (several instances can share a video with different frame ranges). Stored in `data/videos/youtube/<youtube_id>.mp4`.
* **ASLPro** — Flash `.swf` with the referer the official downloader uses, converted to MP4 with `ffmpeg`.
* **Other hosts** — HTTP download; HTML responses are rejected (dead links often return a web page with status 200).
* Polite: small random delay between downloads, 4 workers by default (`SLR_DOWNLOAD_WORKERS`).
* Every attempt is a row in `download_attempts` (method, HTTP status, error, timestamps). A failed video stays `failed`; nothing is substituted from another source or dataset.
* Re-runs skip valid videos; *retry failed* re-attempts failures.
* **Missing videos** — the WLASL authors provide them on request (form linked in the official README). Place them in `data/videos/provided/<video_id>.<ext>` and run *Scan local files*; they are tagged `acquired_via = provided`.

## Validation

A video is `valid` only if OpenCV opens it, every frame decodes, width/height/frame count are positive and the annotated clip range contains at least 4 frames. Measured fps, resolution, frame count, duration, clip length and SHA-256 are stored. Unreadable files are `invalid` with the reason.

`frame_start`/`frame_end` are interpreted as 1-based inclusive frame indices of the source file (`-1` = to the end), as in the official preprocessing code.

## Provenance kept per instance

gloss, class index, video id, signer id, official split, source, URL, annotated fps, frame range, bbox, variation id, instance id — exactly as published — plus acquisition method, file hash, measured properties, landmark extraction status and detection rates.

## Splits

See [EVALUATION.md](EVALUATION.md). Summary on the full WLASL100 metadata:

| Split | Videos train/val/test | Signers train/val/test | Signer-independent |
|---|---|---|---|
| official | 1,442 / 338 / 258 | 91 / 69 / 56 | **No** — 253/258 test and 333/338 val videos are by training signers |
| signer-holdout | 1,426 / 306 / 306 | 69 / 14 / 14 | Yes (1 class missing from val) |
| signer-cv5-fold0…4 | ≈1,223 / 408 / 408 | ≈58 / 19 / 19 | Yes |

These numbers are computed on all annotated instances; the counts actually used for training depend on which videos could be acquired, and are recorded with every experiment (`data_summary`).
