"""ISL video / image sources that need our own MediaPipe extraction (WSLP, CISLR, ISL-Fingerspelling, Mendeley
ISL-CSLTR / emergency / agriculture / static alphabet, HF alphabet image sets).

``catalog`` (what, licence, profile, mode), ``listings`` (metadata -> work items, pure), ``fetch`` (one item -> temp
file, hashed, safe), ``extract`` (worker side), ``engine`` (resumable runner), ``adapters`` (Clip readers).
CLI: ``scripts/ingest_isl_video.py``; queue: ``scripts/isl_queue.sh``; docs: docs/DATASETS_ISL.md 3.3.
"""
