# data/

Local only (git-ignored): official annotation files, WLASL100 videos, extracted landmarks and the SQLite database (app.db).

* `annotations/` - WLASL_v0.3.json and nslt_100.json, fetched from the official repository.
* `videos/youtube`, `videos/direct` - downloaded sources. `videos/provided` - videos supplied by the WLASL authors on request (name them `<video_id>.mp4`).
* `landmarks/` - one `<video_id>.npz` raw MediaPipe sequence per instance.

WLASL videos are licensed under the C-UDA (non-commercial, computational use) and must not be redistributed.
