import { PageHeader } from "../components/ui";

const SECTIONS = [
  ["scope", "Scope"],
  ["background", "ASL recognition background"],
  ["wlasl", "WLASL dataset"],
  ["mediapipe", "MediaPipe architecture"],
  ["features", "Landmark features"],
  ["sequence", "Temporal sequence learning"],
  ["models", "Model architectures"],
  ["evaluation", "Evaluation methodology"],
  ["live", "Live recognition & smoothing"],
  ["privacy", "Privacy"],
  ["licensing", "Dataset licensing & citation"],
  ["limitations", "Limitations"],
  ["reproducibility", "Reproducibility"],
] as const;

export default function ResearchPage() {
  return (
    <div className="fade-in">
      <PageHeader title="Research & documentation"
        description="How the system works, what it can and cannot do, and how to reproduce the results." />
      <div className="grid gap-8 lg:grid-cols-[220px_minmax(0,1fr)]">
        <nav aria-label="Sections" className="lg:sticky lg:top-20 lg:self-start">
          <ul className="flex flex-wrap gap-1 lg:flex-col">
            {SECTIONS.map(([id, label]) => (
              <li key={id}><a href={`#${id}`} className="block rounded-md px-2.5 py-1.5 text-sm text-fg-2 hover:bg-ink-800 hover:text-fg">{label}</a></li>
            ))}
          </ul>
        </nav>

        <article className="prose-doc max-w-3xl rounded-xl border border-ink-700 bg-ink-850 px-6 py-4 md:px-10 md:py-6">
          <h2 id="scope">Scope</h2>
          <p>This is an independent <strong>educational and research implementation</strong> of <strong>isolated, word-level</strong> American Sign Language (ASL)
            recognition. It classifies one sign at a time into one of the 100 glosses of WLASL100 and prints that gloss as text.</p>
          <ul>
            <li><strong>Individual sign recognition</strong> — yes: one isolated sign from a fixed 100-word vocabulary.</li>
            <li><strong>ASL gloss classification</strong> — the output is a gloss label (an English word used to name a sign), not a translation.</li>
            <li><strong>Text output</strong> — a sequence of recognised glosses, e.g. <code>MOTHER WORK</code>. It is not English grammar.</li>
            <li><strong>Continuous sign language understanding</strong> — no. Signs must be separated (hands lowered between signs).</li>
            <li><strong>Full sign language translation</strong> — no. ASL has its own grammar (spatial reference, non-manual markers, classifiers) that this system does not model.</li>
          </ul>
          <p>It does not replace qualified interpreters and must not be used where accurate communication matters.</p>

          <h2 id="background">ASL recognition background</h2>
          <p>Signs combine handshape, orientation, location, movement and non-manual signals (facial expression, mouth shape, head movement).
            Isolated sign recognition is usually tackled either from raw video (3D CNNs such as I3D) or from <strong>pose / landmark sequences</strong>,
            which are compact, fast and less sensitive to background and appearance. Pose-based models on WLASL100 have been reported from about 46% top-1
            (Pose-GRU, the WLASL paper) to about 78% top-1 (SPOTER with MediaPipe landmarks, Boháček et al.), on the official — not signer-independent — split.</p>

          <h2 id="wlasl">WLASL dataset</h2>
          <p>WLASL (Li et al., WACV 2020) contains about 2,000 glosses and 21,083 annotated video instances collected from educational sign websites and
            YouTube, performed by more than 100 signers. <strong>WLASL100</strong> is defined by the authors as the first 100 glosses of
            <code>WLASL_v0.3.json</code>; this application selects exactly those and verifies the selection against the authors' <code>nslt_100.json</code>.</p>
          <ul>
            <li>2,038 annotated instances (1,442 train / 338 val / 258 test in the official split), 18–40 per gloss.</li>
            <li>97 signer ids; all instances annotated at 25 fps; per-instance frame ranges, bounding boxes and dialect-variation ids.</li>
            <li>Videos are referenced by URL only; many links have expired. Missing videos are recorded as unavailable and never substituted.
              The authors provide missing videos on request via the form linked in the official repository.</li>
          </ul>
          <p>Measured properties (resolution, true fps, duration) are recorded per video after download and shown in the Dataset explorer.</p>

          <h2 id="mediapipe">MediaPipe architecture</h2>
          <p>Google MediaPipe <strong>Holistic Landmarker</strong> combines pose (33 landmarks), two hands (21 landmarks each) and a 478-point face mesh in a single
            task. The same <code>holistic_landmarker.task</code> model runs in two places:</p>
          <ul>
            <li><strong>Offline</strong> (dataset videos, Video analysis): Python <code>mediapipe</code> in VIDEO mode on the backend.</li>
            <li><strong>Live</strong> (webcam): <code>@mediapipe/tasks-vision</code> WebAssembly in the browser, GPU delegate when available, CPU otherwise.</li>
          </ul>
          <p>Left/right hand labels refer to the <strong>signer's own hands</strong>. Frames are always processed unmirrored; the webcam preview is only
            mirrored on screen.</p>

          <h2 id="features">Landmark features</h2>
          <p>Raw landmarks are stored with missing detections as <code>NaN</code>. One Python module turns them into features for training, offline analysis
            and live recognition alike, so preprocessing is identical everywhere:</p>
          <ol>
            <li><strong>Aspect correction</strong> — x (and hand z) multiplied by width/height so distances are isotropic across 4:3 and 16:9 cameras.</li>
            <li><strong>Gap handling</strong> — interior gaps of up to 5 frames are linearly interpolated; longer gaps stay missing and are flagged.</li>
            <li><strong>Body frame</strong> — origin at the median shoulder midpoint, unit = median shoulder width. Sequences without any body detection are rejected.</li>
            <li><strong>Hand shape</strong> — each hand relative to its wrist, scaled by wrist→middle-knuckle length (20 points × 3).</li>
            <li><strong>Hand position</strong> — wrist location in the body frame, preserving where the sign is made.</li>
            <li><strong>Upper body</strong> — nose, shoulders, elbows, wrists (hips are often outside a laptop camera frame).</li>
            <li><strong>Face subset</strong> — 20 mesh points (mouth, eyebrows, eye corners, nose, chin, cheeks), optional and evaluated by ablation.</li>
            <li><strong>Presence flags</strong> for each hand and the face, plus <strong>velocity</strong> (first differences) of all positional features.</li>
          </ol>
          <p>The default vector has 359 values per time step (static 178 + velocity 178 + 3 flags).</p>

          <h2 id="sequence">Temporal sequence learning</h2>
          <p>Signs are dynamic, so classification always uses a sequence, never a single frame. Each clip (the annotated frame range) is resampled by timestamp to a
            fixed number of steps (default 48; the Dataset explorer recommends a value from the measured clip lengths). Resampling never blends a detected frame
            with a missing one. Training augmentation: mirroring (left-handed signing), small rotation/scale/shift, temporal crop and time-warp, landmark jitter and
            random hand dropout.</p>

          <h2 id="models">Model architectures</h2>
          <ul>
            <li><strong>GRU</strong> (primary) — LayerNorm + linear projection → 2-layer bidirectional GRU → masked attention pooling → classifier.</li>
            <li><strong>LSTM</strong> — the same with LSTM cells, for direct comparison.</li>
            <li><strong>Temporal CNN</strong> — dilated residual 1D convolutions → attention pooling; a fast non-recurrent baseline.</li>
          </ul>
          <p>Training: AdamW, warm-up + cosine schedule, label smoothing, optional class-weighted loss, gradient clipping, early stopping on validation macro F1,
            fixed seeds, configuration stored with every checkpoint. Bidirectional layers are valid live because a complete sign window is classified at once.</p>

          <h2 id="evaluation">Evaluation methodology</h2>
          <p>A random or official split lets the same person appear in training and testing; the model can then exploit signer-specific appearance and motion.
            In the official WLASL100 split, <strong>253 of 258 test videos and 333 of 338 validation videos</strong> are by signers who also appear in training.</p>
          <ul>
            <li><strong>signer-holdout</strong> — every signer in exactly one partition (~70/15/15), chosen to keep all classes represented in validation and test.</li>
            <li><strong>signer-cv5-fold0…4</strong> — 5-fold grouped cross-validation over signers.</li>
            <li>Signers that share a source video file are grouped so identical footage never crosses partitions.</li>
            <li>Automated leakage checks: signer overlap, content-hash overlap, source-file overlap, duplicate assignment, class coverage.</li>
            <li>Model selection uses validation only. The held-out test set is evaluated by an explicit, separate action and reported separately.</li>
            <li>Metrics: top-1, top-5, macro F1 (over classes present), per-class precision/recall/F1, confusion matrix, per-signer accuracy, CPU latency.</li>
          </ul>
          <p>With ~20 videos per gloss, held-out test sets are small and per-class numbers are noisy; read them with that in mind.</p>

          <h2 id="live">Live recognition & smoothing</h2>
          <p>The browser sends only landmark coordinates (about 3 KB per frame) over a WebSocket to the local backend, which keeps a rolling buffer and runs the
            same feature pipeline and model.</p>
          <ul>
            <li><strong>Sign-boundary mode</strong> (default): a sign starts when hands are detected for 3 consecutive frames and ends when they leave view for
              ~400 ms. The whole segment is classified with three temporal crops; a word is emitted only if the mean confidence passes the threshold, the margin over
              the runner-up is at least 10%, and ≥ 2 of 3 crops agree. Provisional guesses are shown but never emitted.</li>
            <li><strong>Sliding-window mode</strong>: classifies the last 2 s every 200 ms with exponential smoothing; emits after consecutive agreeing windows,
              suppresses repeats until the hands drop or the prediction changes, and applies a cooldown.</li>
            <li>The default threshold is derived from the validation accuracy–coverage curve of the active model. A softmax threshold is a weak detector of
              out-of-vocabulary signs; uncertain results are shown as such rather than forced into a word.</li>
          </ul>

          <h2 id="privacy">Privacy</h2>
          <ul>
            <li>The camera starts only when you press <em>Start camera</em> and the browser permission is granted; a red indicator shows while it is on.</li>
            <li>Video frames are processed in the browser. No webcam image is sent anywhere, saved, or used to identify you.</li>
            <li>Only recognised gloss labels and scores are stored (Recognition history), and can be cleared.</li>
            <li>Uploaded files for Video analysis go only to the local backend and are deleted right after processing.</li>
          </ul>

          <h2 id="licensing">Dataset licensing & citation</h2>
          <p>WLASL is released under the <strong>Computational Use of Data Agreement (C-UDA)</strong>. The official repository states that the data is for
            academic and computational use only and that <strong>no commercial use is allowed</strong>. This project therefore:</p>
          <ul>
            <li>is labelled as an educational and research implementation and must not be deployed commercially;</li>
            <li>never redistributes WLASL videos — they are downloaded by each user from the original URLs and stay in the git-ignored <code>data/</code> folder;</li>
            <li>shows reference clips only from your own local copy;</li>
            <li>credits the authors and asks users to cite the paper.</li>
          </ul>
          <pre>{`@inproceedings{li2020word,
  title     = {Word-level Deep Sign Language Recognition from Video:
               A New Large-scale Dataset and Methods Comparison},
  author    = {Li, Dongxu and Rodriguez, Cristian and Yu, Xin and Li, Hongdong},
  booktitle = {The IEEE Winter Conference on Applications of Computer Vision},
  pages     = {1459--1469},
  year      = {2020}
}`}</pre>
          <p>Official sources: <a href="https://dxli94.github.io/WLASL/" target="_blank" rel="noreferrer">WLASL homepage</a> ·{" "}
            <a href="https://github.com/dxli94/WLASL" target="_blank" rel="noreferrer">GitHub repository</a> ·{" "}
            <a href="https://openaccess.thecvf.com/content_WACV_2020/papers/Li_Word-level_Deep_Sign_Language_Recognition_from_Video_A_New_Large-scale_WACV_2020_paper.pdf" target="_blank" rel="noreferrer">paper</a>.
            MediaPipe is © Google, Apache License 2.0.</p>

          <h2 id="limitations">Limitations</h2>
          <ul>
            <li>100 isolated signs only; no fingerspelling, sentences, grammar, classifiers or role shift.</li>
            <li>WLASL100 does not represent all ASL users, regional variants or signing styles; about 20 examples per sign is very little data.</li>
            <li>Dataset clips are mostly front-facing, well-lit and trimmed. Webcam conditions (framing, lighting, speed, camera angle) differ, so live accuracy
              will be lower than test-set accuracy.</li>
            <li>Landmark extraction can fail with fast motion, occlusion, hands leaving the frame or poor lighting; signs touching the face are hardest.</li>
            <li>The unknown-sign rejection is a confidence heuristic, not a trained out-of-vocabulary detector.</li>
            <li>Video links expire; results depend on which videos could be acquired and are reported with the dataset version.</li>
          </ul>

          <h2 id="reproducibility">Reproducibility</h2>
          <ol>
            <li>Run <code>setup.ps1</code> (Python venv, PyTorch CPU/CUDA, Node packages, MediaPipe model), then <code>start.ps1</code>.</li>
            <li>Dataset explorer → <em>Load annotations</em>. The annotation SHA-256 is recorded as the dataset version.</li>
            <li><em>Download &amp; validate</em> the videos, then <em>Extract landmarks</em>.</li>
            <li>Model laboratory → train on <code>signer-holdout</code>. Every run stores its full configuration, seed, feature settings, split and checkpoint.</li>
            <li>Compare runs on validation; once settled, run the held-out test evaluation once on the Evaluation page.</li>
            <li>For the most robust estimate, train on each <code>signer-cv5-fold</code> split and report the mean and spread.</li>
          </ol>
          <p>Command-line equivalents are documented in the repository README (<code>python -m app.cli …</code>).</p>
        </article>
      </div>
    </div>
  );
}
