# Limitations

## Scope
* **Isolated signs only**, from a fixed vocabulary of 100 WLASL glosses. Anything else is either rejected as uncertain or misclassified as the nearest known sign.
* The output is a **gloss sequence**, not English. No sentence translation, ASL grammar, classifiers, role shift, spatial reference, fingerspelling or numbers.
* Not a communication aid for real conversations and **not a replacement for interpreters**.

## Data
* About 20 videos per gloss (18–40), from a limited set of signers; WLASL100 does not represent all ASL users, ages, regional variants or signing styles.
* Many URLs have expired; the trainable set depends on what could be acquired. Results are reported with the dataset version and per-partition counts.
* Some glosses have several valid variants (`variation_id`); they share one label, which makes those classes harder.
* The official split is not signer-independent; this project reports signer-grouped results separately.

## Model
* Landmark-based models cannot see information MediaPipe misses: fast motion blur, hand–hand and hand–face occlusion, hands outside the frame, unusual lighting.
* Small held-out partitions make per-class metrics noisy.
* Rejection of unknown signs is a confidence/margin/agreement heuristic, not a trained out-of-vocabulary detector.

## Real time
* Live webcam conditions differ from the curated dataset clips (framing, distance, lighting, camera angle, signing speed), so live accuracy is expected to be **lower than test accuracy**. No live accuracy figure is claimed.
* Sign-boundary mode requires lowering the hands between signs; co-articulated continuous signing is not segmented.
* Throughput depends on hardware. In a headless 4-core test container the browser reached ~8 landmark frames/s on CPU; a laptop with a GPU is typically faster. Latency figures shown in the UI are measured live on your machine.
* Browser and Python builds of MediaPipe agreed closely on visible landmarks in a still-image test; agreement on real signing video still needs to be measured on your machine (CHECKLIST.md).

## Deployment
* Runs locally on one machine (Windows targeted). Not hardened for network exposure: the backend binds to 127.0.0.1 and has no authentication.
* CPU training is practical for the default model size; large configurations benefit from CUDA.
