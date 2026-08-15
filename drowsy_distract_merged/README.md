# Merged drowsiness + distraction detection

Two self-contained files (same standalone pattern as `../drowsiness_live.py`) sharing one
camera/MediaPipe/driver-selection pipeline. Drowsiness and distraction each keep their own
normalization: the distraction models expect a rigid bias-correction only (their SVM/NN were
trained on one fixed camera position and still need full head rotation/pose signal to tell
Left/Right/Phone apart), while the drowsiness EAR/MAR/pose features are baseline-relative, not
pose-normalized at all. A strict Kabsch frontal reprojection exists only as an optional debug
view (`--debug-view`) -- it is never fed to either model.

## calibrate.py -- two distinct modes

**Mode A, reference-only** (runs automatically when no reference is known yet): captures the
pose the distraction models were actually trained on. Full uncropped camera view the whole
time, single recording stage, no crop, no baseline. This should really only run once, ever,
at Car-Sensors' original training camera position -- if you have that original reference file,
skip this mode entirely with `--seed-reference`:

```bash
python calibrate.py --camera-id 0 --seed-reference calibration/calibration_ref_transformationM.npz
```

**Mode B, full per-vehicle calibration** (runs whenever a reference already exists, i.e.
every real deployment): stage 1 finds your average head position (uncropped) and derives the
fixed 300x300 crop; stage 2 (now cropped) records the drowsiness baseline (EAR/MAR/pose
medians) and the distraction position bias (R, t) against the reference, together, in one
recording. This is the run you repeat any time the camera moves.

```bash
python calibrate.py --camera-id 0
```

Both modes: wait for a `C` keypress before each recording stage (nothing auto-starts), always
show landmark dots, and put all instructions/status text in a black panel below the video
(word-wrapped, never overlaid on the feed or cut off).

Everything lands in one file: `calibration/calibration.npz` (crop, baseline, `ref_3d`, and
`R`/`t` once Mode B has run).

## classify.py

```bash
python classify.py --camera-id 0
```

Cropped video on top, a black stats panel below with EAR/MAR/PERCLOS, blink/yawn/nod counts
and rates, the drowsiness prediction + confidence, and the gaze prediction + confidence --
same stats as `../drowsiness_live.py`'s live panel, extended with the gaze line.

Flags:
- `--ensemble` / `--ensemble-path` / `--ensemble-top-n` -- same as `../drowsiness_live.py`,
  drowsiness side only. The distraction side isn't tunable this way; it always averages
  `Cubic_SVM.pkl` + `Neural_Network.pkl`, matching the original Car-Sensors `classify.py`.
- `--show-landmarks` -- draw landmark dots on the video. Off by default.
- `--debug-view` -- opens a second window with three panels: RAW (uncorrected), CAMERA-BIAS
  CORRECTED (what's actually fed to the distraction models), and FRONTAL (the strict Kabsch
  reprojection, debug-only, not used by anything). Off by default.

## Distraction model files

`Models/Cubic_SVM.pkl`, `Models/Neural_Network.pkl` (from Car-Sensors) -- already present.
They unpickle with an `InconsistentVersionWarning` (pickled with sklearn 1.6.1, this venv has
1.3.0). Ran fine in testing, but worth pinning versions or re-pickling before deployment.

## Pending: dataset rebuild needed

The mouth landmarks (`mouth_top_outer` etc.) were fixed to use the true outer lip contour
instead of the inner seam (see `../METHODS.md` §7 for the full reasoning) -- this changes the
actual MAR value computed on every frame. `outputs/best_model.pkl`/`all_models.pkl` are still
trained against the old (inner-seam) MAR distribution until the dataset is rebuilt (full
MediaPipe re-extraction, not just `--rewindow`) and `model.py train` is re-run. Live drowsy MAR
readings will look different (more stable, no more 4-6x baseline-ratio spikes) but the deployed
model doesn't know that yet -- retrain before trusting drowsy predictions again.

The blink/yawn single-frame-jitter fix (`min_blink_frames`/`min_yawn_frames` floor of 2) needed
no such rebuild -- it's a live-inference-only debounce, verified to be a no-op against the
dataset's actual (much higher) native frame rate.

## Not yet verified

- The rightmost/forward-most driver-selection heuristic and the fixed crop: not tested against
  a real vehicle camera.
- The distraction models were trained on landmarks from the legacy `mp.solutions.face_mesh`
  API; this pipeline uses the newer Tasks `FaceLandmarker` API instead (needed for multi-face
  driver selection). Both report 478 points in the same topology/order, so the feature vectors
  should line up, but this hasn't been validated against Car-Sensors' actual training data.
- Whatever pose was captured by Mode A before `--seed-reference` existed is not the real
  training-time reference -- if `calibration.npz` was ever written by a plain (non-seeded)
  Mode A run, re-run `calibrate.py --seed-reference <real reference file>` before trusting
  gaze output.
