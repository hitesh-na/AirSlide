# AirSlide

**Control PowerPoint with your hand, using a webcam and a custom-trained neural network.**

AirSlide recognizes four right-hand poses and turns them into presentation commands. It runs locally on your computer, shows a live camera preview, and uses an open-palm reset to prevent a held gesture from repeatedly changing slides.

The active version is in **`palm_v3/`**. Older root-level scripts and `simple_v2/` are development history; use the commands in this README for the current app.

[Setup](#1-set-up-python) · [Gestures](#gesture-controls) · [Real control](#5-control-your-presentation) · [Troubleshooting](#reports-and-troubleshooting)

## Before you start

You need:

- A **64-bit Windows** computer. Presentation control uses Windows APIs.
- **Python 3.11, 64-bit**, the version family used during development.
- A working webcam, with camera access enabled for desktop apps.
- The **desktop Microsoft PowerPoint app**, running on the same computer as AirSlide.
- The compatible trained AirSlide model and its matching dataset files, listed below.

A GPU is not required. The supplied control backend targets PowerPoint's slideshow window; PowerPoint in a browser, Google Slides, macOS, and Linux presentation control are not supported by this implementation.

> **Model availability:** Source code alone is not enough to run this version. The current loader checks the dataset, metadata, lineage and split manifest before loading the trained checkpoint. If your download does not contain the required model/data files, obtain the matching runtime files from the project maintainer before continuing. This README does not include or promise a public pretrained-model download.

## Required project layout

Download or clone the complete project and keep this directory structure. The root folder may be named `AirSlide` or retain the older `GesturePilot` name.

| Path from the project root | Purpose |
|---|---|
| `requirements.txt` | Project Python dependencies |
| `models/hand_landmarker.task` | Google's pretrained hand-landmark model |
| `palm_v3/control_slides.py` | AirSlide controller entry point |
| `palm_v3/controller_preview.py` | Status panel and Arm/Pause buttons |
| `palm_v3/config.json` | Gesture, camera, model and timing settings |
| `palm_v3/models/gesture_mlp.pt` | Compatible trained Palm V3 gesture classifier |
| `palm_v3/data/samples.csv` | Matching normalized landmark dataset |
| `palm_v3/data/samples.metadata.json` | Dataset/gesture/hand metadata |
| `palm_v3/data/lineage.json` | Dataset source and row lineage |
| `palm_v3/data/split_manifest.json` | Saved session split and fingerprints |

Keep the checkpoint and its data files together as a matching set. The older five-class and Simple V2 checkpoints are incompatible with the current Palm V3 loader.

## 1. Set up Python

Open **PowerShell** in the project root. Replace the example path with your own project location:

```powershell
Set-Location "C:\path\to\AirSlide"
python --version
```

Confirm that this is Python **3.11**. If AirSlide already runs in your existing project environment, keep using that environment and skip the installation commands below.

For a fresh setup:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

These commands invoke the environment's Python directly, so activating PowerShell scripts is unnecessary. Use the same `.venv` for the rest of this guide.

The project pins MediaPipe, OpenCV, NumPy and other dependencies in `requirements.txt`; CPU PyTorch is installed separately. Install only one OpenCV package in the environment. This project selects **`opencv-contrib-python`**, which provides the camera preview. A headless OpenCV package does not provide that GUI.

## 2. Add the hand-landmark model

AirSlide uses two different models:

| Model | Role | Location |
|---|---|---|
| MediaPipe Hand Landmarker | Finds 21 hand landmarks | `models/hand_landmarker.task` |
| Custom PyTorch MLP | Classifies the four AirSlide poses | `palm_v3/models/gesture_mlp.pt` |

If `hand_landmarker.task` is missing, run this from the **project root** to download Google's versioned asset:

```powershell
New-Item -ItemType Directory -Force -Path "models" | Out-Null
if (-not (Test-Path ".\models\hand_landmarker.task")) {
    Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile ".\models\hand_landmarker.task" -ErrorAction Stop
}
```

This supplies the landmark extractor. It does **not** download or train the custom AirSlide classifier. Place the compatible `gesture_mlp.pt` and matching data files at the paths listed above.

## 3. Check the installation

Move into the active application folder:

```powershell
Set-Location ".\palm_v3"
..\.venv\Scripts\python.exe control_slides.py --check
```

A successful check prints `CONTROLLER CHECK PASSED`. It validates the model/data checks, controller settings and Windows input layout. It does not open the webcam, attach to PowerPoint, or send keys.

**All commands below assume PowerShell is in `palm_v3/`.** If you open a new terminal, navigate back there before running them.

## 4. Try a dry run

Start with a short presentation in desktop PowerPoint and launch its slideshow manually, for example with **F5**. Then start AirSlide:

```powershell
..\.venv\Scripts\python.exe control_slides.py --dry-run --camera 0
```

The preview stays open until you exit. To end automatically after 60 seconds, add `--duration 60`; that duration includes the time spent waiting and arming.

1. Click **ARM [A]** in the AirSlide preview. The terminal immediately prints **ARMING**.
2. During the **five-second countdown**, select the actual **PowerPoint slideshow**. Keep it focused while gesturing.
3. Show your **right-hand open palm** until the panel says **READY**.
4. Hold your **index finger up**, with your thumb and other fingers folded, for about **one second**.
5. Look for **`WOULD_SEND: next -> right`** in the terminal. The preview shows **`WOULD SEND: next`**.
6. Return to an **open palm** before trying another action.

**Slides stay still in dry run.** The app records proposed actions without sending keyboard input. The preview's **Actions** number is the cumulative proposal count in this mode, not a list of pending commands.

If the controller stays disarmed, click the **ARM button in the preview**, or click the preview and press A there. Typing A in PowerShell does not arm the camera window.

## Gesture controls

Use **one visible right hand**, keep your fingers inside the frame, and use steady poses.

| Gesture | How to show it | PowerPoint action |
|---|---|---|
| **Open palm** | Open your hand with all fingers visible | Prepare the next action; no key is sent |
| **Index up** | Extend only your index finger; fold your thumb and other fingers | Right arrow: next animation or slide |
| **V / peace sign** | Extend index and middle fingers; fold the remaining fingers | Left arrow: previous animation or slide |
| **Closed fist** | Close all fingers into a fist | B: toggle the black screen |

**The normal sequence is: open palm → READY → hold one action → open palm again.**

After a successful action, **NEED NEUTRAL is expected**. It asks for an open palm before another command. Holding a fist or index-up pose continuously should produce only one action. To toggle the black screen off again, show an open palm, then another fist.

## 5. Control your presentation

Close the dry-run preview with **Q** while that window has focus. Keep your PowerPoint slideshow open, then run:

```powershell
..\.venv\Scripts\python.exe control_slides.py --control --camera 0
```

Repeat the same sequence: **ARM → select slideshow during countdown → open palm → action gesture**.

This time the terminal prints **`SENT: next -> right`**, **`SENT: previous -> left`**, or **`SENT: blackout -> b`**. Observe the presentation to confirm the intended change. Right/Left can advance or reverse an animation before changing slides.

`SENT` means Windows accepted the keyboard pair; it is not independent confirmation of a slide change. **`--control` still starts disarmed** and needs the local Arm action.

## Pause, exit and resume

| Control | Effect |
|---|---|
| **PAUSE [D]** button | Disarm the controller |
| **D**, with preview focused | Disarm the controller |
| **Q** or **Esc**, with preview focused | Close AirSlide and release the camera |
| **Ctrl+C**, in its PowerShell terminal | Stop AirSlide |
| Move the mouse into a corner of the primary screen | Pause/disarm control |
| Switch away from the selected PowerPoint slideshow | Pause/disarm control |

Preview keyboard shortcuts are **not global hotkeys**. Clicking the preview while controlling slides changes focus, so delivery pauses. To resume, click **ARM**, select the slideshow during the countdown, and show an open palm again. Move the mouse away from screen corners before arming.

For multiple monitors, select the actual slideshow window. The editor and Presenter View are not accepted targets.

## Understand the status panel

| Displayed text | Meaning |
|---|---|
| **DISARMED** | Waiting for the Arm button or A in the preview |
| **ARMING** | Countdown is running; select the PowerPoint slideshow |
| **NEED NEUTRAL** | Show an open palm to prepare one action |
| **READY** | An open palm was accepted; hold an action gesture |
| **PAUSED** | Read the last message, resolve its cause, then arm again |
| **WOULD SEND** | A gesture produced a proposal in dry run |
| **SENT** | Windows accepted a keyboard command in control mode |
| **NO USABLE HAND** | No hand suitable for classification is currently tracked |

The detected pose and the controller state describe different things. **INDEX UP / NEXT** together with **NEED NEUTRAL** can be normal after a next command: the pose is still visible, but another action needs a new open palm.

## How AirSlide works

OpenCV reads the webcam. MediaPipe estimates the hand's 21 world landmarks. AirSlide centers them at the wrist, scales them by palm length, and rotates them into a coordinate frame defined by the palm. The resulting **63 values** feed a custom **63 → 64 → 32 → 4 MLP**, with ReLU hidden layers and **6,308 trainable parameters**.

The trained network predicts `neutral`, `next`, `previous` or `blackout`. MediaPipe supplies landmark coordinates; it does not supply AirSlide's learned gesture labels. Palm alignment reduces sensitivity to rigid rotation of accurate landmarks, but it cannot recover hidden fingers or incorrect tracking.

The controller requires a steady qualifying pose and a fresh open-palm preparation for every action. It verifies PowerPoint focus and uses Windows `SendInput` with paired key-down/key-up events. No-hand frames reset readiness. Reports contain numerical events; the normal controller does not record webcam video.

Default settings in `palm_v3/config.json` are:

| Setting | Value |
|---|---:|
| Model score threshold | 0.85 |
| Action hold | 0.5 seconds |
| Open-palm hold | 0.3 seconds |
| Minimum action interval | 0.8 seconds |
| Arming countdown | 5 seconds |

A prediction gap longer than **0.25 seconds** also requires a new open palm. Model scores are not calibrated probabilities of correctness. A high score alone does not establish that an arbitrary pose was recognized correctly.

## Reports and troubleshooting

Controller sessions save a timestamped JSON file under **`palm_v3/reports/`**. It records mode, arming events, proposed/sent/blocked actions, hand/no-hand counts, model fingerprints and the exit/error reason. Use the full terminal message and this report when investigating a problem.

| Problem | What to check |
|---|---|
| `No module named ...` | Use `..\.venv\Scripts\python.exe` from `palm_v3/`; install dependencies into that same environment. |
| Missing checkpoint, CSV, metadata, lineage or manifest | Restore the complete matching runtime files listed above. Installing packages does not create these files. |
| Feature, class or dataset hash mismatch | Restore the matching Palm V3 checkpoint/data/configuration set. Keep the checks enabled. |
| Camera does not open | Close other apps using the webcam, check desktop-app camera permissions and reconnect the device. Try `--camera 1` if camera 0 is not the intended device. |
| `Camera stopped returning frames` | The camera read failed repeatedly. The app stops, releases resources and writes an error report. Reconnect/check the webcam and restart with a dry run. |
| No actions and `arming_requests` is 0 | Click the preview's ARM button. No arming request reached the previous session. |
| `PAUSED` during arming | Read the reason. Select the real desktop PowerPoint slideshow, keep it focused and keep the pointer away from primary-screen corners. |
| Pose is detected but no command follows | First show an open palm until READY, then hold the intended action steadily for about a second. Keep your whole hand visible. |
| READY keeps returning to NEED NEUTRAL without a new proposal | Check for hand loss or processing gaps; keep the hand in frame and watch the last message/action count. |
| The display says `WOULD SEND` but slides stay still | This is dry run. Close it and launch with `--control` after the dry run behaves correctly. |
| The display says `SENT` but the presentation does not visibly move | Check for animation steps, the end of the deck or a black-screen toggle. Confirm manual arrow/B shortcuts work and that the intended slideshow has focus. |
| A modifier/key warning appears | Release held Shift/Ctrl/Alt/Windows or shortcut keys, then arm again. |
| Preview text looks outlined or doubled | Use the current `controller_preview.py` panel and matching `control_slides.py`; close old previews before restarting. |

## Optional checks

Run the controller's automated tests from `palm_v3/`:

```powershell
..\.venv\Scripts\python.exe -m unittest discover -s tests -p test_controller.py -v
```

The **28 controller tests** cover arming, holds, neutral reset, dry-run isolation, focus guards, mouse buttons, Windows output pairs and camera cleanup using injected backends. They do not send real presentation keys or prove behavior on every webcam/PowerPoint setup.

For a prediction-only pose diagnostic, hold one pose throughout the measured interval:

```powershell
..\.venv\Scripts\python.exe app.py --dry-run --camera 0 --warmup 5 --duration 15 --device-label desktop --expected-label previous
```

Here `previous` means the V sign. Valid expected labels are `neutral`, `next`, `previous`, `blackout`. The diagnostic measures after warm-up and writes scores/features to reports; it does not train the model. Its frame agreement is not a general accuracy estimate.

## Training and measured results

Starting AirSlide does **not** require retraining when you already have the compatible checkpoint and matching data. Training scripts remain available for reproducing the original validated dataset workflow. They are not a one-command collector/trainer for arbitrary new gestures or session layouts.

The original dataset contains **1,200 rows**, split by recording session into **720 training**, **240 validation** and **240 test** rows. Palm V3 selected **epoch 100** using validation loss and achieved **100% validation accuracy**, macro F1 **1.0000** on those 240 validation rows. Because validation selected the checkpoint, that result is not a new independent test of general performance. The existing test session had already been inspected during earlier development; it should not be presented as a fresh independent Palm V3 test.

In one reported Windows control session, **32 keyboard commands were accepted, with 0 blocked commands**, and the author confirmed PowerPoint responded. That is a working demonstration, not a measured 32/32 intended-command accuracy result. The session ended with a camera read failure; that limitation remains recorded.

The model was developed using one person's right hand. Recognition on other people, left hands, different cameras, occluded fingers or different lighting is not established by these results. Check your own setup in dry run before relying on it for a presentation.

## Technology and references

| Component | Purpose |
|---|---|
| Python | Application and training code |
| OpenCV | Camera input, drawing and preview |
| MediaPipe Tasks Hand Landmarker | Pretrained landmark extraction |
| PyTorch on CPU | Custom gesture MLP |
| NumPy | Landmark geometry and features |
| scikit-learn / Matplotlib | Offline evaluation and plots |
| Windows APIs through `ctypes` | Slideshow focus checks and keyboard output |

The root requirements also retain PyAutoGUI for older project utilities; the current slide controller uses Windows `SendInput`.

- [MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)
- [PyTorch installation guide](https://pytorch.org/get-started/locally/)
- [OpenCV Python packages](https://pypi.org/project/opencv-contrib-python/)
- [PowerPoint slideshow shortcuts](https://support.microsoft.com/en-us/accessibility/powerpoint/use-keyboard-shortcuts-to-deliver-powerpoint-presentations)
- [Windows SendInput](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-sendinput)

AirSlide was previously named GesturePilot. Older folder names, archive filenames and version tags may still use that name. The active app and its documentation use **AirSlide**.
