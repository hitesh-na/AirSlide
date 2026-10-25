# AirSlide PowerPoint control

AirSlide controls slides using the trained Palm V3 model on your Windows PC.
It uses the same classifier and model file as your successful pose checks.
The existing `app.py` remains available for prediction diagnostics.

## AirSlide name update

Extract `AirSlide_Rename.zip` into your existing `V:\Projects\GesturePilot`
folder, then use the same commands below. The preview header and window title
now read AirSlide. This patch contains branding and documentation changes.

## Preview readability update

If the old preview text looks doubled or is hard to read, extract
`GesturePilot_Readable_Preview.zip` into the existing `V:\Projects\GesturePilot`
folder, then run the same `control_slides.py` command. The update replaces
the overlay with a solid status panel beside the camera image. Text is
larger, wrapped to fit, and drawn once. The preview is 1024 by 576 pixels.

Close the old preview before replacing the files. For a readability check
without a time limit, run:

```powershell
..\.venv\Scripts\python.exe control_slides.py --dry-run --camera 0
```

Press Q in the preview to exit. You can leave it disarmed while checking
the text, then click ARM when the PowerPoint slideshow is ready.

Click the large **ARM [A]** button to start the five-second countdown.
The terminal immediately prints **ARMING: button/key received**. Select
your slideshow during that countdown. The **PAUSE [D]** button pauses;
the existing A/D/Q/Esc keyboard controls still work in the preview.
Status, detected pose, messages and action count each have their own area.
The update keeps the same trained model, gesture timings and focus checks.
It needs no new package installation and sends no keys in dry run.

## First run

Extract `GesturePilot_PowerPoint_Control.zip` into `V:\Projects\GesturePilot`.
The archive adds files under `palm_v3/`; it contains no dataset or model.

From PowerShell:

```powershell
Set-Location "V:\Projects\GesturePilot\palm_v3"
..\.venv\Scripts\python.exe -m unittest discover -s tests -p test_controller.py -v
..\.venv\Scripts\python.exe control_slides.py --check
```

Open a short deck in the desktop PowerPoint app and press F5 to start the
slideshow. Return to PowerShell and start a one-minute dry run:

```powershell
..\.venv\Scripts\python.exe control_slides.py --dry-run --camera 0 --duration 60
```

1. Click **ARM [A]** in the camera preview (or click the preview and press A).
2. During the five-second countdown, click the **PowerPoint slideshow**.
3. Show an open palm for about one second. The preview should say **READY**.
4. Hold an action gesture for about one second. The preview and terminal
   show **WOULD SEND**. The slides stay in place during this dry run.
5. Return to an open palm before the next action. Holding the action pose
   continuously must produce only one proposed action.

| Right-hand pose | PowerPoint action |
|---|---|
| Open palm | Prepare one action; sends no key |
| Index finger up, other fingers folded | Right arrow: next animation or slide |
| V sign | Left arrow: previous animation or slide |
| Closed fist | B: toggle black screen |

Test palm → index → palm → V → palm → fist. Keep the fist held for a few
seconds to check that it does not repeat. Remove your hand once; the next
action must wait for a new open palm. Switching away from the slideshow
pauses the controller. It will require **A** and a new countdown to resume.

The preview stays on top so you can see its status while PowerPoint has
focus. On multiple monitors, select the actual slide-show window rather
than Presenter View or the editor. Preview keys apply when the preview
has focus; they are not global keyboard shortcuts.

## Real slide control

After the dry run works, use this command with the slideshow already open:

```powershell
..\.venv\Scripts\python.exe control_slides.py --control --camera 0
```

Click ARM in the preview, click the slideshow during the countdown, then
use an open palm followed by an action. This command runs until you exit.
`--duration 60` can also be added for a timed session.

Clicking the preview pauses delivery because PowerPoint loses focus.
Press **D** there to keep it disarmed. Press **Q** or **Esc** in the preview
to exit, or Ctrl+C in the terminal. Moving the mouse into a corner of the
primary screen also pauses control. Move it away before arming again.
Normal manual PowerPoint shortcuts continue to work.

## Timing and behavior

The controller uses the existing configuration: score threshold0.85,
action hold0.5s, neutral hold0.3s, minimum action interval0.8s and arming
countdown5s. A low score or changed gesture resets the action hold. Hand
loss or a prediction gap longer than0.25s clears readiness and requires a
new neutral palm. Each action consumes one neutral-palm preparation, so
holding a fist does not repeatedly toggle black screen.

Both dry run and control start disarmed and check the selected target.
The Windows target must be a desktop PowerPoint slideshow (`screenClass`
owned by `POWERPNT.EXE`). The same window and process must retain focus;
the check runs again just before delivery. The app does not move focus or
start a slideshow automatically. Keyboard modifiers or an already-held
shortcut key prevent delivery. The model's scores remain uncalibrated.

Keyboard output uses Windows SendInput through Python's standard library,
with paired key-down/key-up events. A partially accepted pair triggers a
key-up cleanup attempt and pauses the controller. No new package install
is needed. This replaces the early PyAutoGUI implementation idea while
retaining a mouse-corner stop and adding a PowerPoint focus check.

## Reports and verification

Each session writes a new `reports/control_<UTC>.json` with mode, model/data
fingerprints, frame/no-hand counts and controller events. `would_send`
events are dry-run proposals; `sent` means Windows accepted the keyboard
pair, not that the slide change was independently observed. Reports never
append training samples or modify weights. Error/early-exit reports state
their actual exit reason; camera resources are released on Q, Ctrl+C and
camera errors.

The 28 supplied controller tests cover timing, reset rules, one action per
neutral preparation, dry-run isolation, focus/target checks, mouse stops,
keyboard layout/pairs, partial-send cleanup, mouse-button arming and camera cleanup through
injected test backends. They ran in the assistant environment without
sending real keys. Actual Windows delivery, PowerPoint behavior and the
new controller's live performance still require your PC check.

Official references:

- [PowerPoint slideshow shortcuts](https://support.microsoft.com/en-us/accessibility/powerpoint/use-keyboard-shortcuts-to-deliver-powerpoint-presentations)
- [Windows SendInput](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-sendinput)
- [Windows GetForegroundWindow](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getforegroundwindow)
- [Windows QueryFullProcessImageNameW](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-queryfullprocessimagenamew)
- [WebRTC PowerPoint slideshow-window detection](https://webrtc.googlesource.com/src/+/f17e7061c01805b81738d0ba9b8c9650e060892d/modules/desktop_capture/win/full_screen_win_application_handler.cc)
