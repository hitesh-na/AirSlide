"""PowerPoint gesture control: dry run by default; --control enables keys."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import math
import sys
import time

from app import load_inputs
from camera_drawing import draw_hand
from controller import ControllerSettings, GestureController
from controller_preview import ControllerPreview
from live_utils import classify_result_with_features, video_timestamp
from powerpoint_windows import MouseCornerStop, TargetError, WindowsDesktop
from training_utils import project_path, write_json


class ControlSession:
    def __init__(self, settings, desktop, *, control=False):
        self.controller = GestureController(settings)
        self.desktop = desktop
        self.control = control
        self.target = None
        self.events = []
        self.message = 'Click ARM to begin.'

    def _event(self, now, kind, **details):
        item = dict(at_monotonic=now, kind=kind, **details)
        self.events.append(item)
        return item

    def arm(self, now):
        self.target = None
        self.controller.request_arm(now)
        self.message = 'Five-second countdown started.'
        self._event(now, 'arming')

    def pause(self, now, reason):
        self.controller.disarm()
        self.target = None
        self.message = reason
        self._event(now, 'disarmed', reason=reason)

    def _delivery_checks(self):
        self.desktop.check_mouse_stop()
        if not self.desktop.matches(self.target):
            raise TargetError('PowerPoint lost focus: arm again from the preview')
        if self.desktop.modifiers_down():
            raise TargetError('Release Shift/Ctrl/Alt/Windows keys, then arm again')

    def update(self, now, prediction):
        controller = self.controller
        if controller.phase != 'DISARMED':
            try:
                self.desktop.check_mouse_stop()
                if controller.phase == 'ARMING':
                    if now + 1e-9 >= controller.deadline:
                        self.target = self.desktop.capture_slideshow()
                        self.message = 'PowerPoint slideshow verified.'
                        self._event(now, 'armed')
                elif not self.desktop.matches(self.target):
                    raise TargetError('PowerPoint lost focus: arm again from the preview')
            except (MouseCornerStop, TargetError, OSError) as exc:
                self.pause(now, str(exc))
        action = controller.update(now, prediction)
        if action is None:
            return None
        try:
            # Recheck immediately before the output path, including in dry run.
            self._delivery_checks()
            if self.control:
                self.desktop.press_key(action.key)
                kind = 'sent'
            else:
                kind = 'would_send'
            self.message = f'{"SENT" if self.control else "WOULD SEND"}: {action.gesture}'
            return self._event(now, kind, gesture=action.gesture, key=action.key,
                               held_seconds=action.held_seconds)
        except (MouseCornerStop, TargetError, OSError, RuntimeError) as exc:
            self._event(now, 'blocked', gesture=action.gesture, key=action.key, reason=str(exc))
            self.pause(now, str(exc))
            return None


def print_control_events(events):
    for event in events:
        if event['kind'] == 'arming':
            print('ARMING: button/key received. Click the PowerPoint slideshow during the countdown.', flush=True)
        elif event['kind'] == 'armed':
            print('ARMED: PowerPoint slideshow verified; show open palm.', flush=True)
        elif event['kind'] == 'disarmed':
            print(f"PAUSED: {event['reason']}", flush=True)
        elif event['kind'] in ('sent', 'would_send'):
            print(f"{event['kind'].upper()}: {event['gesture']} -> {event['key']}", flush=True)


def run_camera(args, config, classifier, metadata, checkpoint_hash, desktop):
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    settings = ControllerSettings.from_config(config)
    session = ControlSession(settings, desktop, control=args.control)
    landmark_path = project_path(config['model_path'])
    if not landmark_path.is_file():
        raise FileNotFoundError(f'Missing hand landmark model: {landmark_path}')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    report_path = project_path(args.report or f'reports/control_{stamp}.json')
    if report_path.exists():
        raise ValueError('Controller report already exists; choose a new filename')
    camera_index = config['camera_index'] if args.camera is None else args.camera
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(landmark_path)),
        running_mode=vision.RunningMode.VIDEO, num_hands=1,
        min_hand_detection_confidence=.5, min_hand_presence_confidence=.5,
        min_tracking_confidence=.5)
    camera = None
    ready = time.perf_counter()
    last_timestamp = -1
    failures = frames = no_hand = 0
    labels = Counter()
    reason = 'user_exit'
    failure = None
    width = height = None
    window = 'AirSlide - PowerPoint control'
    mode = 'CONTROL' if args.control else 'DRY RUN - NO KEYS'
    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            camera = cv2.VideoCapture(camera_index)
            if not camera.isOpened():
                raise RuntimeError(f'Camera {camera_index} did not open')
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
            preview = ControllerPreview(cv2)
            preview.attach(window)
            if hasattr(cv2, 'WND_PROP_TOPMOST'):
                cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
            ready = time.perf_counter()
            print(f'{mode}: starts DISARMED. Click ARM in preview (or press A); then click the running PowerPoint slideshow.', flush=True)
            print('Open palm prepares one action. Hold index up / V / fist for an action; return to palm between actions.', flush=True)
            print('D in preview pauses. Q/Esc in preview or Ctrl+C in terminal exits. A primary-screen mouse corner pauses control.', flush=True)
            while True:
                started = time.perf_counter()
                ok, frame = camera.read()
                if not ok or frame is None or frame.size == 0:
                    failures += 1
                    if failures >= 10:
                        raise RuntimeError('Camera stopped returning frames')
                    # Reset pending holds on every missing camera frame.
                    session.update(time.perf_counter(), None)
                    time.sleep(.05)
                    continue
                failures = 0
                height, width = frame.shape[:2]
                last_timestamp = video_timestamp(last_timestamp, started-ready)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), last_timestamp)
                prediction, _ = classify_result_with_features(result, classifier)
                frames += 1
                no_hand += prediction is None
                if prediction is not None:
                    labels[prediction.label] += 1
                before = len(session.events)
                now = time.perf_counter()
                previous_phase = session.controller.phase
                session.update(now, prediction)
                print_control_events(session.events[before:])
                if session.controller.phase == 'READY' and previous_phase != 'READY':
                    print('READY: open palm accepted; hold an action gesture.', flush=True)
                if result.hand_landmarks:
                    draw_hand(cv2, frame, result.hand_landmarks[0])
                elapsed = max(.001, time.perf_counter()-ready)
                display = preview.render(frame, session, prediction, now, frames/elapsed)
                cv2.imshow(window, display)
                key = cv2.waitKey(1) & 0xff
                if key in (27, ord('q'), ord('Q')) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                before = len(session.events)
                command = preview.take_command(key)
                if command == 'arm':
                    session.arm(time.perf_counter())
                elif command == 'pause':
                    session.pause(time.perf_counter(), 'Paused by user')
                print_control_events(session.events[before:])
                if args.duration > 0 and time.perf_counter()-ready >= args.duration:
                    reason = 'duration_reached'
                    break
    except KeyboardInterrupt as exc:
        failure, reason = exc, 'keyboard_interrupt'
    except Exception as exc:
        failure, reason = exc, 'error'
    finally:
        session.controller.disarm()
        session.target = None
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
        print('Camera released; controller preview closed.', flush=True)
    end = time.perf_counter()
    counts = Counter(e['kind'] for e in session.events)
    report = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'mode': 'control' if args.control else 'dry-run', 'presentation_app': 'PowerPoint desktop',
        'preview_layout_version': 'panel_v1',
        'exit_reason': reason, 'error': str(failure) if failure else None,
        'camera_index': camera_index, 'frame_width': width, 'frame_height': height,
        'frames': frames, 'usable_hand_frames': frames-no_hand, 'no_usable_hand_frames': no_hand,
        'raw_label_counts': dict(labels), 'elapsed_seconds': max(0, end-ready),
        'actions_would_send': counts['would_send'], 'actions_sent': counts['sent'],
        'actions_blocked': counts['blocked'], 'arming_requests': counts['arming'],
        'settings': asdict(settings), 'events': session.events,
        'checkpoint_sha256': checkpoint_hash, 'dataset_sha256': metadata['dataset_sha256'],
        'model_input_version': metadata['model_input_version'], 'selected_epoch': metadata['selected_epoch'],
        'scope': 'Controller event audit. Sent means Windows accepted a key-down/key-up pair; slide changes and latency require observation. Not an accuracy evaluation.'}
    write_json(report_path, report)
    print(f'Controller report: {report_path}')
    print(f"Would-send actions: {counts['would_send']} | Sent actions: {counts['sent']} | Blocked actions: {counts['blocked']}")
    print(f'Hand frames: {frames-no_hand} | No-hand frames: {no_hand}')
    print('CONTROLLER SESSION ENDED' if failure is None else 'CONTROLLER SESSION STOPPED WITH ERROR')
    if failure is not None:
        raise failure
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--checkpoint', default='models/gesture_mlp.pt')
    parser.add_argument('--camera', type=int)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true', help='Display proposed actions without sending keys (default)')
    modes.add_argument('--control', action='store_true', help='Enable key output after deliberate arming and focus verification')
    parser.add_argument('--duration', type=float, default=0, help='Total preview seconds; 0 runs until Q/Esc/Ctrl+C')
    parser.add_argument('--device-label', choices=('desktop', 'laptop'), default='desktop')
    parser.add_argument('--report', help='New JSON report filename')
    parser.add_argument('--check', action='store_true', help='Check model, controller settings and Windows API layout without camera/key output')
    args = parser.parse_args()
    if args.camera is not None and args.camera < 0:
        parser.error('--camera must be non-negative')
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error('--duration must be finite and non-negative')
    config, classifier, metadata, checkpoint_hash = load_inputs(args)
    ControllerSettings.from_config(config)
    desktop = WindowsDesktop()
    if args.check:
        print('CONTROLLER CHECK PASSED - model/settings/Windows layout; no camera, PowerPoint attachment or key output.')
        return
    run_camera(args, config, classifier, metadata, checkpoint_hash, desktop)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except (ImportError, OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f'CONTROLLER FAILED: {exc}', file=sys.stderr)
        sys.exit(1)
