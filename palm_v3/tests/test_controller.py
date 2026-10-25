import contextlib
import ctypes as ct
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from controller import ControllerSettings, GestureController
from control_slides import ControlSession, run_camera
from powerpoint_windows import INPUT, POINT, MouseCornerStop, SlideshowTarget, TargetError, WindowsDesktop


def prediction(label, score=.99):
    return SimpleNamespace(label=label, score=score)


def feed(controller, label, start, duration, score=.99):
    actions = []
    count = round(duration/.05)
    for i in range(count+1):
        action = controller.update(round(start+i*.05, 8), None if label is None else prediction(label, score))
        if action is not None:
            actions.append(action)
    return actions


def ready_controller(**overrides):
    controller = GestureController(ControllerSettings(arming_countdown_seconds=0, **overrides))
    controller.request_arm(0.)
    feed(controller, 'neutral', 0, .3)
    assert controller.phase == 'READY'
    return controller


class ControllerTests(unittest.TestCase):
    def test_starts_disarmed_and_never_emits_without_arming(self):
        controller = GestureController(ControllerSettings())
        self.assertEqual(feed(controller, 'next', 0, 4), [])
        self.assertEqual(controller.phase, 'DISARMED')

    def test_full_countdown_and_neutral_hold_are_required(self):
        controller = GestureController(ControllerSettings())
        controller.request_arm(0)
        self.assertEqual(feed(controller, 'next', 0, 5), [])
        self.assertEqual(controller.phase, 'NEED NEUTRAL')
        feed(controller, 'neutral', 5.05, .25)
        self.assertEqual(controller.phase, 'NEED NEUTRAL')
        controller.update(5.35, prediction('neutral'))
        self.assertEqual(controller.phase, 'READY')

    def test_short_pose_does_not_emit_and_boundary_emits_once(self):
        controller = ready_controller()
        self.assertEqual(feed(controller, 'next', .35, .45), [])
        action = controller.update(.85, prediction('next'))
        self.assertEqual((action.gesture, action.key), ('next', 'right'))
        self.assertEqual(feed(controller, 'next', .9, 3), [])
        self.assertEqual(controller.phase, 'NEED NEUTRAL')

    def test_all_three_actions_use_powerpoint_shortcuts(self):
        for label, key in [('next', 'right'), ('previous', 'left'), ('blackout', 'b')]:
            controller = ready_controller()
            actions = feed(controller, label, .35, .5)
            self.assertEqual([(a.gesture, a.key) for a in actions], [(label, key)])

    def test_action_changes_reset_hold(self):
        controller = ready_controller()
        self.assertEqual(feed(controller, 'next', .35, .25), [])
        self.assertEqual(feed(controller, 'previous', .65, .45), [])
        action = controller.update(1.15, prediction('previous'))
        self.assertEqual(action.gesture, 'previous')

    def test_low_score_resets_action_timer(self):
        controller = ready_controller()
        self.assertEqual(feed(controller, 'next', .35, .3), [])
        controller.update(.7, prediction('next', .84))
        self.assertEqual(feed(controller, 'next', .75, .45), [])
        self.assertEqual(controller.update(1.25, prediction('next')).gesture, 'next')

    def test_unknown_and_nonfinite_predictions_cannot_emit(self):
        for label, score in [('start', .99), ('next', float('nan')), ('next', 1.1)]:
            controller = ready_controller()
            self.assertEqual(feed(controller, label, .35, 1, score), [])

    def test_no_hand_consumes_readiness_and_requires_neutral_again(self):
        controller = ready_controller()
        controller.update(.35, None)
        self.assertEqual(feed(controller, 'blackout', .4, 2), [])
        feed(controller, 'neutral', 2.45, .3)
        self.assertEqual(len(feed(controller, 'blackout', 2.8, .5)), 1)

    def test_held_action_and_other_actions_do_not_repeat_without_palm(self):
        controller = ready_controller()
        self.assertEqual(len(feed(controller, 'blackout', .35, .5)), 1)
        self.assertEqual(feed(controller, 'blackout', .9, 2), [])
        self.assertEqual(feed(controller, 'next', 2.95, 1), [])
        feed(controller, 'neutral', 4, .3)
        self.assertEqual(len(feed(controller, 'next', 4.35, .5)), 1)

    def test_cooldown_is_enforced_even_after_neutral(self):
        controller = ready_controller(minimum_action_interval_seconds=2)
        feed(controller, 'next', .35, .5)
        feed(controller, 'neutral', .9, .3)
        self.assertEqual(feed(controller, 'previous', 1.25, 1.55), [])
        self.assertEqual(controller.update(2.85, prediction('previous')).key, 'left')

    def test_long_gap_cannot_complete_a_gesture_hold(self):
        controller = ready_controller()
        feed(controller, 'next', .35, .2)
        self.assertIsNone(controller.update(2., prediction('next')))
        self.assertEqual(feed(controller, 'next', 2.05, .4), [])

    def test_disarm_resets_pending_actions(self):
        controller = ready_controller()
        feed(controller, 'next', .35, .3)
        controller.disarm()
        self.assertEqual(feed(controller, 'next', .7, 1), [])

    def test_invalid_timers_and_backward_clock_are_rejected(self):
        for settings in ({'gesture_hold_seconds': 0}, {'score_threshold': float('nan')},
                         {'score_threshold': True}, {'arming_countdown_seconds': -1}):
            with self.assertRaises(ValueError):
                ControllerSettings(**settings)
        controller = ready_controller()
        with self.assertRaises(ValueError):
            controller.update(.1, prediction('next'))


class FakeDesktop:
    def __init__(self):
        self.focus = True
        self.corner = False
        self.modifiers = False
        self.keys = []
        self.fail_key = False
        self.valid_target = True
        self.match_calls = 0
        self.lose_at_match = None

    def capture_slideshow(self):
        if not self.valid_target:
            raise TargetError('PowerPoint slideshow not active')
        return SlideshowTarget(42, 7)

    def matches(self, target):
        self.match_calls += 1
        if self.lose_at_match == self.match_calls:
            return False
        return self.focus and target is not None

    def check_mouse_stop(self):
        if self.corner:
            raise MouseCornerStop('Mouse corner stop')

    def modifiers_down(self):
        return self.modifiers

    def press_key(self, key):
        if self.fail_key:
            raise RuntimeError('Windows refused keyboard event')
        self.keys.append(key)


def ready_session(*, control=False):
    desktop = FakeDesktop()
    session = ControlSession(ControllerSettings(arming_countdown_seconds=0), desktop, control=control)
    session.arm(0.)
    feed(session, 'neutral', 0, .3)
    return session, desktop


class SessionTests(unittest.TestCase):
    def test_dry_run_never_sends_keys(self):
        session, desktop = ready_session()
        feed(session, 'next', .35, .5)
        self.assertEqual(desktop.keys, [])
        self.assertEqual([e['kind'] for e in session.events].count('would_send'), 1)

    def test_control_requires_arming_then_sends_exactly_once(self):
        session = ControlSession(ControllerSettings(arming_countdown_seconds=0), FakeDesktop(), control=True)
        feed(session, 'next', 0, 1)
        self.assertEqual(session.desktop.keys, [])
        session.arm(1.1)
        feed(session, 'neutral', 1.1, .3)
        feed(session, 'next', 1.45, 1)
        self.assertEqual(session.desktop.keys, ['right'])

    def test_wrong_window_cannot_arm(self):
        desktop = FakeDesktop(); desktop.valid_target = False
        session = ControlSession(ControllerSettings(arming_countdown_seconds=0), desktop, control=True)
        session.arm(0); feed(session, 'neutral', 0, .3)
        self.assertEqual(session.controller.phase, 'DISARMED')
        self.assertEqual(desktop.keys, [])

    def test_focus_loss_disarms_and_does_not_resume_automatically(self):
        session, desktop = ready_session(control=True)
        desktop.focus = False
        session.update(.35, prediction('next'))
        desktop.focus = True
        feed(session, 'next', .4, 1)
        self.assertEqual(session.controller.phase, 'DISARMED')
        self.assertEqual(desktop.keys, [])

    def test_focus_is_rechecked_immediately_before_delivery(self):
        session, desktop = ready_session(control=True)
        feed(session, 'next', .35, .45)
        desktop.lose_at_match = desktop.match_calls + 2
        session.update(.85, prediction('next'))
        self.assertEqual(desktop.keys, [])
        self.assertEqual(session.controller.phase, 'DISARMED')
        self.assertEqual([e['kind'] for e in session.events].count('blocked'), 1)

    def test_mouse_stop_modifiers_and_send_failure_disarm(self):
        for kind in ('corner', 'modifiers', 'fail_key'):
            session, desktop = ready_session(control=True)
            setattr(desktop, kind, True)
            feed(session, 'next', .35, .5)
            self.assertEqual(desktop.keys, [], kind)
            self.assertEqual(session.controller.phase, 'DISARMED', kind)


class Function:
    def __init__(self, call):
        self.call = call
    def __call__(self, *args):
        return self.call(*args)


class FakeWindows:
    def __init__(self):
        self.hwnd, self.pid = 42, 7
        self.image, self.klass = r'C:\Office\POWERPNT.EXE', 'screenClass'
        self.cursor = (300, 300)
        self.inject_count = 2
        self.batches = []
        self.closed = []
        self.query_ok = True
        self.user32 = SimpleNamespace(
            GetForegroundWindow=Function(lambda: self.hwnd),
            IsWindow=Function(lambda h: h == 42),
            GetWindowThreadProcessId=Function(self.pid_read),
            GetClassNameW=Function(self.class_read),
            GetCursorPos=Function(self.cursor_read),
            GetSystemMetrics=Function(lambda i: 1920 if i == 0 else 1080),
            GetAsyncKeyState=Function(lambda k: 0),
            SendInput=Function(self.send))
        self.kernel32 = SimpleNamespace(OpenProcess=Function(lambda *a: 900),
                                        QueryFullProcessImageNameW=Function(self.image_read),
                                        CloseHandle=Function(lambda h: self.closed.append(h) or 1))
        self.desktop = WindowsDesktop(self.user32, self.kernel32)

    def pid_read(self, hwnd, out):
        ct.cast(out, ct.POINTER(ct.c_uint32)).contents.value = self.pid
        return 10

    def class_read(self, hwnd, out, size):
        out.value = self.klass
        return len(self.klass)

    def image_read(self, handle, flags, out, size):
        if not self.query_ok:
            return 0
        out.value = self.image
        return 1

    def cursor_read(self, out):
        point = ct.cast(out, ct.POINTER(POINT)).contents
        point.x, point.y = self.cursor
        return 1

    def send(self, count, inputs, size):
        self.batches.append([(inputs[i].ki.wVk, inputs[i].ki.dwFlags) for i in range(count)])
        return self.inject_count if count == 2 else 1


class WindowsOutputTests(unittest.TestCase):
    def test_layout_and_key_down_up_pairs(self):
        windows = FakeWindows()
        self.assertEqual(ct.sizeof(INPUT), 40 if ct.sizeof(ct.c_void_p) == 8 else 28)
        windows.desktop.press_key('right'); windows.desktop.press_key('left'); windows.desktop.press_key('b')
        self.assertEqual(windows.batches, [[(0x27, 1), (0x27, 3)], [(0x25, 1), (0x25, 3)], [(0x42, 0), (0x42, 2)]])

    def test_partial_input_releases_key_and_raises(self):
        windows = FakeWindows(); windows.inject_count = 1
        with self.assertRaisesRegex(RuntimeError, '1/2'):
            windows.desktop.press_key('b')
        self.assertEqual(windows.batches[-1], [(0x42, 2)])

    def test_already_held_shortcut_is_not_injected(self):
        windows = FakeWindows()
        windows.user32.GetAsyncKeyState.call = lambda key: 0x8000 if key == 0x42 else 0
        with self.assertRaisesRegex(RuntimeError, 'already held'):
            windows.desktop.press_key('b')
        self.assertEqual(windows.batches, [])

    def test_target_checks_process_class_focus_and_pid(self):
        windows = FakeWindows()
        target = windows.desktop.capture_slideshow()
        self.assertTrue(windows.desktop.matches(target))
        self.assertEqual(windows.closed, [900])
        windows.pid = 8
        self.assertFalse(windows.desktop.matches(target))
        windows.pid = 7; windows.hwnd = 99
        self.assertFalse(windows.desktop.matches(target))
        windows.hwnd = 42; windows.klass = 'PPTFrameClass'
        with self.assertRaises(TargetError): windows.desktop.capture_slideshow()
        windows.klass = 'screenClass'; windows.image = r'C:\Other\notepad.exe'
        with self.assertRaises(TargetError): windows.desktop.capture_slideshow()

    def test_process_handle_is_closed_on_query_error(self):
        windows = FakeWindows(); windows.query_ok = False
        with self.assertRaises(TargetError): windows.desktop.capture_slideshow()
        self.assertEqual(windows.closed, [900])

    def test_primary_screen_corners_stop_and_other_positions_do_not(self):
        windows = FakeWindows()
        windows.desktop.check_mouse_stop()
        for corner in [(0, 0), (1919, 0), (0, 1079), (1919, 1079)]:
            windows.cursor = corner
            with self.assertRaises(MouseCornerStop): windows.desktop.check_mouse_stop()


class CameraCleanupTests(unittest.TestCase):
    def exercise(self, fail=False, mouse=False):
        self.camera_released = self.windows_closed = self.landmarker_closed = False
        owner = self
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        class Camera:
            def isOpened(self): return True
            def set(self, *args): pass
            def read(self): return (False, None) if fail else (True, frame)
            def release(self): owner.camera_released = True
        class Landmarker:
            def __enter__(self): return self
            def __exit__(self, *args): owner.landmarker_closed = True
            def detect_for_video(self, *args): return SimpleNamespace(hand_world_landmarks=[], hand_landmarks=[])
        keys = iter([255 if mouse else ord('a'), 0, ord('q')])
        clicked = False
        def callback(window, handler):
            owner.mouse_callback = handler
        def wait_key(*args):
            nonlocal clicked
            if mouse and not clicked:
                owner.mouse_callback(4, 760, 40, 0, None)
                clicked = True
            return next(keys)
        def close(): owner.windows_closed = True
        cv2 = SimpleNamespace(VideoCapture=lambda *a: Camera(), CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4,
                              WINDOW_AUTOSIZE=1, WND_PROP_VISIBLE=1, COLOR_BGR2RGB=1, FONT_HERSHEY_SIMPLEX=0,
                              EVENT_LBUTTONUP=4, LINE_AA=16, setMouseCallback=callback,
                              namedWindow=lambda *a: None, cvtColor=lambda x, *a: x, flip=lambda x, *a: x,
                              resize=lambda x, size: np.zeros((size[1], size[0], 3), dtype=np.uint8),
                              getTextSize=lambda text, font, scale, thickness: ((round(len(text)*scale*18), round(scale*22)), 5),
                              rectangle=lambda *a: None, putText=lambda *a: None,
                              imshow=lambda *a: None, waitKey=wait_key,
                              getWindowProperty=lambda *a: 1, destroyAllWindows=close)
        vision = SimpleNamespace(HandLandmarkerOptions=lambda **kw: object(), RunningMode=SimpleNamespace(VIDEO=1),
                                 HandLandmarker=SimpleNamespace(create_from_options=lambda *a: Landmarker()))
        python = SimpleNamespace(BaseOptions=lambda **kw: object(), vision=vision)
        mp = SimpleNamespace(Image=lambda **kw: object(), ImageFormat=SimpleNamespace(SRGB=1), tasks=SimpleNamespace(python=python))
        modules = {'cv2': cv2, 'mediapipe': mp, 'mediapipe.tasks': mp.tasks, 'mediapipe.tasks.python': python}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); (path/'model.task').write_bytes(b'fixture')
            config = {'controller': vars(ControllerSettings()), 'model_path': str(path/'model.task'), 'camera_index': 0}
            args = SimpleNamespace(control=False, camera=0, report=str(path/'report.json'), duration=0)
            meta = dict(dataset_sha256='a'*64, model_input_version='palm_frame_v1', selected_epoch=100)
            console = io.StringIO()
            with mock.patch.dict(sys.modules, modules), mock.patch('control_slides.time.sleep'), contextlib.redirect_stdout(console):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, 'Camera stopped'):
                        run_camera(args, config, object(), meta, 'b'*64, FakeDesktop())
                else:
                    run_camera(args, config, object(), meta, 'b'*64, FakeDesktop())
            report = json.loads((path/'report.json').read_text())
            self.console = console.getvalue()
        self.assertTrue(self.camera_released and self.windows_closed and self.landmarker_closed)
        self.assertEqual(report['actions_sent'], 0)
        return report

    def test_q_exit_releases_resources_and_counts_no_hand(self):
        report = self.exercise()
        self.assertEqual(report['exit_reason'], 'user_exit')
        self.assertEqual(report['no_usable_hand_frames'], 3)

    def test_click_arm_is_audited_and_logged_without_output(self):
        report = self.exercise(mouse=True)
        self.assertEqual(report['arming_requests'], 1)
        self.assertIn('ARMING: button/key received.', self.console)
        self.assertEqual(report['actions_sent'], 0)
        self.assertEqual(report['exit_reason'], 'user_exit')

    def test_camera_failure_releases_resources_and_reports_error(self):
        report = self.exercise(fail=True)
        self.assertEqual(report['exit_reason'], 'error')
        self.assertIn('Camera stopped', report['error'])


if __name__ == '__main__': unittest.main()
