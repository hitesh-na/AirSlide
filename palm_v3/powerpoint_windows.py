"""PowerPoint foreground verification and Windows key events; no extra packages."""
import ctypes as ct
from dataclasses import dataclass
import os
import ntpath


class POINT(ct.Structure):
    _fields_ = [('x', ct.c_int32), ('y', ct.c_int32)]


class MOUSEINPUT(ct.Structure):
    _fields_ = [('dx', ct.c_int32), ('dy', ct.c_int32), ('mouseData', ct.c_uint32),
                ('dwFlags', ct.c_uint32), ('time', ct.c_uint32), ('dwExtraInfo', ct.c_size_t)]


class KEYBDINPUT(ct.Structure):
    _fields_ = [('wVk', ct.c_uint16), ('wScan', ct.c_uint16), ('dwFlags', ct.c_uint32),
                ('time', ct.c_uint32), ('dwExtraInfo', ct.c_size_t)]


class HARDWAREINPUT(ct.Structure):
    _fields_ = [('uMsg', ct.c_uint32), ('wParamL', ct.c_uint16), ('wParamH', ct.c_uint16)]


class INPUTUNION(ct.Union):
    _fields_ = [('mi', MOUSEINPUT), ('ki', KEYBDINPUT), ('hi', HARDWAREINPUT)]


class INPUT(ct.Structure):
    _anonymous_ = ('event',)
    _fields_ = [('type', ct.c_uint32), ('event', INPUTUNION)]


@dataclass(frozen=True)
class SlideshowTarget:
    hwnd: int
    process_id: int


class TargetError(RuntimeError):
    pass


class MouseCornerStop(RuntimeError):
    pass


class WindowsDesktop:
    def __init__(self, user32=None, kernel32=None):
        if user32 is None:
            if os.name != 'nt':
                raise RuntimeError('PowerPoint control uses the Windows desktop app')
            user32 = ct.WinDLL('user32', use_last_error=True)
        if kernel32 is None:
            if os.name != 'nt':
                raise RuntimeError('Windows process APIs are required')
            kernel32 = ct.WinDLL('kernel32', use_last_error=True)
        self.user32, self.kernel32 = user32, kernel32
        signatures = {
            'GetForegroundWindow': ([], ct.c_void_p),
            'GetClassNameW': ([ct.c_void_p, ct.c_wchar_p, ct.c_int], ct.c_int),
            'GetWindowThreadProcessId': ([ct.c_void_p, ct.POINTER(ct.c_uint32)], ct.c_uint32),
            'IsWindow': ([ct.c_void_p], ct.c_int),
            'GetCursorPos': ([ct.POINTER(POINT)], ct.c_int),
            'GetSystemMetrics': ([ct.c_int], ct.c_int),
            'GetAsyncKeyState': ([ct.c_int], ct.c_int16),
            'SendInput': ([ct.c_uint32, ct.POINTER(INPUT), ct.c_int], ct.c_uint32),
        }
        for name, (args, result) in signatures.items():
            function = getattr(user32, name)
            function.argtypes, function.restype = args, result
        for name, (args, result) in {
            'OpenProcess': ([ct.c_uint32, ct.c_int, ct.c_uint32], ct.c_void_p),
            'QueryFullProcessImageNameW': ([ct.c_void_p, ct.c_uint32, ct.c_wchar_p, ct.POINTER(ct.c_uint32)], ct.c_int),
            'CloseHandle': ([ct.c_void_p], ct.c_int),
        }.items():
            function = getattr(kernel32, name)
            function.argtypes, function.restype = args, result
        expected = 40 if ct.sizeof(ct.c_void_p) == 8 else 28
        if ct.sizeof(INPUT) != expected:
            raise RuntimeError('Windows keyboard input structure has the wrong size')

    def foreground(self):
        return int(self.user32.GetForegroundWindow() or 0)

    def process_id(self, hwnd):
        value = ct.c_uint32()
        if not self.user32.GetWindowThreadProcessId(hwnd, ct.byref(value)) or value.value == 0:
            raise TargetError('PowerPoint window is no longer available')
        return value.value

    def window_class(self, hwnd):
        text = ct.create_unicode_buffer(256)
        if not self.user32.GetClassNameW(hwnd, text, len(text)):
            raise TargetError('Could not verify the PowerPoint window')
        return text.value

    def process_image(self, process_id):
        handle = self.kernel32.OpenProcess(0x1000, False, process_id)
        if not handle:
            raise TargetError('Could not verify the PowerPoint process')
        try:
            text = ct.create_unicode_buffer(32768)
            size = ct.c_uint32(len(text))
            if not self.kernel32.QueryFullProcessImageNameW(handle, 0, text, ct.byref(size)):
                raise TargetError('Could not read the PowerPoint executable name')
            return text.value
        finally:
            self.kernel32.CloseHandle(handle)

    def capture_slideshow(self):
        foreground = self.foreground()
        if not foreground or not self.user32.IsWindow(foreground):
            raise TargetError('Click the PowerPoint slideshow during the countdown')
        pid = self.process_id(foreground)
        # Desktop PowerPoint slide shows use screenClass; the editor does not.
        # Also verify the executable, not just a title or class-name match.
        if (ntpath.basename(self.process_image(pid)).casefold() != 'powerpnt.exe'
                or self.window_class(foreground).casefold() != 'screenclass'):
            raise TargetError('Start desktop PowerPoint with F5 and click the slide show, then arm again')
        if self.foreground() != foreground:
            raise TargetError('PowerPoint lost focus during the window check; arm again')
        return SlideshowTarget(foreground, pid)

    def matches(self, target):
        if target is None or not self.user32.IsWindow(target.hwnd) or self.foreground() != target.hwnd:
            return False
        try:
            return (self.process_id(target.hwnd) == target.process_id
                    and self.window_class(target.hwnd).casefold() == 'screenclass')
        except TargetError:
            return False

    def check_mouse_stop(self):
        point = POINT()
        if not self.user32.GetCursorPos(ct.byref(point)):
            raise MouseCornerStop('Could not read mouse position; controller paused')
        width, height = self.user32.GetSystemMetrics(0), self.user32.GetSystemMetrics(1)
        if width <= 0 or height <= 0:
            raise MouseCornerStop('Could not read primary screen size; controller paused')
        if point.x in (0, width-1) and point.y in (0, height-1):
            raise MouseCornerStop('Mouse corner stop: move the cursor away and arm again')

    def modifiers_down(self):
        return any(int(self.user32.GetAsyncKeyState(key)) & 0x8000 for key in (0x10, 0x11, 0x12, 0x5b, 0x5c))

    def press_key(self, key):
        mapping = {'right': (0x27, 1), 'left': (0x25, 1), 'b': (0x42, 0)}
        if key not in mapping:
            raise ValueError('Unsupported PowerPoint shortcut')
        vk, extended = mapping[key]
        if int(self.user32.GetAsyncKeyState(vk)) & 0x8000:
            raise RuntimeError('Shortcut key is already held; release it and arm again')
        down = INPUT(type=1, ki=KEYBDINPUT(wVk=vk, dwFlags=extended))
        up = INPUT(type=1, ki=KEYBDINPUT(wVk=vk, dwFlags=extended | 2))
        batch = (INPUT * 2)(down, up)
        count = int(self.user32.SendInput(2, batch, ct.sizeof(INPUT)))
        if count != 2:
            # A partially accepted batch may have inserted only key-down.
            # Always attempt key-up before reporting failure and disarming.
            self.user32.SendInput(1, (INPUT * 1)(up), ct.sizeof(INPUT))
            raise RuntimeError(f'Windows accepted {count}/2 keyboard events; controller paused')
