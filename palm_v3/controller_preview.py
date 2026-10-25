"""Readable controller preview; camera pixels and status text stay separate."""
import math

import numpy as np


WIDTH, HEIGHT = 1024, 576
ARM_BUTTON = (700, 18, 836, 76)
PAUSE_BUTTON = (850, 18, 1008, 76)


class ControllerPreview:
    def __init__(self, cv2):
        self.cv2 = cv2
        self.pending_command = None

    def attach(self, window):
        self.cv2.setMouseCallback(window, self.on_mouse)

    def on_mouse(self, event, x, y, flags, param):
        if event != self.cv2.EVENT_LBUTTONUP:
            return
        for bounds, command in ((ARM_BUTTON, 'arm'), (PAUSE_BUTTON, 'pause')):
            left, top, right, bottom = bounds
            if left <= x < right and top <= y < bottom:
                self.pending_command = command

    def take_command(self, key):
        command, self.pending_command = self.pending_command, None
        if key in (ord('a'), ord('A')):
            return 'arm'
        if key in (ord('d'), ord('D')):
            return 'pause'
        return command

    def _size(self, text, scale, thickness=1):
        return self.cv2.getTextSize(text, self.cv2.FONT_HERSHEY_SIMPLEX,
                                    scale, thickness)[0]

    def _text(self, image, text, x, y, *, scale=.66, color=(238, 238, 238),
              thickness=1, width=440, lines=1):
        # Wrap to measured pixel width. Render each line once on a solid fill.
        pending, wrapped = str(text), []
        while pending:
            end = len(pending)
            while end > 1 and self._size(pending[:end], scale, thickness)[0] > width:
                end -= 1
            if end < len(pending):
                space = pending.rfind(' ', 0, end + 1)
                if space > 0:
                    end = space
            wrapped.append(pending[:end].rstrip())
            pending = pending[end:].lstrip()
        if len(wrapped) > lines:
            wrapped = wrapped[:lines]
            last = wrapped[-1]
            while last and self._size(last + '...', scale, thickness)[0] > width:
                last = last[:-1]
            wrapped[-1] = last.rstrip() + '...'
        for i, line in enumerate(wrapped):
            self.cv2.putText(image, line, (x, y + 28 * i),
                             self.cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                             thickness, self.cv2.LINE_AA)

    def _button(self, image, bounds, label, color):
        x1, y1, x2, y2 = bounds
        self.cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
        width, height = self._size(label, .66, 2)
        self._text(image, label, (x1 + x2 - width) // 2,
                   (y1 + y2 + height) // 2, scale=.66, thickness=2,
                   width=x2-x1-12)

    def render(self, frame, session, prediction, now, fps):
        cv2 = self.cv2
        image = np.full((HEIGHT, WIDTH, 3), (24, 22, 20), dtype=np.uint8)
        self._text(image, 'AirSlide', 20, 40, scale=.94, thickness=2)
        mode = 'CONTROL - KEYS ENABLED' if session.control else 'DRY RUN - NO KEYS'
        self._text(image, mode, 20, 77, scale=.65, color=(90, 210, 255))
        self._button(image, ARM_BUTTON, 'ARM [A]', (135, 92, 30))
        self._button(image, PAUSE_BUTTON, 'PAUSE [D]', (62, 65, 72))

        cv2.rectangle(image, (16, 110), (496, 470), (0, 0, 0), -1)
        height, width = frame.shape[:2]
        ratio = min(480 / width, 360 / height)
        new_width, new_height = max(1, round(width*ratio)), max(1, round(height*ratio))
        camera = cv2.resize(cv2.flip(frame, 1), (new_width, new_height))
        x, y = 16 + (480-new_width)//2, 110 + (360-new_height)//2
        image[y:y+new_height, x:x+new_width] = camera
        self._text(image, 'Use your RIGHT hand', 20, 509, scale=.7)
        self._text(image, 'Open palm before every action', 20, 544, scale=.63)

        cv2.rectangle(image, (516, 96), (1008, 558), (39, 35, 31), -1)
        controller = session.controller
        phase = controller.phase
        paused = phase == 'DISARMED' and session.events and session.events[-1]['kind'] == 'disarmed'
        state = 'PAUSED' if paused else phase
        color = (110, 220, 130) if phase == 'READY' else (90, 210, 255)
        if phase == 'ARMING':
            remaining = max(0, math.ceil(controller.deadline-now))
            state = f'ARMING - {remaining}s'
        self._text(image, 'STATUS', 536, 126, scale=.52, color=(185, 185, 185))
        self._text(image, state, 536, 166, scale=1.02, thickness=2, color=color)
        if phase == 'DISARMED':
            instruction = 'Click ARM, then select your slideshow.'
        elif phase == 'ARMING':
            instruction = 'Click the PowerPoint slideshow now.'
        elif phase == 'NEED NEUTRAL':
            instruction = 'Hold an open palm until READY appears.'
        else:
            instruction = 'Hold index up, a V sign, or a fist.'
        self._text(image, instruction, 536, 206, lines=2)

        self._text(image, 'DETECTED HAND', 536, 280, scale=.52, color=(185, 185, 185))
        names = {'neutral': 'OPEN PALM', 'next': 'INDEX UP / NEXT',
                 'previous': 'V SIGN / PREVIOUS', 'blackout': 'FIST / BLACK SCREEN'}
        label = 'NO USABLE HAND' if prediction is None else names.get(prediction.label, prediction.label)
        self._text(image, label, 536, 317, scale=.8, thickness=2)
        if prediction is not None:
            score = f'Score {prediction.score:.3f}'
            if prediction.score < controller.settings.score_threshold:
                score += ' - hold still'
            self._text(image, score, 536, 350, scale=.61)

        self._text(image, 'LAST MESSAGE', 536, 393, scale=.52, color=(185, 185, 185))
        self._text(image, session.message, 536, 424, scale=.61, lines=3)
        actions = sum(event['kind'] in ('would_send', 'sent') for event in session.events)
        self._text(image, f'Actions: {actions}    FPS: {fps:.1f}', 536, 537, scale=.58)
        self._text(image, 'Q / Esc in this window exits', 390, 77, scale=.53, width=290)
        return image
