"""Draw detected hand landmarks; no model or camera initialization."""

import math
import numpy as np

CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


def draw_hand(cv2, frame, landmarks):
    height, width = frame.shape[:2]
    if len(landmarks) != 21:
        return
    points = []
    for point in landmarks:
        if not math.isfinite(point.x) or not math.isfinite(point.y):
            return
        x = int(np.clip(point.x, 0, 1) * (width - 1))
        y = int(np.clip(point.y, 0, 1) * (height - 1))
        points.append((x, y))
    for start, end in CONNECTIONS:
        cv2.line(frame, points[start], points[end], (40, 220, 40), 2)
    for point in points:
        cv2.circle(frame, point, 4, (40, 80, 255), -1)

