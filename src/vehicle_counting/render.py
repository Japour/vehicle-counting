"""Отрисовка рамок, линии и счетчиков поверх кадра.

Вынесено отдельно, потому что в проде на Jetson окно не открывают и
ничего не рисуют: это лишние миллисекунды на каждый кадр.
"""

import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Цвета BGR, чтобы различались на ночном кадре
COLORS = {
    "car": (219, 152, 52),
    "bus": (113, 204, 46),
    "tram": (182, 89, 155),
}


def draw_box(frame, x1, y1, x2, y2, label, track_id):
    color = COLORS.get(label, (200, 200, 200))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"{label} {track_id}"
    (w, h), _ = cv2.getTextSize(text, FONT, 0.6, 2)
    # если рамка у верхнего края, подпись уедет за кадр - кладем внутрь
    top = y1 - h - 8 if y1 - h - 8 > 0 else y1 + 2
    cv2.rectangle(frame, (x1, top), (x1 + w + 8, top + h + 8), color, -1)
    cv2.putText(frame, text, (x1 + 4, top + h + 2), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def draw_line(frame, line_x):
    cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (0, 255, 255), 2)


def draw_panel(frame, counter):
    rows = [
        (f"left  {counter.left}", (255, 120, 120)),
        (f"right {counter.right}", (120, 255, 120)),
        (f"total {counter.total}", (220, 220, 220)),
    ]
    # ширина панели по тексту, иначе справа остается пустая черная плашка
    pad, step = 14, 30
    text_w = max(cv2.getTextSize(t, FONT, 0.7, 2)[0][0] for t, _ in rows)
    panel_w = text_w + pad * 2
    panel_h = step * len(rows) + pad

    h, w = frame.shape[:2]
    x, y = w - panel_w - 20, h - panel_h - 20

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    for i, (text, color) in enumerate(rows):
        pos = (x + pad, y + pad + step * i + 12)
        cv2.putText(frame, text, pos, FONT, 0.7, color, 2, cv2.LINE_AA)
