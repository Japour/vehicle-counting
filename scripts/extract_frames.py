#!/usr/bin/env python3
"""Нарезка видео на кадры для разметки.

Этап 1 конвейера данных: из 43 минут видео 1920×1080 получилось 2600 кадров.

Интервал важнее, чем кажется. При 25 fps соседние кадры почти одинаковы:
машина смещается на несколько пикселей. Такие кадры не добавляют модели
информации, но раздувают датасет и попадают одновременно в train и val:
модель «запоминает» валидацию и метрики врут. Один кадр в секунду это
компромисс, при котором объекты успевают заметно сместиться.

Пример:
    python scripts/extract_frames.py road.MOV data/frames --every 1.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def extract(video: Path, output: Path, every_seconds: float, quality: int):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"не удалось открыть видео: {video}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(fps * every_seconds))
    output.mkdir(parents=True, exist_ok=True)

    saved = 0
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % step == 0:
            path = output / f"frame_{saved:05d}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            saved += 1
        index += 1

    capture.release()
    print(f"{video.name}: {index} кадров прочитано, {saved} сохранено (каждый {step}-й)")
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--every", type=float, default=1.0, help="интервал в секундах")
    parser.add_argument("--quality", type=int, default=90, help="качество JPEG, 1..100")
    args = parser.parse_args()
    extract(args.video, args.output, args.every, args.quality)


if __name__ == "__main__":
    main()
