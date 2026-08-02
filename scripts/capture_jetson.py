#!/usr/bin/env python3
"""Сбор кадров с CSI-камеры Jetson (этап 2 датасета).

Задача этапа - уйти от датасета «одна ночь с балкона на телефон» к съемке
с той камеры и того ракурса, на которых система будет работать в проде.
Камера пишет сама, сутками.

Почему GStreamer, а не cv2.VideoCapture(0): CSI-камера на Jetson идет
через nvarguscamerasrc - ISP на видеоядре, а не через V4L2. Без этого
конвейера камера просто не открывается.

Скрипт рассчитан на многосуточный прогон, поэтому:
  * нумерация продолжается с последнего файла в папке - после ребута
    и перезапуска сервиса кадры не затираются;
  * подряд идущие ошибки чтения ограничены, чтобы отвалившаяся камера
    не крутила пустой цикл вечно;
  * интервал считается от начала итерации, а не после записи, иначе
    период плывет на время сохранения JPEG.

Запускать на самом Jetson, обычно как systemd-юнит.
"""

from __future__ import annotations

import argparse
import re
import signal
import sys
import time
from pathlib import Path

import cv2

GSTREAMER_PIPELINE = (
    "nvarguscamerasrc sensor-id={sensor} ee-mode=0 tnr-mode=0 tnr-strength=0 ! "
    "video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1 ! "
    "nvvidconv ! video/x-raw, format=(string)BGRx ! "
    "videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=true sync=false"
)

_FRAME_PATTERN = re.compile(r"frame_(\d+)\.jpg$")
_running = True


def _stop(signum, frame):  # noqa: ARG001 - сигнатура задана signal
    global _running
    _running = False


def last_index(directory: Path):
    """Наибольший номер уже сохраненного кадра, или 0 если папка пуста."""
    indices = [
        int(match.group(1))
        for path in directory.glob("frame_*.jpg")
        if (match := _FRAME_PATTERN.search(path.name))
    ]
    return max(indices, default=0)


def capture(
    output: Path,
    interval: float,
    quality: int,
    sensor: int,
    width: int,
    height: int,
    fps: int,
    max_errors: int,
):
    output.mkdir(parents=True, exist_ok=True)
    pipeline = GSTREAMER_PIPELINE.format(
        sensor=sensor, width=width, height=height, fps=fps
    )
    camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not camera.isOpened():
        print("камера недоступна: проверь CSI-overlay и nvargus-daemon", file=sys.stderr)
        return 1

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    saved = last_index(output)
    session = 0
    errors = 0
    print(f"запись в {output} каждые {interval} с, продолжаю с {saved + 1}")

    try:
        while _running:
            started = time.monotonic()
            ok, frame = camera.read()
            if ok:
                errors = 0
                saved += 1
                session += 1
                cv2.imwrite(
                    str(output / f"frame_{saved:05d}.jpg"),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, quality],
                )
            else:
                errors += 1
                print(f"кадр не прочитан ({errors}/{max_errors})", file=sys.stderr)
                if errors >= max_errors:
                    print("камера не отвечает, останавливаюсь", file=sys.stderr)
                    return 1
            time.sleep(max(0.0, interval - (time.monotonic() - started)))
    finally:
        camera.release()
        print(f"сохранено за сессию: {session}, всего в папке: {saved}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("--interval", type=float, default=5.0, help="секунд между кадрами")
    parser.add_argument("--quality", type=int, default=85)
    parser.add_argument("--sensor", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-errors", type=int, default=10)
    args = parser.parse_args()
    return capture(
        args.output,
        args.interval,
        args.quality,
        args.sensor,
        args.width,
        args.height,
        args.fps,
        args.max_errors,
    )


if __name__ == "__main__":
    raise SystemExit(main())
