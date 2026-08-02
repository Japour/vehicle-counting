"""Точка входа: читает видео, детектирует, считает, печатает итог.

Запуск:
    python -m vehicle_counting --source road.mp4 --weights models/best.pt
"""

import argparse
import csv

import cv2

from .counter import LineCounter
from .render import draw_box, draw_line, draw_panel


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Подсчет транспорта в видеопотоке")
    p.add_argument("--weights", default="models/best.pt", help="путь к .pt или .engine")
    p.add_argument("--source", default="0", help="номер камеры или путь к видео")
    p.add_argument("--device", default=None, help="cpu, 0, mps")
    p.add_argument("--conf", type=float, default=0.35, help="порог уверенности")
    p.add_argument("--line", type=float, default=0.5, help="линия, доля ширины кадра")
    p.add_argument("--save-video", help="куда писать видео с разметкой")
    p.add_argument("--save-csv", help="куда писать список проездов")
    p.add_argument("--show", action="store_true", help="показывать окно")
    return p.parse_args(argv)


def open_source(source):
    """Открыть камеру (source - число) или видеофайл (source - путь)."""
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"не удалось открыть источник: {source}")
    return cap


def main(argv=None):
    args = parse_args(argv)

    from ultralytics import YOLO

    model = YOLO(args.weights)
    cap = open_source(args.source)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    line_x = int(width * args.line)
    counter = LineCounter(line_x)
    draw = args.show or args.save_video

    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_video, fourcc, fps, (width, height))

    csv_file = csv_writer = None
    if args.save_csv:
        csv_file = open(args.save_csv, "w", newline="", encoding="utf-8")  # noqa: SIM115
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame", "sec", "track_id", "class", "direction"])

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # persist=True: трекер помнит объекты между кадрами
            results = model.track(
                frame,
                tracker="botsort.yaml",
                conf=args.conf,
                persist=True,
                verbose=False,
                device=args.device,
            )
            boxes = results[0].boxes

            if boxes is not None and boxes.id is not None:
                names = model.names
                for i in range(len(boxes.id)):
                    x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[i])
                    track_id = int(boxes.id[i])
                    label = names.get(int(boxes.cls[i]), "object")

                    center_x = (x1 + x2) // 2
                    direction = counter.update(track_id, center_x, label, frame_index)
                    if direction and csv_writer:
                        csv_writer.writerow(
                            [frame_index, round(frame_index / fps, 2), track_id, label, direction]
                        )
                    if draw:
                        draw_box(frame, x1, y1, x2, y2, label, track_id)

            counter.forget_stale(frame_index)

            if draw:
                draw_line(frame, line_x)
                draw_panel(frame, counter)
            if writer:
                writer.write(frame)
            if args.show:
                cv2.imshow("vehicle-counting", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
    finally:
        cap.release()
        if writer:
            writer.release()
        if csv_file:
            csv_file.close()
        if args.show:
            cv2.destroyAllWindows()

    print(f"кадров обработано: {frame_index}")
    print(f"влево:  {counter.left}")
    print(f"вправо: {counter.right}")
    print(f"всего:  {counter.total}")
    for direction in ("left", "right"):
        by_class = counter.by_class[direction]
        if by_class:
            parts = ", ".join(f"{name} {n}" for name, n in sorted(by_class.items()))
            print(f"  {direction}: {parts}")
    return 0
