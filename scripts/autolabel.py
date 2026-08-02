#!/usr/bin/env python3
"""Предразметка кадров обученной моделью.

На первом датасете (2333 кадра) предразметку делала стоковая YOLO11n на
классах COCO - и разметка была плохой: чужие классы, лишние объекты,
пропуски трамваев. Вторая итерация исходит из другого: предразмечать
кадры своей же моделью с первого этапа. Она уже знает ровно три нужных
класса и этот ракурс, поэтому мне остается только проверять
с нуля.

Это не «разметка бесплатно». Модель уверенно повторяет собственные
ошибки, и без ручной проверки в CVAT датасет закрепляет их вместо того,
чтобы исправлять. Порог здесь намеренно ниже боевого (0.25 против 0.35):
лишнюю рамку удалить одним кликом дешевле, чем заметить пропущенную.

Пример:
    python scripts/autolabel.py models/best.pt data/frames data/labels
"""

from __future__ import annotations

import argparse
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def autolabel(
    weights: Path, images: Path, output: Path, confidence: float, device: str | None
):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    output.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise SystemExit(f"в {images} нет изображений")

    done = 0
    failed = 0
    for path in files:
        try:
            result = model.predict(
                str(path), conf=confidence, device=device, verbose=False
            )[0]
        except Exception as error:  # битый JPEG среди десятков тысяч - норма
            print(f"пропущен {path.name}: {error}")
            failed += 1
            continue

        # Пустой файл для кадра без машин нужен обязательно: YOLO считает
        # такой кадр негативным примером. Если файла нет, кадр молча
        # выпадает из обучения, и модель не учится на пустой дороге.
        lines = []
        for box in result.boxes:
            class_id = int(box.cls)
            x, y, w, h = box.xywhn[0].tolist()
            lines.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        (output / f"{path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        done += 1

        if done % 500 == 0:
            print(f"{done}/{len(files)}")

    print(f"размечено: {done}, пропущено: {failed}")
    print("дальше - ручная проверка в CVAT, без нее датасет не готов")
    return done, failed


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("weights", type=Path)
    parser.add_argument("images", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--conf", type=float, default=0.25, dest="confidence")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    autolabel(args.weights, args.images, args.output, args.confidence, args.device)


if __name__ == "__main__":
    main()
