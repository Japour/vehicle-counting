#!/usr/bin/env python3
"""Упаковка YOLO-разметки в формат CVAT 1.1 для загрузки на проверку.

CVAT принимает разметку архивом со строго определенной структурой:

    obj.names            имена классов, по одному в строке
    obj.data             classes / names / train
    train.txt            список изображений, пути относительно корня
    obj_train_data/*.txt разметка в формате YOLO

Порядок имен в obj.names задает числовые id классов и обязан совпадать с
data.yaml. Разошлись - CVAT покажет автобусы трамваями, и ошибка вылезет
только глазами разметчика.

Пример:
    python scripts/export_cvat.py data/labels build/cvat --names car bus tram --zip
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def export(labels: Path, output: Path, names: list[str], image_suffix: str):
    data_dir = output / "obj_train_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(labels.glob("*.txt"))
    if not label_files:
        raise SystemExit(f"в {labels} нет .txt разметки")

    for label in label_files:
        shutil.copy2(label, data_dir / label.name)

    (output / "obj.names").write_text("\n".join(names) + "\n", encoding="utf-8")
    (output / "obj.data").write_text(
        f"classes = {len(names)}\nnames = obj.names\ntrain = train.txt\n",
        encoding="utf-8",
    )
    (output / "train.txt").write_text(
        "".join(f"obj_train_data/{p.stem}{image_suffix}\n" for p in label_files),
        encoding="utf-8",
    )
    print(f"подготовлено {len(label_files)} файлов в {output}")
    return len(label_files)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("labels", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--names", nargs="+", default=["car", "bus", "tram"])
    parser.add_argument("--image-suffix", default=".jpg")
    parser.add_argument("--zip", action="store_true", help="дополнительно собрать архив")
    args = parser.parse_args()

    export(args.labels, args.output, args.names, args.image_suffix)
    if args.zip:
        archive = shutil.make_archive(str(args.output), "zip", root_dir=args.output)
        print(f"архив: {archive}")


if __name__ == "__main__":
    main()
