#!/usr/bin/env python3
"""Разбиение размеченных кадров на train/val в структуре Ultralytics.

Две вещи, из-за которых этот скрипт существует отдельно, а не «руками».

Seed фиксирован. Без него каждый перезапуск дает другое разбиение, и
сравнивать метрики двух обучений бессмысленно: часть валидации прошлого
прогона оказалась в обучении текущего.

Разбиение по времени (--split-by time) вместо случайного. Кадры нарезаны
из одного непрерывного видео, поэтому соседние кадры почти одинаковы.
При случайном разбиении кадр N попадает в train, а кадр N+1 - в val, и
модель валидируется на том, что уже видела: метрики завышены. Честнее
отдать в val последний по времени кусок записи. Случайный режим оставлен
как есть, потому что именно на нем получены метрики первого этапа: цифры
из README и код должны сходиться.

Пример:
    python scripts/split_dataset.py data/frames data/labels data/dataset --split-by time
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def collect_pairs(images: Path, labels: Path):
    """Пары «изображение + разметка». Кадры без разметки отбрасываются."""
    pairs = []
    missing = 0
    for image in sorted(p for p in images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        label = labels / f"{image.stem}.txt"
        if label.exists():
            pairs.append((image, label))
        else:
            missing += 1
    if missing:
        print(f"без разметки, пропущено: {missing}")
    return pairs


def split(
    pairs: list[tuple[Path, Path]], val_fraction: float, by: str, seed: int
):
    if by == "random":
        pairs = list(pairs)
        random.Random(seed).shuffle(pairs)
    train_size = int(len(pairs) * (1 - val_fraction))
    return {"train": pairs[:train_size], "val": pairs[train_size:]}


def write_dataset(
    splits: dict[str, list[tuple[Path, Path]]], output: Path, names: list[str]
):
    for name, pairs in splits.items():
        (output / "images" / name).mkdir(parents=True, exist_ok=True)
        (output / "labels" / name).mkdir(parents=True, exist_ok=True)
        for image, label in pairs:
            shutil.copy2(image, output / "images" / name / image.name)
            shutil.copy2(label, output / "labels" / name / label.name)
        print(f"{name}: {len(pairs)}")

    data_yaml = output / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {output.resolve()}",
                "train: images/train",
                "val: images/val",
                f"nc: {len(names)}",
                "names:",
                *[f"  - {name}" for name in names],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"конфиг: {data_yaml}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("images", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-by", choices=["random", "time"], default="time")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--names", nargs="+", default=["car", "bus", "tram"])
    args = parser.parse_args()

    pairs = collect_pairs(args.images, args.labels)
    if not pairs:
        raise SystemExit("не найдено ни одной пары изображение+разметка")
    splits = split(pairs, args.val_fraction, args.split_by, args.seed)
    write_dataset(splits, args.output, args.names)


if __name__ == "__main__":
    main()
