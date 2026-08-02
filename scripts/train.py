#!/usr/bin/env python3
"""Дообучение YOLO11 на своем датасете.

Тонкая обертка над ultralytics: сам по себе train - одна строка, но здесь
она зафиксирована вместе с гиперпараметрами, на которых получены метрики
из docs/results.md. Смысл файла в воспроизводимости, а не в коде.

workers=0 по умолчанию - наследство Windows-машины с RTX 4070 Ti, на
которой шло обучение: там многопоточная загрузка данных требует запуска
через `if __name__ == "__main__"` и все равно регулярно виснет. На Linux
можно смело ставить 8.

Пример:
    python scripts/train.py --data data/dataset/data.yaml --model yolo11l.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="путь к data.yaml")
    parser.add_argument("--model", default="yolo11l.pt", help="стартовые веса COCO")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16, help="16 влезает в 12 ГБ VRAM")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--patience", type=int, default=30, help="ранняя остановка")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/vehicle_counting")
    parser.add_argument("--name", default="exp")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
    )

    # Валидация отдельным вызовом: train печатает метрики последней эпохи,
    # а в README нужны метрики лучших весов (best.pt), это разные числа.
    metrics = model.val()
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
