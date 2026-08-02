# Развертывание на Jetson Orin Nano

Цель второго этапа - не «запустить на другом компьютере», а перенести
систему на устройство, которое стоит у дороги само: 8 ГБ памяти, ~15 Вт,
без монитора и без меня рядом.

Платформа: Jetson Orin Nano 8 GB, JetPack 6.1, CSI-камера через
`nvarguscamerasrc`.

## Зависимости

```bash
sudo apt update && sudo apt install -y python3-pip
pip3 install ultralytics opencv-python
```

Проверка, что CUDA видна из PyTorch:

```bash
python3 -c "import torch; print(torch.cuda.is_available())"   # ожидается True
```

### Грабли: torchvision сносит PyTorch от NVIDIA

На Jetson нельзя ставить `torch`/`torchvision` из PyPI: там сборки под
x86 и обычную CUDA, а на Jetson нужны сборки NVIDIA под aarch64 и L4T.
`pip3 install torchvision` тянет за собой обычный `torch` и молча заменяет
правильный - после этого `cuda.is_available()` возвращает `False`, а
ultralytics уходит считать на CPU.

Лечится переустановкой в правильном порядке, колесами с сайта NVIDIA:

```bash
pip3 uninstall torch torchvision -y
pip3 install --no-cache-dir <torch-*-linux_aarch64.whl>
pip3 install --no-cache-dir <torchvision-*-linux_aarch64.whl>
```

Версии колес должны соответствовать установленному JetPack - брать с
`developer.download.nvidia.com/compute/redist/jp/`.

## Экспорт в TensorRT

PyTorch-веса на Jetson дают 5-8 FPS - для потока по четырем полосам мало.
TensorRT в FP16 ускоряет в 2-3 раза, точность при этом падает в пределах
погрешности.

```bash
yolo export model=best.pt format=engine device=0 half=True
```

Занимает 5-15 минут: TensorRT подбирает ядра под конкретный чип. Поэтому
`.engine` не переносится между устройствами и не хранится в git:
собирается на той машине, где будет работать.

Проверка:

```bash
yolo predict model=best.engine source=0 device=0
```

## Запуск

```bash
python -m vehicle_counting --weights models/best.engine --source 0 --device 0 --save-csv data/crossings.csv
```

Окно и запись видео в поле не нужны: смотреть некому, а отрисовка и
энкодинг съедают кадры, которых на 15 Вт и так немного. Поэтому без
`--show` и `--save-video`, единственный выход это CSV с проездами.

## Ожидаемая производительность

| Вариант | FPS |
|---|---|
| `best.pt`, PyTorch | 5-8 |
| `best.engine`, TensorRT FP16 | 15-25 |

Это оценка по документации NVIDIA и замерам на схожих задачах. **Свой замер
на этом железе еще не сделан** - как только будет, цифры здесь заменятся на
фактические.

## Автозапуск

Сбор кадров и счетчик держатся systemd-юнитами: устройство должно
подниматься само после отключения питания, без ручного запуска.

```ini
# /etc/systemd/system/vehicle-counter.service
[Unit]
Description=Vehicle counter
After=network.target

[Service]
Type=simple
User=jetson
WorkingDirectory=/home/jetson/vehicle-counting
ExecStart=/usr/bin/python3 -m vehicle_counting --weights models/best.engine --source 0 --device 0 --save-csv data/crossings.csv
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now vehicle-counter
journalctl -u vehicle-counter -f
```

## Статус

- [x] сбор датасета с CSI-камеры работает, ~11.7 тыс. кадров собрано
- [x] окружение поднято, CUDA доступна
- [ ] экспорт в TensorRT и замер реального FPS
- [ ] прогон счетчика на живой камере сутки подряд
