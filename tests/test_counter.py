"""Тесты счетчика на выдуманных траекториях.

Каждый тест это случай, который реально ломал подсчет на видео.
Модель и камера не нужны, все прогоняется за доли секунды.
"""

import pytest

from vehicle_counting.counter import LineCounter


@pytest.fixture
def counter():
    return LineCounter(line_x=100, min_travel=5, forget_after=30)


def drive(counter, track_id, xs, label="car"):
    """Провести машину по заданным x, вернуть список проездов."""
    events = []
    for frame, x in enumerate(xs):
        direction = counter.update(track_id, x, label, frame)
        if direction:
            events.append(direction)
    return events


def test_sleva_napravo(counter):
    assert drive(counter, 1, [40, 60, 80, 120, 140]) == ["right"]
    assert counter.right == 1
    assert counter.left == 0


def test_sprava_nalevo(counter):
    assert drive(counter, 1, [160, 140, 120, 80, 60]) == ["left"]


def test_odna_mashina_schitaetsya_odin_raz(counter):
    """Машина проехала, развернулась и уехала обратно."""
    assert drive(counter, 1, [40, 60, 140, 160, 140, 60, 40]) == ["right"]


def test_drozhanie_na_linii_ne_schitaetsya(counter):
    """Машина стоит на линии в пробке, рамка дрожит на пиксель.

    Без мертвой зоны это давало десятки ложных проездов за одну
    красную фазу светофора.
    """
    assert drive(counter, 1, [99, 101, 100, 102, 98, 101, 99]) == []
    assert counter.total == 0


def test_podehal_i_uehal_nazad(counter):
    """Подъехал к линии, не пересек, уехал обратно."""
    assert drive(counter, 1, [40, 70, 90, 96, 90, 70, 40]) == []


def test_dve_mashiny_nezavisimo(counter):
    drive(counter, 1, [40, 80, 140, 180])
    drive(counter, 2, [180, 140, 80, 40])
    assert counter.left == 1
    assert counter.right == 1
    assert counter.total == 2


def test_schet_po_klassam(counter):
    drive(counter, 1, [40, 140], label="car")
    drive(counter, 2, [40, 140], label="car")
    drive(counter, 3, [40, 140], label="bus")
    assert counter.by_class["right"] == {"car": 2, "bus": 1}


def test_starye_treki_zabyvayutsya(counter):
    """Иначе словари растут все время работы."""
    counter.update(1, 40, "car", frame=0)
    counter.forget_stale(frame=100)
    assert counter.last_seen == {}
    assert counter.side == {}


def test_trekker_pereispolzoval_id(counter):
    """После долгой паузы трекер выдает тот же id новой машине.

    Это не та же самая машина, ее надо посчитать заново.
    """
    drive(counter, 7, [40, 140])
    counter.forget_stale(frame=1000)
    assert counter.update(7, 40, "car", 1001) is None
    assert counter.update(7, 140, "car", 1002) == "right"
    assert counter.total == 2


def test_side_of(counter):
    assert counter.side_of(40) == "left"
    assert counter.side_of(160) == "right"
    assert counter.side_of(100) is None
    assert counter.side_of(102) is None
