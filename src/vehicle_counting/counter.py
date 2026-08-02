"""Подсчет машин, пересекающих вертикальную линию.

Здесь нет ни OpenCV, ни YOLO: на вход идут номер трека и координата x,
на выход - факт проезда. Поэтому логику можно проверить тестами, не
запуская видео.
"""


class LineCounter:
    """Считает проезды через вертикальную линию, один раз на трек.

    min_travel - мертвая зона у линии. Без нее машина, вставшая на линии
    в пробке, дает десятки проездов: рамка детектора дрожит на пиксель.

    forget_after - через сколько кадров забыть пропавший трек. Иначе
    словари растут все время работы.
    """

    def __init__(self, line_x, min_travel=6, forget_after=90):
        self.line_x = line_x
        self.min_travel = min_travel
        self.forget_after = forget_after

        self.side = {}        # для каждого трека: "left" или "right"
        self.last_seen = {}   # для каждого трека: номер последнего кадра
        self.counted = set()  # треки, которые уже посчитали

        self.left = 0
        self.right = 0
        self.by_class = {"left": {}, "right": {}}

    def side_of(self, x):
        """С какой стороны линии точка. None - слишком близко к линии."""
        if abs(x - self.line_x) < self.min_travel:
            return None
        return "left" if x < self.line_x else "right"

    def update(self, track_id, x, label, frame):
        """Обновить трек по новому кадру.

        Возвращает "left" или "right", если машина только что пересекла
        линию, иначе None.
        """
        self.last_seen[track_id] = frame
        side = self.side_of(x)
        if side is None:
            return None

        before = self.side.get(track_id)
        self.side[track_id] = side

        # проезд = смена стороны у трека, которого еще не считали
        if before is None or before == side or track_id in self.counted:
            return None

        self.counted.add(track_id)
        if side == "left":
            self.left += 1
        else:
            self.right += 1
        counts = self.by_class[side]
        counts[label] = counts.get(label, 0) + 1
        return side

    def forget_stale(self, frame):
        """Выбросить треки, которых давно не видно. Звать раз в кадр."""
        old = [t for t, seen in self.last_seen.items() if frame - seen > self.forget_after]
        for t in old:
            del self.last_seen[t]
            self.side.pop(t, None)
            self.counted.discard(t)

    @property
    def total(self):
        return self.left + self.right
