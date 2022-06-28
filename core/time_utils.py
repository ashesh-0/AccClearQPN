from datetime import datetime, timedelta

from core.constants import TIME_GRANULARITY_MIN


class TimeSteps:
    @staticmethod
    def nth_next(dt: datetime, n: int):
        return dt + timedelta(minutes=TIME_GRANULARITY_MIN * n)

    @staticmethod
    def nth_previous(dt: datetime, n: int):
        return dt - timedelta(minutes=TIME_GRANULARITY_MIN * n)

    @staticmethod
    def next(dt: datetime):
        return TimeSteps.nth_next(dt, 1)

    @staticmethod
    def previous(dt: datetime):
        return TimeSteps.nth_previous(dt, 1)
