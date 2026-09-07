from dataclasses import dataclass
from datetime import datetime


@dataclass
class Tank:
    name: str
    capacity: float
    minimum: float = 0.0
    initial_level: float | None = None


@dataclass
class Collection:
    time: datetime
    target_tonnes: float
    loading_rate: float
    arrived: bool = True
    missed_reason: str | None = None


@dataclass
class ProductionPeriod:
    start: datetime
    end: datetime
    rate: float
    source: str = "manual"


@dataclass
class SimulationConfiguration:
    name: str
    tank_capacities: list[float]
    tank_minimums: list[float]
    loading_rate: float
    max_parallel_loads: int = 1
    collections_per_day: int = 2
