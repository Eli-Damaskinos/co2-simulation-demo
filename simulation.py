from __future__ import annotations

from dataclasses import replace
from datetime import time as dt_time
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from models import (
    Collection,
    ProductionPeriod,
    SimulationConfiguration,
    Tank,
)


STRESS_SCENARIOS = (
    "Normal schedule",
    "Four-day no deliveries",
    "Back-to-back pair",
    "Two-hour interval",
    "24-hour delivery gap",
    "Four-day gap + back-to-back recovery",
    "24-hour plant outage",
    "Monte Carlo disruptions",
)


def _get_production_rate(
    timestamp: pd.Timestamp,
    default_rate: float,
    production_periods: Sequence[ProductionPeriod],
) -> float:
    """Return the lowest active override rate, or the default rate."""
    active_rates = [float(default_rate)]

    for period in production_periods:
        if pd.Timestamp(period.start) <= timestamp < pd.Timestamp(period.end):
            active_rates.append(float(period.rate))

    return max(0.0, min(active_rates))


def _prepare_collection_lookup(
    collections: Sequence[Collection],
    step_minutes: int,
) -> dict[pd.Timestamp, list[Collection]]:
    """Snap arrivals up to the next timestep so a trailer never arrives early."""
    lookup: dict[pd.Timestamp, list[Collection]] = {}

    for collection in collections:
        timestamp = pd.Timestamp(collection.time).ceil(f"{step_minutes}min")
        lookup.setdefault(timestamp, []).append(collection)

    return lookup


def _as_clock_time(value: str | int | dt_time) -> dt_time:
    if isinstance(value, dt_time):
        return value
    if isinstance(value, int):
        return dt_time(hour=value)

    parsed = pd.Timestamp(f"2000-01-01 {value}")
    return dt_time(hour=parsed.hour, minute=parsed.minute)


def evenly_spaced_collection_times(
    collections_per_day: int,
    first_hour: int = 8,
    last_hour: int = 18,
) -> tuple[str, ...]:
    """Create editable daytime collection times for one or more trailers."""
    if collections_per_day < 1:
        raise ValueError("collections_per_day must be at least 1")

    if collections_per_day == 1:
        minutes = [first_hour * 60]
    else:
        minutes = np.linspace(
            first_hour * 60,
            last_hour * 60,
            collections_per_day,
        ).round().astype(int)

    return tuple(f"{minute // 60:02d}:{minute % 60:02d}" for minute in minutes)


def generate_default_collections(
    simulation_start: str | pd.Timestamp,
    simulation_days: int,
    target_tonnes: float = 19.0,
    loading_rate: float = 8.0,
    collection_times: Sequence[str | int | dt_time] = ("08:00", "18:00"),
    collection_hours: Sequence[int] | None = None,
) -> list[Collection]:
    """Generate a recurring collection schedule across the simulation."""
    if simulation_days < 1:
        return []

    if collection_hours is not None:
        collection_times = tuple(collection_hours)

    clocks = [_as_clock_time(value) for value in collection_times]
    start = pd.Timestamp(simulation_start).normalize()
    collections: list[Collection] = []

    for day_offset in range(simulation_days):
        day = start + pd.Timedelta(days=day_offset)
        for clock in clocks:
            collections.append(
                Collection(
                    time=(
                        day
                        + pd.Timedelta(hours=clock.hour, minutes=clock.minute)
                    ).to_pydatetime(),
                    target_tonnes=float(target_tonnes),
                    loading_rate=float(loading_rate),
                    arrived=True,
                )
            )

    return sorted(collections, key=lambda collection: pd.Timestamp(collection.time))


def generate_random_downtime_periods(
    simulation_start: str | pd.Timestamp,
    simulation_days: int,
    downtime_percentage: float,
    seed: int = 42,
    minimum_duration_hours: float = 2.0,
    maximum_duration_hours: float = 18.0,
    step_minutes: int = 10,
) -> list[ProductionPeriod]:
    """Generate reproducible blocks covering approximately the requested downtime."""
    if not 0 <= downtime_percentage <= 100:
        raise ValueError("downtime_percentage must be between 0 and 100")
    if minimum_duration_hours <= 0 or maximum_duration_hours < minimum_duration_hours:
        raise ValueError("Invalid downtime duration range")

    steps_per_day = 24 * 60 // step_minutes
    total_steps = simulation_days * steps_per_day
    target_steps = int(round(total_steps * downtime_percentage / 100.0))

    if target_steps == 0:
        return []

    start = pd.Timestamp(simulation_start)
    if target_steps >= total_steps:
        return [
            ProductionPeriod(
                start=start.to_pydatetime(),
                end=(start + pd.Timedelta(minutes=total_steps * step_minutes)).to_pydatetime(),
                rate=0.0,
                source="random downtime",
            )
        ]

    rng = np.random.default_rng(seed)
    downtime = np.zeros(total_steps, dtype=bool)
    minimum_steps = max(1, int(round(minimum_duration_hours * 60 / step_minutes)))
    maximum_steps = max(minimum_steps, int(round(maximum_duration_hours * 60 / step_minutes)))
    attempts = 0
    maximum_attempts = max(2_000, target_steps * 40)

    while int(downtime.sum()) < target_steps and attempts < maximum_attempts:
        remaining = target_steps - int(downtime.sum())
        duration = int(rng.integers(minimum_steps, maximum_steps + 1))
        duration = min(duration, remaining, total_steps)
        block_start = int(rng.integers(0, total_steps - duration + 1))
        downtime[block_start : block_start + duration] = True
        attempts += 1

    remaining = target_steps - int(downtime.sum())
    if remaining > 0:
        available = np.flatnonzero(~downtime)
        chosen = rng.choice(available, size=remaining, replace=False)
        downtime[chosen] = True

    change_points = np.flatnonzero(np.diff(np.r_[False, downtime, False]))
    periods: list[ProductionPeriod] = []

    for block_start, block_end in change_points.reshape(-1, 2):
        periods.append(
            ProductionPeriod(
                start=(start + pd.Timedelta(minutes=int(block_start) * step_minutes)).to_pydatetime(),
                end=(start + pd.Timedelta(minutes=int(block_end) * step_minutes)).to_pydatetime(),
                rate=0.0,
                source="random downtime",
            )
        )

    return periods


def apply_collection_disruptions(
    collections: Sequence[Collection],
    no_delivery_periods: Iterable[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
    random_missed_percentage: float = 0.0,
    seed: int = 42,
) -> list[Collection]:
    """Apply manual no-delivery windows and reproducible additional random misses."""
    if not 0 <= random_missed_percentage <= 100:
        raise ValueError("random_missed_percentage must be between 0 and 100")

    periods = [
        (pd.Timestamp(start), pd.Timestamp(end))
        for start, end in (no_delivery_periods or [])
    ]
    disrupted = [replace(collection) for collection in collections]

    for index, collection in enumerate(disrupted):
        timestamp = pd.Timestamp(collection.time)
        if not collection.arrived:
            disrupted[index] = replace(
                collection,
                missed_reason=collection.missed_reason or "manually missed",
            )
        elif any(start <= timestamp < end for start, end in periods):
            disrupted[index] = replace(
                collection,
                arrived=False,
                missed_reason="manual no-delivery period",
            )

    eligible = [index for index, item in enumerate(disrupted) if item.arrived]
    random_count = int(round(len(eligible) * random_missed_percentage / 100.0))

    if random_count:
        rng = np.random.default_rng(seed)
        missed_indices = rng.choice(eligible, size=random_count, replace=False)
        for index in missed_indices:
            disrupted[index] = replace(
                disrupted[index],
                arrived=False,
                missed_reason="random miss",
            )

    return sorted(disrupted, key=lambda collection: pd.Timestamp(collection.time))


def apply_named_stress_scenario(
    collections: Sequence[Collection],
    scenario: str,
    simulation_start: str | pd.Timestamp,
    simulation_days: int,
) -> list[Collection]:
    """Transform a normal schedule into one named operational stress case."""
    if scenario not in STRESS_SCENARIOS:
        raise ValueError(f"Unknown stress scenario: {scenario}")

    stressed = [replace(collection) for collection in collections]
    start = pd.Timestamp(simulation_start).normalize()
    stress_offset = min(max(1, simulation_days // 3), max(1, simulation_days - 1))
    stress_day = start + pd.Timedelta(days=stress_offset)

    if scenario in {"Four-day no deliveries", "Four-day gap + back-to-back recovery"}:
        gap_end = min(
            start + pd.Timedelta(days=simulation_days),
            stress_day + pd.Timedelta(days=4),
        )
        stressed = apply_collection_disruptions(
            stressed,
            no_delivery_periods=[(stress_day, gap_end)],
        )

        if scenario == "Four-day gap + back-to-back recovery":
            recovery_day = gap_end.normalize()
            recovery_indices = [
                index
                for index, collection in enumerate(stressed)
                if pd.Timestamp(collection.time).normalize() == recovery_day
                and collection.arrived
            ][:2]
            for index in recovery_indices:
                stressed[index] = replace(
                    stressed[index],
                    time=(recovery_day + pd.Timedelta(hours=8)).to_pydatetime(),
                )

    elif scenario == "24-hour delivery gap":
        stressed = apply_collection_disruptions(
            stressed,
            no_delivery_periods=[(stress_day, stress_day + pd.Timedelta(hours=24))],
        )

    elif scenario in {"Back-to-back pair", "Two-hour interval"}:
        indices = [
            index
            for index, collection in enumerate(stressed)
            if pd.Timestamp(collection.time).normalize() == stress_day
        ][:2]
        offsets = (8, 8) if scenario == "Back-to-back pair" else (8, 10)
        for index, hour in zip(indices, offsets):
            stressed[index] = replace(
                stressed[index],
                time=(stress_day + pd.Timedelta(hours=hour)).to_pydatetime(),
            )

    return sorted(stressed, key=lambda collection: pd.Timestamp(collection.time))


def run_simulation(
    tanks: Sequence[Tank],
    collections: Sequence[Collection],
    production_periods: Sequence[ProductionPeriod] | None = None,
    default_fill_rate: float = 1.6,
    testing_required_hours: float = 2.0,
    simulation_start: str | pd.Timestamp = "2024-01-01 00:00",
    simulation_days: int = 30,
    step_minutes: int = 10,
    max_parallel_loads: int = 1,
    random_missed_percentage: float = 0.0,
    random_seed: int = 42,
):
    """Run the simulation, calling a trailer only for one ready full-load tank."""
    if not tanks:
        raise ValueError("At least one tank is required")
    if simulation_days < 1 or step_minutes < 1:
        raise ValueError("Simulation length and timestep must be positive")
    if max_parallel_loads < 1:
        raise ValueError("max_parallel_loads must be at least 1")
    if not 0 <= random_missed_percentage <= 100:
        raise ValueError("random_missed_percentage must be between 0 and 100")

    production_periods = list(production_periods or [])
    start = pd.Timestamp(simulation_start)
    end = start + pd.Timedelta(days=simulation_days)
    in_horizon_collections = [
        collection
        for collection in collections
        if start <= pd.Timestamp(collection.time) < end
    ]

    tank_levels: list[float] = []
    for tank in tanks:
        if tank.capacity <= tank.minimum:
            raise ValueError(f"{tank.name}: capacity must exceed minimum")
        initial_level = tank.minimum if tank.initial_level is None else tank.initial_level
        if not tank.minimum <= initial_level <= tank.capacity:
            raise ValueError(f"{tank.name}: initial level must be between minimum and capacity")
        tank_levels.append(float(initial_level))

    step_hours = step_minutes / 60.0
    times = pd.date_range(
        start=start,
        periods=simulation_days * 24 * (60 // step_minutes),
        freq=f"{step_minutes}min",
    )
    num_tanks = len(tanks)
    tank_static_hours = [0.0] * num_tanks
    active_fill_tank = 0
    initial_storage = float(sum(tank_levels))

    collection_lookup = _prepare_collection_lookup(
        in_horizon_collections,
        step_minutes,
    )
    collection_queue: list[dict] = []
    active_collections: list[dict] = []
    all_jobs: list[dict] = []
    completed_jobs: list[dict] = []
    records: list[dict] = []
    maximum_waiting_queue = 0
    maximum_outstanding = 0
    random_generator = np.random.default_rng(random_seed)
    failed_partial_jobs: list[dict] = []

    for timestamp in times:
        arrivals = collection_lookup.get(timestamp, [])
        missed_this_step = 0
        called_this_step = 0
        not_called_this_step = 0

        reserved_tanks = {
            job["tank_index"]
            for job in active_collections + collection_queue
            if job["tank_index"] is not None
        }

        for collection in arrivals:
            target = float(collection.target_tonnes)
            candidates = [
                index
                for index, tank in enumerate(tanks)
                if index not in reserved_tanks
                and tank_levels[index] - tank.minimum >= target - 1e-9
                and tank_static_hours[index] >= testing_required_hours
            ]

            if not candidates:
                not_called_this_step += 1
                continue

            collection_tank = max(
                candidates,
                key=lambda index: tank_levels[index] - tanks[index].minimum,
            )
            called_this_step += 1
            random_miss = (
                collection.arrived
                and random_generator.random() < random_missed_percentage / 100.0
            )

            if not collection.arrived or random_miss:
                missed_this_step += 1
                continue

            job = {
                "job_id": len(all_jobs) + 1,
                "scheduled_time": pd.Timestamp(collection.time),
                "target_tonnes": target,
                "loading_rate": float(collection.loading_rate),
                "collected": 0.0,
                "tank_index": collection_tank,
                "phase": "ready",
                "loading_started_at": None,
                "completed_at": None,
                "status": "queued",
            }
            collection_queue.append(job)
            all_jobs.append(job)
            reserved_tanks.add(collection_tank)

        while collection_queue and len(active_collections) < max_parallel_loads:
            job = collection_queue.pop(0)
            job["status"] = "active"
            job["phase"] = "loading"
            job["loading_started_at"] = timestamp
            active_collections.append(job)

        tank_states = ["available"] * num_tanks
        for job in collection_queue:
            tank_index = job["tank_index"]
            if tank_index is not None:
                tank_states[tank_index] = "reserved"
        for job in active_collections:
            tank_index = job["tank_index"]
            if tank_index is not None:
                tank_states[tank_index] = "collection"

        emptied = [0.0] * num_tanks
        completed_this_step = 0
        still_active: list[dict] = []

        for job in active_collections:
            tank_index = job["tank_index"]
            if tank_index is not None and job["phase"] == "loading":
                remaining_target = max(0.0, job["target_tonnes"] - job["collected"])
                available = max(0.0, tank_levels[tank_index] - tanks[tank_index].minimum)
                amount = min(job["loading_rate"] * step_hours, remaining_target, available)

                if amount > 0:
                    tank_levels[tank_index] -= amount
                    emptied[tank_index] += amount
                    job["collected"] += amount

                if job["collected"] >= job["target_tonnes"] - 1e-9:
                    job["status"] = "completed"
                    job["completed_at"] = timestamp + pd.Timedelta(minutes=step_minutes)
                    completed_jobs.append(job)
                    completed_this_step += 1
                    continue

                if tank_levels[tank_index] <= tanks[tank_index].minimum + 1e-9:
                    job["status"] = "failed_partial"
                    job["completed_at"] = timestamp + pd.Timedelta(minutes=step_minutes)
                    failed_partial_jobs.append(job)
                    continue

            still_active.append(job)

        active_collections = still_active

        can_fill = [
            tank_states[index] == "available"
            and tank_levels[index] < tank.capacity - 1e-9
            for index, tank in enumerate(tanks)
        ]
        if can_fill[active_fill_tank]:
            fill_tank = active_fill_tank
        else:
            fill_tank = None
            for offset in range(1, num_tanks + 1):
                candidate = (active_fill_tank + offset) % num_tanks
                if can_fill[candidate]:
                    fill_tank = candidate
                    active_fill_tank = candidate
                    break

        production_rate = _get_production_rate(
            timestamp,
            default_fill_rate,
            production_periods,
        )
        incoming = production_rate * step_hours
        filled = [0.0] * num_tanks

        if fill_tank is None:
            lost_co2 = incoming
        else:
            available_space = tanks[fill_tank].capacity - tank_levels[fill_tank]
            amount = min(incoming, available_space)
            filled[fill_tank] = amount
            tank_levels[fill_tank] += amount
            lost_co2 = incoming - amount

        for index in range(num_tanks):
            if filled[index] > 0 or emptied[index] > 0:
                tank_static_hours[index] = 0.0
            else:
                tank_static_hours[index] = round(
                    tank_static_hours[index] + step_hours,
                    10,
                )

        waiting_count = len(collection_queue)
        outstanding_count = waiting_count + len(active_collections)
        maximum_waiting_queue = max(maximum_waiting_queue, waiting_count)
        maximum_outstanding = max(maximum_outstanding, outstanding_count)

        record = {
            "time": timestamp,
            "production_rate_tph": production_rate,
            "produced_tonnes": incoming,
            "lost_co2": lost_co2,
            "filling_tank": tanks[fill_tank].name if fill_tank is not None else None,
            "collection_active_count": len(active_collections),
            "collection_queue": waiting_count,
            "outstanding_collections": outstanding_count,
            "called_collections": called_this_step,
            "not_called_insufficient_inventory": not_called_this_step,
            "missed_collections": missed_this_step,
            "collection_completed": completed_this_step,
        }

        for index, tank in enumerate(tanks, start=1):
            offset = index - 1
            record[f"tank{index}_name"] = tank.name
            record[f"tank{index}_tonnes"] = tank_levels[offset]
            record[f"tank{index}_capacity"] = tank.capacity
            record[f"tank{index}_state"] = tank_states[offset]
            record[f"tank{index}_static_hours"] = tank_static_hours[offset]
            record[f"tank{index}_test_ready"] = (
                tank_static_hours[offset] >= testing_required_hours
            )
            record[f"tank{index}_filled"] = filled[offset]
            record[f"tank{index}_emptied"] = emptied[offset]

        records.append(record)

    tank_df = pd.DataFrame(records)
    tank_df["date"] = tank_df["time"].dt.date
    tank_df["total_collected"] = sum(
        tank_df[f"tank{index}_emptied"] for index in range(1, num_tanks + 1)
    )
    tank_df["total_filled"] = sum(
        tank_df[f"tank{index}_filled"] for index in range(1, num_tanks + 1)
    )
    tank_df["cumulative_collected"] = tank_df["total_collected"].cumsum()
    tank_df["cumulative_lost_co2"] = tank_df["lost_co2"].cumsum()
    tank_df["cumulative_produced"] = tank_df["produced_tonnes"].cumsum()

    aggregation = {
        "produced_tonnes": ("produced_tonnes", "sum"),
        "filled_tonnes": ("total_filled", "sum"),
        "collected_tonnes": ("total_collected", "sum"),
        "lost_co2": ("lost_co2", "sum"),
        "called_collections": ("called_collections", "sum"),
        "not_called_insufficient_inventory": (
            "not_called_insufficient_inventory",
            "sum",
        ),
        "missed_collections": ("missed_collections", "sum"),
        "completed_collections": ("collection_completed", "sum"),
        "maximum_outstanding": ("outstanding_collections", "max"),
        "minimum_production_rate": ("production_rate_tph", "min"),
        "maximum_production_rate": ("production_rate_tph", "max"),
    }
    for index in range(1, num_tanks + 1):
        aggregation[f"max_tank{index}"] = (f"tank{index}_tonnes", "max")

    daily_summary = tank_df.groupby("date").agg(**aggregation).reset_index()

    total_produced = float(tank_df["produced_tonnes"].sum())
    total_filled = float(tank_df["total_filled"].sum())
    total_collected = float(tank_df["total_collected"].sum())
    total_lost = float(tank_df["lost_co2"].sum())
    final_storage = float(sum(tank_levels))
    pending_jobs = [
        job for job in all_jobs if job["status"] in {"queued", "active"}
    ]
    partially_completed = len(failed_partial_jobs) + sum(
        1 for job in pending_jobs if job["collected"] > 1e-9
    )
    start_delays = [
        (job["loading_started_at"] - job["scheduled_time"]).total_seconds() / 3600
        for job in completed_jobs
        if job["loading_started_at"] is not None
    ]
    turnaround_times = [
        (job["completed_at"] - job["scheduled_time"]).total_seconds() / 3600
        for job in completed_jobs
        if job["completed_at"] is not None
    ]
    expected_full_production = default_fill_rate * simulation_days * 24
    mass_balance_error = (
        initial_storage
        + total_produced
        - total_collected
        - total_lost
        - final_storage
    )
    called_count = int(tank_df["called_collections"].sum())
    missed_count = int(tank_df["missed_collections"].sum())
    arrived_count = called_count - missed_count
    not_called_count = int(
        tank_df["not_called_insufficient_inventory"].sum()
    )

    metrics = {
        "scheduled_collections": len(in_horizon_collections),
        "candidate_collection_slots": len(in_horizon_collections),
        "called_collections": called_count,
        "not_called_insufficient_inventory": not_called_count,
        "arrived_collections": arrived_count,
        "completed_collections": len(completed_jobs),
        "missed_collections": missed_count,
        "pending_collections_at_end": len(pending_jobs),
        "partially_completed_collections": partially_completed,
        "maximum_queue_length": maximum_waiting_queue,
        "maximum_outstanding_collections": maximum_outstanding,
        "average_collection_delay_hours": (
            float(np.mean(start_delays)) if start_delays else None
        ),
        "maximum_collection_delay_hours": (
            float(np.max(start_delays)) if start_delays else None
        ),
        "average_collection_turnaround_hours": (
            float(np.mean(turnaround_times)) if turnaround_times else None
        ),
        "initial_storage": initial_storage,
        "final_storage": final_storage,
        "total_produced": total_produced,
        "expected_full_production": expected_full_production,
        "production_shortfall": max(0.0, expected_full_production - total_produced),
        "total_filled": total_filled,
        "total_collected": total_collected,
        "total_lost": total_lost,
        "loss_percentage": total_lost / total_produced * 100 if total_produced else 0.0,
        "capture_percentage": total_filled / total_produced * 100 if total_produced else 0.0,
        "completed_arrivals_percentage": (
            len(completed_jobs) / arrived_count * 100 if arrived_count else 100.0
        ),
        "plant_offline_hours": float(
            (tank_df["production_rate_tph"] == 0).sum() * step_hours
        ),
        "storage_constrained_hours": float(
            (tank_df["lost_co2"] > 1e-9).sum() * step_hours
        ),
        "mass_balance_error": float(mass_balance_error),
    }

    return tank_df, daily_summary, metrics


def _configuration_tanks(configuration: SimulationConfiguration) -> list[Tank]:
    if len(configuration.tank_capacities) != len(configuration.tank_minimums):
        raise ValueError(f"{configuration.name}: capacities and minimums must align")

    return [
        Tank(
            name=f"Tank {index}",
            capacity=float(capacity),
            minimum=float(minimum),
        )
        for index, (capacity, minimum) in enumerate(
            zip(configuration.tank_capacities, configuration.tank_minimums),
            start=1,
        )
    ]


def run_resilience_comparison(
    configurations: Sequence[SimulationConfiguration],
    simulation_start: str | pd.Timestamp,
    simulation_days: int,
    default_fill_rate: float,
    testing_required_hours: float,
    collection_target_tonnes: float,
    manual_production_periods: Sequence[ProductionPeriod] | None = None,
    stress_scenarios: Sequence[str] = STRESS_SCENARIOS,
    random_downtime_percentage: float = 0.0,
    random_missed_percentage: float = 0.0,
    monte_carlo_trials: int = 20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare configurations under deterministic stresses and repeated random trials."""
    if not configurations:
        raise ValueError("At least one configuration is required")
    if monte_carlo_trials < 1:
        raise ValueError("monte_carlo_trials must be at least 1")

    manual_production_periods = list(manual_production_periods or [])
    results: list[dict] = []

    for scenario in stress_scenarios:
        trial_count = monte_carlo_trials if scenario == "Monte Carlo disruptions" else 1

        for trial in range(trial_count):
            trial_seed = seed + trial * 10_007
            automatic_downtime = []
            if scenario == "Monte Carlo disruptions":
                automatic_downtime = generate_random_downtime_periods(
                    simulation_start,
                    simulation_days,
                    random_downtime_percentage,
                    seed=trial_seed,
                )
            elif scenario == "24-hour plant outage":
                outage_start = (
                    pd.Timestamp(simulation_start).normalize()
                    + pd.Timedelta(days=min(max(1, simulation_days // 3), simulation_days - 1))
                )
                automatic_downtime = [
                    ProductionPeriod(
                        start=outage_start.to_pydatetime(),
                        end=(outage_start + pd.Timedelta(hours=24)).to_pydatetime(),
                        rate=0.0,
                        source="stress scenario",
                    )
                ]

            for configuration in configurations:
                times = evenly_spaced_collection_times(configuration.collections_per_day)
                collections = generate_default_collections(
                    simulation_start,
                    simulation_days,
                    target_tonnes=collection_target_tonnes,
                    loading_rate=configuration.loading_rate,
                    collection_times=times,
                )
                collections = apply_named_stress_scenario(
                    collections,
                    scenario,
                    simulation_start,
                    simulation_days,
                )
                tanks = _configuration_tanks(configuration)
                _, _, metrics = run_simulation(
                    tanks=tanks,
                    collections=collections,
                    production_periods=manual_production_periods + automatic_downtime,
                    default_fill_rate=default_fill_rate,
                    testing_required_hours=testing_required_hours,
                    simulation_start=simulation_start,
                    simulation_days=simulation_days,
                    max_parallel_loads=configuration.max_parallel_loads,
                    random_missed_percentage=(
                        random_missed_percentage
                        if scenario == "Monte Carlo disruptions"
                        else 0.0
                    ),
                    random_seed=trial_seed + 1,
                )
                results.append(
                    {
                        "configuration": configuration.name,
                        "scenario": scenario,
                        "trial": trial + 1,
                        "tank_count": len(configuration.tank_capacities),
                        "nominal_capacity_t": sum(configuration.tank_capacities),
                        "usable_capacity_t": sum(
                            capacity - minimum
                            for capacity, minimum in zip(
                                configuration.tank_capacities,
                                configuration.tank_minimums,
                            )
                        ),
                        "loading_rate_tph": configuration.loading_rate,
                        "parallel_loading_bays": configuration.max_parallel_loads,
                        "collections_per_day": configuration.collections_per_day,
                        "loss_percentage": metrics["loss_percentage"],
                        "lost_tonnes": metrics["total_lost"],
                        "collected_tonnes": metrics["total_collected"],
                        "completed_arrivals_percentage": metrics[
                            "completed_arrivals_percentage"
                        ],
                        "called_collections": metrics["called_collections"],
                        "not_called_insufficient_inventory": metrics[
                            "not_called_insufficient_inventory"
                        ],
                        "pending_collections": metrics["pending_collections_at_end"],
                        "maximum_outstanding": metrics[
                            "maximum_outstanding_collections"
                        ],
                        "storage_constrained_hours": metrics[
                            "storage_constrained_hours"
                        ],
                        "production_shortfall_t": metrics["production_shortfall"],
                    }
                )

    results_df = pd.DataFrame(results)
    summary = (
        results_df.groupby("configuration", as_index=False)
        .agg(
            tank_count=("tank_count", "first"),
            nominal_capacity_t=("nominal_capacity_t", "first"),
            usable_capacity_t=("usable_capacity_t", "first"),
            loading_rate_tph=("loading_rate_tph", "first"),
            parallel_loading_bays=("parallel_loading_bays", "first"),
            collections_per_day=("collections_per_day", "first"),
            average_loss_pct=("loss_percentage", "mean"),
            worst_case_loss_pct=("loss_percentage", "max"),
            p95_loss_pct=("loss_percentage", lambda values: values.quantile(0.95)),
            average_completed_arrivals_pct=("completed_arrivals_percentage", "mean"),
            average_not_called_slots=(
                "not_called_insufficient_inventory",
                "mean",
            ),
            worst_pending_collections=("pending_collections", "max"),
            average_storage_constrained_hours=("storage_constrained_hours", "mean"),
        )
    )
    summary = summary.sort_values(
        [
            "worst_case_loss_pct",
            "average_loss_pct",
            "worst_pending_collections",
            "nominal_capacity_t",
        ],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    summary.insert(0, "resilience_rank", np.arange(1, len(summary) + 1))

    return results_df, summary
