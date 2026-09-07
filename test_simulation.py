import pandas as pd
import unittest

from models import (
    Collection,
    ProductionPeriod,
    SimulationConfiguration,
    Tank,
)
from simulation import (
    apply_collection_disruptions,
    generate_default_collections,
    generate_random_downtime_periods,
    run_resilience_comparison,
    run_simulation,
)


START = pd.Timestamp("2024-01-01")


class SimulationTests(unittest.TestCase):
    def test_default_schedule_generates_two_collections_per_day(self):
        collections = generate_default_collections(START, 30)
        self.assertEqual(len(collections), 60)
        self.assertEqual(
            {pd.Timestamp(item.time).strftime("%H:%M") for item in collections},
            {"08:00", "18:00"},
        )

    def test_trailer_is_called_only_for_one_full_load_tank(self):
        collections = generate_default_collections(
            START,
            30,
            collection_times=("08:00", "14:00"),
        )
        tank_df, _, metrics = run_simulation(
            tanks=[Tank("Tank 1", 45, 5), Tank("Tank 2", 45, 5)],
            collections=collections,
            default_fill_rate=1.6,
            testing_required_hours=2,
            simulation_start=START,
            simulation_days=30,
        )
        self.assertEqual(metrics["candidate_collection_slots"], 60)
        self.assertEqual(
            metrics["called_collections"]
            + metrics["not_called_insufficient_inventory"],
            60,
        )
        self.assertEqual(metrics["completed_collections"], 58)
        self.assertEqual(metrics["pending_collections_at_end"], 0)
        self.assertAlmostEqual(metrics["mass_balance_error"], 0.0, places=8)

        for tank_number in (1, 2):
            emptied = tank_df[f"tank{tank_number}_emptied"]
            starts = tank_df[(emptied > 0) & (emptied.shift(fill_value=0) == 0)]
            usable_before_loading = (
                starts[f"tank{tank_number}_tonnes"]
                + starts[f"tank{tank_number}_emptied"]
                - 5
            )
            self.assertTrue((usable_before_loading >= 19 - 1e-9).all())

    def test_random_miss_is_applied_only_after_a_trailer_is_called(self):
        collection = [Collection(START + pd.Timedelta(hours=8), 19, 8)]
        _, _, enough_metrics = run_simulation(
            [Tank("Tank 1", 45, 5, 40)],
            collection,
            testing_required_hours=0,
            simulation_start=START,
            simulation_days=1,
            random_missed_percentage=100,
        )
        self.assertEqual(enough_metrics["called_collections"], 1)
        self.assertEqual(enough_metrics["missed_collections"], 1)

        _, _, insufficient_metrics = run_simulation(
            [Tank("Tank 1", 45, 5, 5)],
            collection,
            default_fill_rate=0,
            testing_required_hours=0,
            simulation_start=START,
            simulation_days=1,
            random_missed_percentage=100,
        )
        self.assertEqual(insufficient_metrics["called_collections"], 0)
        self.assertEqual(insufficient_metrics["missed_collections"], 0)
        self.assertEqual(
            insufficient_metrics["not_called_insufficient_inventory"],
            1,
        )

    def test_random_downtime_matches_requested_percentage(self):
        periods = generate_random_downtime_periods(
            START,
            30,
            downtime_percentage=20,
            seed=10,
        )
        hours = sum(
            (pd.Timestamp(period.end) - pd.Timestamp(period.start)).total_seconds()
            / 3600
            for period in periods
        )
        self.assertAlmostEqual(hours, 30 * 24 * 0.20, delta=1 / 6)

    def test_manual_and_random_collection_misses_are_applied(self):
        collections = generate_default_collections(START, 10)
        disrupted = apply_collection_disruptions(
            collections,
            no_delivery_periods=[
                (START + pd.Timedelta(days=2), START + pd.Timedelta(days=3))
            ],
            random_missed_percentage=10,
            seed=11,
        )
        manual_misses = [
            item
            for item in disrupted
            if item.missed_reason == "manual no-delivery period"
        ]
        random_misses = [
            item for item in disrupted if item.missed_reason == "random miss"
        ]
        self.assertEqual(len(manual_misses), 2)
        self.assertEqual(len(random_misses), 2)

    def test_manual_zero_production_override(self):
        collections = generate_default_collections(START, 2)
        outage = ProductionPeriod(
            start=(START + pd.Timedelta(hours=6)).to_pydatetime(),
            end=(START + pd.Timedelta(hours=12)).to_pydatetime(),
            rate=0,
        )
        tank_df, _, _ = run_simulation(
            [Tank("Tank 1", 100, 5)],
            collections,
            production_periods=[outage],
            simulation_start=START,
            simulation_days=2,
        )
        outage_rows = tank_df[
            (tank_df["time"] >= START + pd.Timedelta(hours=6))
            & (tank_df["time"] < START + pd.Timedelta(hours=12))
        ]
        self.assertTrue((outage_rows["production_rate_tph"] == 0).all())

    def test_parallel_loading_and_variable_capacities_run(self):
        collections = [
            Collection(START + pd.Timedelta(hours=8), 19, 8),
            Collection(START + pd.Timedelta(hours=8), 19, 8),
        ]
        tank_df, _, metrics = run_simulation(
            [
                Tank("Small", 50, 2.5, 40),
                Tank("Large", 100, 5, 80),
            ],
            collections,
            testing_required_hours=0,
            simulation_start=START,
            simulation_days=1,
            max_parallel_loads=2,
        )
        self.assertEqual(tank_df["collection_active_count"].max(), 2)
        self.assertEqual(metrics["completed_collections"], 2)

    def test_resilience_comparison_returns_ranked_configurations(self):
        configurations = [
            SimulationConfiguration("2 x 100", [100, 100], [5, 5], 8, 1, 2),
            SimulationConfiguration(
                "4 x 50", [50, 50, 50, 50], [2.5] * 4, 8, 1, 2
            ),
        ]
        results, summary = run_resilience_comparison(
            configurations,
            START,
            simulation_days=10,
            default_fill_rate=1.6,
            testing_required_hours=2,
            collection_target_tonnes=19,
            stress_scenarios=["Normal schedule", "Monte Carlo disruptions"],
            random_downtime_percentage=10,
            random_missed_percentage=10,
            monte_carlo_trials=3,
        )
        self.assertEqual(len(results), 8)
        self.assertEqual(list(summary["resilience_rank"]), [1, 2])


if __name__ == "__main__":
    unittest.main()
