from __future__ import annotations

import pandas as pd
import streamlit as st

from charts import (
    collection_pressure_chart,
    cumulative_loss_chart,
    daily_loss_chart,
    production_chart,
    resilience_heatmap,
    resilience_summary_chart,
    tank_level_chart,
)
from models import (
    Collection,
    ProductionPeriod,
    SimulationConfiguration,
    Tank,
)
from simulation import (
    STRESS_SCENARIOS,
    apply_collection_disruptions,
    generate_default_collections,
    generate_random_downtime_periods,
    run_resilience_comparison,
    run_simulation,
)


st.set_page_config(
    page_title="CO₂ Storage Resilience",
    page_icon="◫",
    layout="wide",
)


def collections_to_dataframe(collections: list[Collection]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "time": pd.Timestamp(collection.time),
                "target_tonnes": collection.target_tonnes,
                "loading_rate": collection.loading_rate,
                "arrived": collection.arrived,
            }
            for collection in collections
        ]
    )


def collections_from_dataframe(table: pd.DataFrame) -> list[Collection]:
    collections: list[Collection] = []
    for _, row in table.iterrows():
        if (
            pd.notna(row.get("time"))
            and pd.notna(row.get("target_tonnes"))
            and pd.notna(row.get("loading_rate"))
        ):
            collections.append(
                Collection(
                    time=pd.Timestamp(row["time"]).to_pydatetime(),
                    target_tonnes=float(row["target_tonnes"]),
                    loading_rate=float(row["loading_rate"]),
                    arrived=bool(row.get("arrived", True)),
                )
            )
    return collections


def parse_collection_times(value: str) -> tuple[str, ...]:
    times = tuple(item.strip() for item in value.split(",") if item.strip())
    if not times:
        raise ValueError("Enter at least one collection time")
    for item in times:
        parsed = pd.Timestamp(f"2000-01-01 {item}")
        if not (0 <= parsed.hour <= 23 and 0 <= parsed.minute <= 59):
            raise ValueError(f"Invalid collection time: {item}")
    return times


def parse_number_list(value: str, label: str) -> list[float]:
    try:
        numbers = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{label} must be comma-separated numbers") from exc
    if not numbers:
        raise ValueError(f"{label} cannot be empty")
    return numbers


def format_optional_hours(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f} h"


st.title("CO₂ Storage Resilience Simulator")
st.caption(
    "Test tank layouts, production interruptions, collection failures, pump rates "
    "and simultaneous trailer loading."
)

st.sidebar.header("Shared simulation settings")
simulation_start = st.sidebar.date_input(
    "Simulation start",
    value=pd.Timestamp("2024-01-01").date(),
    key="simulation_start_key",
)
simulation_days = st.sidebar.slider(
    "Simulation length (days)",
    min_value=5,
    max_value=180,
    value=30,
    key="simulation_days_key",
)
default_fill_rate = st.sidebar.number_input(
    "Normal production rate (t/hour)",
    min_value=0.0,
    max_value=20.0,
    value=1.6,
    step=0.1,
    key="default_fill_rate_key",
)
testing_required_hours = st.sidebar.number_input(
    "Tank static/certification time (hours)",
    min_value=0.0,
    max_value=24.0,
    value=2.0,
    step=0.5,
    key="testing_required_hours_key",
)
random_seed = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    value=42,
    step=1,
    help="Use the same seed to reproduce the same random disruptions.",
)

single_tab, comparison_tab = st.tabs(
    ["Single simulation", "Compare configurations"]
)

with single_tab:
    st.header("1. Production")
    st.write(
        "The normal rate applies unless a manual override or generated downtime "
        "period sets a lower rate."
    )

    production_table = st.data_editor(
        pd.DataFrame(columns=["start", "end", "rate"]),
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "start": st.column_config.DatetimeColumn(
                "Start", format="DD MMM YYYY HH:mm"
            ),
            "end": st.column_config.DatetimeColumn(
                "End", format="DD MMM YYYY HH:mm"
            ),
            "rate": st.column_config.NumberColumn(
                "Production rate (t/hour)", min_value=0.0, step=0.1
            ),
        },
        key="production_editor",
    )

    p1, p2, p3 = st.columns(3)
    random_downtime_percentage = p1.slider(
        "Random downtime",
        min_value=0,
        max_value=100,
        value=0,
        format="%d%%",
        help="Creates reproducible blocks of zero production across the run.",
    )
    minimum_downtime_hours = p2.number_input(
        "Minimum downtime block (hours)",
        min_value=0.5,
        max_value=72.0,
        value=2.0,
        step=0.5,
    )
    maximum_downtime_hours = p3.number_input(
        "Maximum downtime block (hours)",
        min_value=0.5,
        max_value=168.0,
        value=18.0,
        step=0.5,
    )

    manual_production_periods: list[ProductionPeriod] = []
    invalid_production_period = False
    for _, row in production_table.iterrows():
        if (
            pd.notna(row.get("start"))
            and pd.notna(row.get("end"))
            and pd.notna(row.get("rate"))
        ):
            start = pd.Timestamp(row["start"])
            end = pd.Timestamp(row["end"])
            if end <= start:
                invalid_production_period = True
            manual_production_periods.append(
                ProductionPeriod(
                    start=start.to_pydatetime(),
                    end=end.to_pydatetime(),
                    rate=float(row["rate"]),
                    source="manual",
                )
            )

    st.header("2. Tank configuration")
    st.write(
        "Add or remove tanks. Capacity, minimum heel and starting level can differ "
        "for every tank."
    )
    tank_table = st.data_editor(
        pd.DataFrame(
            [
                {
                    "name": "Tank 1",
                    "capacity": 100.0,
                    "minimum": 5.0,
                    "initial_level": 50.0,
                },
                {
                    "name": "Tank 2",
                    "capacity": 100.0,
                    "minimum": 5.0,
                    "initial_level": 50.0,
                },
            ]
        ),
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("Tank", required=True),
            "capacity": st.column_config.NumberColumn(
                "Capacity (t)", min_value=0.1, step=5.0, required=True
            ),
            "minimum": st.column_config.NumberColumn(
                "Minimum heel (t)", min_value=0.0, step=1.0, required=True
            ),
            "initial_level": st.column_config.NumberColumn(
                "Starting level (t)", min_value=0.0, step=5.0, required=True
            ),
        },
        key="tank_editor",
    )
    max_parallel_loads = st.number_input(
        "Trailers that can load simultaneously",
        min_value=1,
        max_value=6,
        value=1,
        step=1,
        help="Use 2 to model two independent loading connections/bays.",
    )

    tanks: list[Tank] = []
    for _, row in tank_table.iterrows():
        if all(
            pd.notna(row.get(column))
            for column in ("name", "capacity", "minimum", "initial_level")
        ):
            tanks.append(
                Tank(
                    name=str(row["name"]),
                    capacity=float(row["capacity"]),
                    minimum=float(row["minimum"]),
                    initial_level=float(row["initial_level"]),
                )
            )

    st.header("3. Trailer schedule")
    st.write(
        "These times are call opportunities. A trailer is called only when one "
        "certified tank contains the complete target load above its minimum heel."
    )
    c1, c2, c3 = st.columns(3)
    collection_times_text = c1.text_input(
        "Daily collection times",
        value="08:00, 18:00",
        key="collection_times_input",
        help="Add more comma-separated times to schedule more than two trailers.",
    )
    collection_target = c2.number_input(
        "Target per trailer (t)",
        min_value=0.1,
        max_value=100.0,
        value=19.0,
        step=1.0,
        key="collection_target_key",
    )
    default_loading_rate = c3.number_input(
        "Offtake pump rate (t/hour)",
        min_value=0.1,
        max_value=100.0,
        value=8.0,
        step=0.5,
        key="loading_rate_key",
    )

    try:
        current_collection_times = parse_collection_times(collection_times_text)
        collection_times_valid = True
    except ValueError as exc:
        st.error(str(exc))
        current_collection_times = ("08:00", "18:00")
        collection_times_valid = False

    desired_schedule_basis = (
        str(simulation_start),
        int(simulation_days),
        current_collection_times,
        float(collection_target),
        float(default_loading_rate),
    )

    if "collection_schedule_df" not in st.session_state:
        st.session_state.collection_schedule_df = collections_to_dataframe(
            generate_default_collections(
                simulation_start,
                simulation_days,
                collection_target,
                default_loading_rate,
                current_collection_times,
            )
        )
        st.session_state.collection_schedule_basis = desired_schedule_basis

    b1, b2 = st.columns(2)
    if b1.button("Regenerate from current pattern", use_container_width=True):
        if collection_times_valid:
            st.session_state.collection_schedule_df = collections_to_dataframe(
                generate_default_collections(
                    simulation_start,
                    simulation_days,
                    collection_target,
                    default_loading_rate,
                    current_collection_times,
                )
            )
            st.session_state.collection_schedule_basis = desired_schedule_basis
            st.session_state.pop("collection_schedule_editor", None)
            st.rerun()

    if b2.button(
        "Reset to default 2/day schedule",
        use_container_width=True,
    ):
        default_times = ("08:00", "18:00")
        st.session_state.collection_schedule_df = collections_to_dataframe(
            generate_default_collections(
                simulation_start,
                simulation_days,
                collection_target,
                default_loading_rate,
                default_times,
            )
        )
        st.session_state.collection_schedule_basis = (
            str(simulation_start),
            int(simulation_days),
            default_times,
            float(collection_target),
            float(default_loading_rate),
        )
        st.session_state.pop("collection_schedule_editor", None)
        st.rerun()

    if st.session_state.get("collection_schedule_basis") != desired_schedule_basis:
        st.warning(
            "The simulation period or recurring schedule has changed. Regenerate "
            "the schedule if you want its dates and defaults updated."
        )

    collection_table = st.data_editor(
        st.session_state.collection_schedule_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "time": st.column_config.DatetimeColumn(
                "Collection time", format="DD MMM YYYY HH:mm", required=True
            ),
            "target_tonnes": st.column_config.NumberColumn(
                "Target (t)", min_value=0.1, step=1.0, required=True
            ),
            "loading_rate": st.column_config.NumberColumn(
                "Loading rate (t/hour)", min_value=0.1, step=0.5, required=True
            ),
            "arrived": st.column_config.CheckboxColumn(
                "Truck available if called", default=True
            ),
        },
        key="collection_schedule_editor",
    )
    st.session_state.collection_schedule_df = collection_table

    st.subheader("Collection disruption assumptions")
    d1, d2 = st.columns(2)
    random_missed_percentage = d1.slider(
        "Chance a called trailer is randomly missed",
        min_value=0,
        max_value=100,
        value=0,
        format="%d%%",
    )
    d2.caption(
        "Use the table below for known no-delivery dates, such as a four-day "
        "bank-holiday shutdown. The end date is inclusive."
    )
    no_delivery_table = st.data_editor(
        pd.DataFrame(columns=["start_date", "end_date"]),
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "start_date": st.column_config.DateColumn("First no-delivery day"),
            "end_date": st.column_config.DateColumn("Last no-delivery day"),
        },
        key="no_delivery_editor",
    )
    no_delivery_periods: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    invalid_no_delivery_period = False
    for _, row in no_delivery_table.iterrows():
        if pd.notna(row.get("start_date")) and pd.notna(row.get("end_date")):
            start = pd.Timestamp(row["start_date"]).normalize()
            end = pd.Timestamp(row["end_date"]).normalize() + pd.Timedelta(days=1)
            if end <= start:
                invalid_no_delivery_period = True
            no_delivery_periods.append((start, end))

    st.divider()
    if st.button(
        "Run single simulation",
        type="primary",
        use_container_width=True,
    ):
        errors: list[str] = []
        if not tanks:
            errors.append("Add at least one tank.")
        for tank in tanks:
            if tank.capacity <= tank.minimum:
                errors.append(f"{tank.name}: capacity must exceed minimum.")
            if not tank.minimum <= float(tank.initial_level) <= tank.capacity:
                errors.append(
                    f"{tank.name}: starting level must lie between minimum and capacity."
                )
        if tanks and not any(
            tank.capacity - tank.minimum >= collection_target for tank in tanks
        ):
            errors.append(
                "No single tank can hold one complete trailer target above its minimum."
            )
        if invalid_production_period:
            errors.append("Every production override must end after it starts.")
        if invalid_no_delivery_period:
            errors.append("Every no-delivery period must end on or after it starts.")
        if maximum_downtime_hours < minimum_downtime_hours:
            errors.append("Maximum downtime block must be at least the minimum.")
        if not collection_times_valid:
            errors.append("Correct the daily collection times.")

        if errors:
            for error in errors:
                st.error(error)
        else:
            automatic_downtime = generate_random_downtime_periods(
                simulation_start,
                simulation_days,
                random_downtime_percentage,
                seed=int(random_seed),
                minimum_duration_hours=minimum_downtime_hours,
                maximum_duration_hours=maximum_downtime_hours,
            )
            scheduled_collections = collections_from_dataframe(collection_table)
            disrupted_collections = apply_collection_disruptions(
                scheduled_collections,
                no_delivery_periods=no_delivery_periods,
            )
            tank_df, daily_summary, metrics = run_simulation(
                tanks=tanks,
                collections=disrupted_collections,
                production_periods=manual_production_periods + automatic_downtime,
                default_fill_rate=default_fill_rate,
                testing_required_hours=testing_required_hours,
                simulation_start=pd.Timestamp(simulation_start),
                simulation_days=simulation_days,
                max_parallel_loads=int(max_parallel_loads),
                random_missed_percentage=random_missed_percentage,
                random_seed=int(random_seed) + 1,
            )
            st.session_state.single_results = {
                "tank_df": tank_df,
                "daily_summary": daily_summary,
                "metrics": metrics,
                "tanks": tanks,
                "automatic_downtime": automatic_downtime,
                "collections": disrupted_collections,
            }

    if "single_results" in st.session_state:
        result = st.session_state.single_results
        tank_df = result["tank_df"]
        daily_summary = result["daily_summary"]
        metrics = result["metrics"]
        result_tanks = result["tanks"]

        st.header("Simulation results")
        row1 = st.columns(6)
        row1[0].metric("Produced", f"{metrics['total_produced']:.1f} t")
        row1[1].metric("Collected", f"{metrics['total_collected']:.1f} t")
        row1[2].metric("CO₂ lost", f"{metrics['total_lost']:.1f} t")
        row1[3].metric("Loss rate", f"{metrics['loss_percentage']:.1f}%")
        row1[4].metric("Final storage", f"{metrics['final_storage']:.1f} t")
        row1[5].metric(
            "Production shortfall", f"{metrics['production_shortfall']:.1f} t"
        )

        row2 = st.columns(6)
        row2[0].metric("Call opportunities", metrics["candidate_collection_slots"])
        row2[1].metric("Called", metrics["called_collections"])
        row2[2].metric(
            "Not called",
            metrics["not_called_insufficient_inventory"],
            help="No single certified tank held the full trailer load.",
        )
        row2[3].metric("Missed after call", metrics["missed_collections"])
        row2[4].metric("Completed", metrics["completed_collections"])
        row2[5].metric("Pending", metrics["pending_collections_at_end"])

        row3 = st.columns(2)
        row3[0].metric(
            "Average delay",
            format_optional_hours(metrics["average_collection_delay_hours"]),
        )
        row3[1].metric(
            "Maximum outstanding",
            metrics["maximum_outstanding_collections"],
        )

        nominal_capacity = sum(tank.capacity for tank in result_tanks)
        usable_capacity = sum(
            tank.capacity - tank.minimum for tank in result_tanks
        )
        st.info(
            f"**{len(result_tanks)} tanks** | "
            f"**{nominal_capacity:.1f} t nominal capacity** | "
            f"**{usable_capacity:.1f} t usable capacity** | "
            f"mass-balance error: **{metrics['mass_balance_error']:.6f} t**"
        )

        st.plotly_chart(
            tank_level_chart(tank_df, result_tanks),
            use_container_width=True,
        )
        st.plotly_chart(production_chart(tank_df), use_container_width=True)
        st.plotly_chart(
            collection_pressure_chart(tank_df),
            use_container_width=True,
        )
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                cumulative_loss_chart(tank_df),
                use_container_width=True,
            )
        with right:
            st.plotly_chart(
                daily_loss_chart(daily_summary),
                use_container_width=True,
            )

        with st.expander("Generated random downtime periods"):
            downtime_rows = [
                {
                    "start": period.start,
                    "end": period.end,
                    "duration_hours": (
                        pd.Timestamp(period.end) - pd.Timestamp(period.start)
                    ).total_seconds()
                    / 3600,
                }
                for period in result["automatic_downtime"]
            ]
            st.dataframe(
                pd.DataFrame(downtime_rows),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("Collection opportunities supplied"):
            actual_schedule = pd.DataFrame(
                [
                    {
                        "time": collection.time,
                        "target_tonnes": collection.target_tonnes,
                        "loading_rate": collection.loading_rate,
                        "arrived": collection.arrived,
                        "missed_reason": collection.missed_reason,
                    }
                    for collection in result["collections"]
                ]
            )
            st.dataframe(actual_schedule, use_container_width=True, hide_index=True)

        st.subheader("Daily summary")
        st.dataframe(
            daily_summary.round(3),
            use_container_width=True,
            hide_index=True,
        )
        with st.expander("View 10-minute simulation data"):
            st.dataframe(tank_df.round(3), use_container_width=True)

        st.download_button(
            "Download simulation CSV",
            data=tank_df.to_csv(index=False).encode("utf-8"),
            file_name="co2_simulation.csv",
            mime="text/csv",
        )

with comparison_tab:
    st.header("Configuration resilience comparison")
    st.write(
        "Each configuration is tested against named operational stresses and "
        "repeated random disruption trials. Ranking uses lowest worst-case loss, "
        "then lowest average loss, pending trailers and finally storage capacity."
    )
    st.warning(
        "This is an operational resilience ranking, not a financial recommendation. "
        "CAPEX and OPEX must be added before judging economic efficiency."
    )

    default_configurations = pd.DataFrame(
        [
            {
                "name": "Current 2 × 100 t",
                "capacities_t": "100,100",
                "minimums_t": "5,5",
                "pump_rate_tph": 8.0,
                "parallel_bays": 1,
                "collections_per_day": 2,
            },
            {
                "name": "Split 4 × 50 t",
                "capacities_t": "50,50,50,50",
                "minimums_t": "2.5,2.5,2.5,2.5",
                "pump_rate_tph": 8.0,
                "parallel_bays": 1,
                "collections_per_day": 2,
            },
            {
                "name": "Higher pump",
                "capacities_t": "100,100",
                "minimums_t": "5,5",
                "pump_rate_tph": 16.0,
                "parallel_bays": 1,
                "collections_per_day": 2,
            },
            {
                "name": "Dual trailer loading",
                "capacities_t": "100,100",
                "minimums_t": "5,5",
                "pump_rate_tph": 8.0,
                "parallel_bays": 2,
                "collections_per_day": 2,
            },
            {
                "name": "Split + pump + dual",
                "capacities_t": "50,50,50,50",
                "minimums_t": "2.5,2.5,2.5,2.5",
                "pump_rate_tph": 16.0,
                "parallel_bays": 2,
                "collections_per_day": 2,
            },
            {
                "name": "Additional 100 t tank",
                "capacities_t": "100,100,100",
                "minimums_t": "5,5,5",
                "pump_rate_tph": 8.0,
                "parallel_bays": 1,
                "collections_per_day": 2,
            },
        ]
    )
    configuration_table = st.data_editor(
        default_configurations,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("Configuration", required=True),
            "capacities_t": st.column_config.TextColumn(
                "Tank capacities (comma separated)", required=True
            ),
            "minimums_t": st.column_config.TextColumn(
                "Minimum heels (comma separated)", required=True
            ),
            "pump_rate_tph": st.column_config.NumberColumn(
                "Pump rate (t/hour)", min_value=0.1, step=1.0, required=True
            ),
            "parallel_bays": st.column_config.NumberColumn(
                "Parallel loading bays", min_value=1, max_value=6, step=1, required=True
            ),
            "collections_per_day": st.column_config.NumberColumn(
                "Planned trailers/day", min_value=1, max_value=8, step=1, required=True
            ),
        },
        key="configuration_editor",
    )

    selected_stresses = st.multiselect(
        "Stress cases",
        options=list(STRESS_SCENARIOS),
        default=list(STRESS_SCENARIOS),
    )
    r1, r2, r3 = st.columns(3)
    comparison_downtime = r1.slider(
        "Random-trial downtime",
        min_value=0,
        max_value=100,
        value=20,
        format="%d%%",
    )
    comparison_misses = r2.slider(
        "Random-trial missed deliveries",
        min_value=0,
        max_value=100,
        value=10,
        format="%d%%",
    )
    monte_carlo_trials = r3.slider(
        "Monte Carlo trials per configuration",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
    )

    if st.button(
        "Run resilience comparison",
        type="primary",
        use_container_width=True,
    ):
        configurations: list[SimulationConfiguration] = []
        configuration_errors: list[str] = []
        for _, row in configuration_table.iterrows():
            if pd.isna(row.get("name")):
                continue
            try:
                capacities = parse_number_list(
                    str(row["capacities_t"]),
                    f"{row['name']} capacities",
                )
                minimums = parse_number_list(
                    str(row["minimums_t"]),
                    f"{row['name']} minimums",
                )
                if len(capacities) != len(minimums):
                    raise ValueError(
                        f"{row['name']}: capacities and minimums need the same number of values"
                    )
                if any(capacity <= minimum for capacity, minimum in zip(capacities, minimums)):
                    raise ValueError(
                        f"{row['name']}: every capacity must exceed its minimum"
                    )
                configurations.append(
                    SimulationConfiguration(
                        name=str(row["name"]),
                        tank_capacities=capacities,
                        tank_minimums=minimums,
                        loading_rate=float(row["pump_rate_tph"]),
                        max_parallel_loads=int(row["parallel_bays"]),
                        collections_per_day=int(row["collections_per_day"]),
                    )
                )
            except (TypeError, ValueError) as exc:
                configuration_errors.append(str(exc))

        if not selected_stresses:
            configuration_errors.append("Select at least one stress case.")
        if not configurations:
            configuration_errors.append("Add at least one valid configuration.")

        if configuration_errors:
            for error in configuration_errors:
                st.error(error)
        else:
            with st.spinner("Running configuration stress tests..."):
                results_df, summary_df = run_resilience_comparison(
                    configurations=configurations,
                    simulation_start=simulation_start,
                    simulation_days=simulation_days,
                    default_fill_rate=default_fill_rate,
                    testing_required_hours=testing_required_hours,
                    collection_target_tonnes=19.0,
                    manual_production_periods=manual_production_periods,
                    stress_scenarios=selected_stresses,
                    random_downtime_percentage=comparison_downtime,
                    random_missed_percentage=comparison_misses,
                    monte_carlo_trials=monte_carlo_trials,
                    seed=int(random_seed),
                )
                st.session_state.comparison_results = {
                    "results": results_df,
                    "summary": summary_df,
                }

    if "comparison_results" in st.session_state:
        comparison = st.session_state.comparison_results
        summary_df = comparison["summary"]
        results_df = comparison["results"]
        best = summary_df.iloc[0]

        st.success(
            f"Best operational resilience: **{best['configuration']}**. "
            f"Worst-case loss {best['worst_case_loss_pct']:.2f}% and "
            f"average loss {best['average_loss_pct']:.2f}% across the selected runs."
        )
        st.dataframe(
            summary_df.round(3),
            use_container_width=True,
            hide_index=True,
        )
        st.plotly_chart(
            resilience_summary_chart(summary_df),
            use_container_width=True,
        )
        st.plotly_chart(
            resilience_heatmap(results_df),
            use_container_width=True,
        )

        with st.expander("All scenario and Monte Carlo results"):
            st.dataframe(
                results_df.round(3),
                use_container_width=True,
                hide_index=True,
            )

        st.download_button(
            "Download resilience comparison CSV",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="co2_resilience_comparison.csv",
            mime="text/csv",
        )
