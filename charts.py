import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def tank_level_chart(tank_df, tanks):
    tank_columns = [
        f"tank{i}_tonnes"
        for i in range(1, len(tanks) + 1)
    ]

    plot_df = tank_df[
        ["time"] + tank_columns
    ].copy()

    plot_df = plot_df.rename(
        columns={
            f"tank{i}_tonnes": tanks[i - 1].name
            for i in range(1, len(tanks) + 1)
        }
    )

    long_df = plot_df.melt(
        id_vars="time",
        var_name="Tank",
        value_name="CO₂ stored",
    )

    fig = px.line(
        long_df,
        x="time",
        y="CO₂ stored",
        color="Tank",
        title="Tank levels",
    )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="CO₂ stored (tonnes)",
        hovermode="x unified",
        height=500,
    )

    return fig


def cumulative_loss_chart(tank_df):
    fig = px.line(
        tank_df,
        x="time",
        y="cumulative_lost_co2",
        title="Cumulative CO₂ lost",
    )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="CO₂ lost (tonnes)",
        hovermode="x unified",
        height=400,
    )

    return fig


def daily_loss_chart(daily_summary):
    fig = px.bar(
        daily_summary,
        x="date",
        y="lost_co2",
        title="Daily CO₂ lost",
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="CO₂ lost (tonnes)",
        height=400,
    )

    return fig


def production_chart(tank_df):
    fig = px.line(
        tank_df,
        x="time",
        y="production_rate_tph",
        title="Plant CO₂ production rate",
    )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Production rate (t/hour)",
        hovermode="x unified",
        height=350,
    )

    return fig


def collection_activity_chart(tank_df):
    collection_df = tank_df[
        tank_df["total_collected"] > 0
    ].copy()

    if collection_df.empty:
        return None

    collection_df["collection_rate"] = (
        collection_df["total_collected"]
        / (
            (
                tank_df["time"].iloc[1]
                - tank_df["time"].iloc[0]
            ).total_seconds()
            / 3600
        )
    )

    fig = px.bar(
        collection_df,
        x="time",
        y="collection_rate",
        title="Collection activity",
    )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Collection rate (t/hour)",
        height=300,
    )

    return fig


def collection_pressure_chart(tank_df):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=tank_df["time"],
            y=tank_df["outstanding_collections"],
            name="Outstanding trailers",
            mode="lines",
            line={"width": 1.5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=tank_df["time"],
            y=tank_df["collection_active_count"],
            name="Actively loading/testing",
            mode="lines",
            line={"width": 1.2, "dash": "dot"},
        )
    )
    fig.update_layout(
        title="Trailer pressure and loading activity",
        xaxis_title="Time",
        yaxis_title="Trailers",
        hovermode="x unified",
        height=350,
    )
    return fig


def resilience_summary_chart(summary_df):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=summary_df["configuration"],
            y=summary_df["average_loss_pct"],
            name="Average loss",
        )
    )
    fig.add_trace(
        go.Bar(
            x=summary_df["configuration"],
            y=summary_df["worst_case_loss_pct"],
            name="Worst-case loss",
        )
    )
    fig.update_layout(
        title="Average and worst-case CO₂ loss",
        xaxis_title="Configuration",
        yaxis_title="Loss (%)",
        barmode="group",
        height=450,
    )
    return fig


def resilience_heatmap(results_df):
    heatmap_data = (
        results_df.groupby(["configuration", "scenario"])["loss_percentage"]
        .mean()
        .unstack(fill_value=0.0)
    )
    fig = px.imshow(
        heatmap_data,
        labels={"x": "Stress scenario", "y": "Configuration", "color": "Loss (%)"},
        aspect="auto",
        color_continuous_scale="YlOrRd",
        text_auto=".1f",
        title="Mean loss by configuration and stress scenario",
    )
    fig.update_layout(height=max(400, 65 * len(heatmap_data)))
    return fig
