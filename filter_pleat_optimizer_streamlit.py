
"""
filter_pleat_optimizer.py

Purpose
-------
Optimize a pleated filter pack for:
1) clean-filter restriction at a required maximum airflow,
2) media area,
3) pleat depth,
4) number of pleats,
5) practical pleat behavior limits such as deep-pleat blinding, uneven loading,
   and pleat collapse/distortion under vacuum.

Important assumptions
---------------------
- Units are metric:
    airflow: m^3/h
    pressure: Pa
    dimensions: mm
    area: m^2
- Pleats are assumed to run the full usable frame height.
- Pleat count is across the usable frame width.
- Gross media area is approximated as:
      gross_area = 2 * pleat_depth * usable_height * pleat_count
  adjusted for fold-tip loss.
- Media pressure drop follows:
      dp = K * media_velocity^n
  where K is calibrated from the Palas/bench test.
- The model includes empirical penalties for:
    a) loss of usable area in very deep/tight pleats,
    b) maldistribution/channel restriction,
    c) pleat collapse under vacuum.
- This is an engineering screening tool. Final values should be verified by
  prototype testing because pleat stiffness, adhesive beads, spacers, backing mesh,
  media grade, humidity, dust cake behavior, and vibration strongly affect results.
"""

from dataclasses import dataclass
from math import pi, floor
from typing import Dict, List, Optional, Tuple
import csv


@dataclass
class FilterDesignInputs:
    # Required operating and test data
    q_max_m3_h: float                 # Required maximum airflow through the finished filter
    dp_limit_pa: float                # Maximum allowable clean-filter restriction at q_max_m3_h
    dp_test_pa: float                 # Measured restriction from Palas/bench test
    q_test_m3_h: float                # Airflow used for the Palas/bench test
    test_media_area_m2: float         # Media area in the test. Required to convert test flow to media velocity.
    frame_height_mm: float            # Filter frame height
    frame_width_mm: float             # Filter frame width

    # Package/manufacturing limits
    max_pleat_depth_mm: float = 60.0  # Physical package depth available for the pleat
    min_pleat_depth_mm: float = 8.0
    pleat_depth_step_mm: float = 1.0
    edge_dead_zone_mm: float = 5.0    # Lost to potting, gasket, frame edges, adhesive, etc.
    tip_loss_mm: float = 0.8          # Ineffective media at each pleat tip/fold
    min_pleat_pitch_mm: float = 4.0   # Center-to-center minimum pleat spacing
    min_open_channel_mm: float = 2.0  # Remaining open channel after fold/tip allowance

    # Media pressure drop model
    flow_exponent: float = 1.25       # dp scales with velocity^n. Use 1.0-1.6 unless test curve proves otherwise.
    media_safety_factor: float = 1.15 # Covers lot variation, screen, adhesive, humidity, installation losses
    base_media_utilization: float = 0.92

    # Empirical pleat behavior model
    preferred_depth_to_pitch: float = 3.5
    severe_depth_to_pitch: float = 8.0
    lower_half_blinding_strength: float = 0.06
    min_depth_utilization: float = 0.45
    channel_penalty_strength: float = 0.035

    # Collapse/distortion model
    # Raise collapse_support_factor if the design has separators, hot-melt beads, wire backing,
    # stiff media, or other supports that prevent pleats from touching under vacuum.
    collapse_support_factor: float = 2.0
    collapse_reference_dp_pa: float = 250.0
    collapse_onset_index: float = 1.0
    collapse_strength: float = 0.35
    min_collapse_utilization: float = 0.35

    # Objective:
    # "balanced" = recommended: restriction + media efficiency + pleat risk
    # "lowest_dp" = lowest predicted restriction, often uses more media
    # "least_media_meeting_dp" = smallest media area that still meets dp_limit_pa
    objective: str = "balanced"
    top_n: int = 10


def sample_area_from_diameter_mm(diameter_mm: float) -> float:
    """Convenience function for circular Palas/bench media samples."""
    d_m = diameter_mm / 1000.0
    return pi * (d_m ** 2) / 4.0


def validate_inputs(x: FilterDesignInputs) -> None:
    required_positive = [
        "q_max_m3_h", "dp_limit_pa", "dp_test_pa", "q_test_m3_h",
        "test_media_area_m2", "frame_height_mm", "frame_width_mm",
        "max_pleat_depth_mm", "min_pleat_depth_mm", "pleat_depth_step_mm",
        "flow_exponent", "media_safety_factor", "base_media_utilization",
        "collapse_reference_dp_pa", "collapse_support_factor",
    ]
    for field in required_positive:
        if getattr(x, field) <= 0:
            raise ValueError(f"{field} must be > 0")

    if x.max_pleat_depth_mm < x.min_pleat_depth_mm:
        raise ValueError("max_pleat_depth_mm must be >= min_pleat_depth_mm")

    if x.objective not in {"balanced", "lowest_dp", "least_media_meeting_dp"}:
        raise ValueError("objective must be 'balanced', 'lowest_dp', or 'least_media_meeting_dp'")


def media_resistance_constant(x: FilterDesignInputs) -> float:
    """
    Calibrate K from the Palas/bench test.

    dp = K * media_velocity^n

    This requires the media area used in the test. If the Palas data is from a
    finished filter, use the tested filter's media area. If it is a flat media
    sample, use the exposed flat sample area.
    """
    q_test_m3_s = x.q_test_m3_h / 3600.0
    v_test_m_s = q_test_m3_s / x.test_media_area_m2
    return x.dp_test_pa / (v_test_m_s ** x.flow_exponent)


def required_effective_area_no_pleat_penalty_m2(x: FilterDesignInputs) -> float:
    """
    Effective media area required to meet dp_limit with no pleat blinding/collapse penalty.
    This is a useful lower-bound target area.
    """
    K = media_resistance_constant(x)
    q_max_m3_s = x.q_max_m3_h / 3600.0
    allowable_media_dp = x.dp_limit_pa / x.media_safety_factor
    max_media_velocity = (allowable_media_dp / K) ** (1.0 / x.flow_exponent)
    return q_max_m3_s / max_media_velocity


def _frange(start: float, stop: float, step: float):
    value = start
    while value <= stop + 1e-9:
        yield round(value, 6)
        value += step


def evaluate_candidate(
    x: FilterDesignInputs,
    pleat_count: int,
    pleat_depth_mm: float,
    K: float,
) -> Optional[Dict[str, float]]:
    usable_height_mm = x.frame_height_mm - 2.0 * x.edge_dead_zone_mm
    usable_width_mm = x.frame_width_mm - 2.0 * x.edge_dead_zone_mm

    if usable_height_mm <= 0 or usable_width_mm <= 0:
        raise ValueError("edge_dead_zone_mm is too large for the frame dimensions")

    pleat_pitch_mm = usable_width_mm / pleat_count
    if pleat_pitch_mm < x.min_pleat_pitch_mm:
        return None

    open_channel_mm = pleat_pitch_mm - 2.0 * x.tip_loss_mm
    if open_channel_mm < x.min_open_channel_mm:
        return None

    effective_leg_depth_mm = pleat_depth_mm - x.tip_loss_mm
    if effective_leg_depth_mm <= 0:
        return None

    gross_media_area_m2 = (
        2.0 * effective_leg_depth_mm * usable_height_mm * pleat_count
    ) / 1_000_000.0

    depth_to_pitch = pleat_depth_mm / pleat_pitch_mm

    # Deep/tight pleat penalty. This is the "lower-half blinding" term.
    excess_ratio = max(0.0, depth_to_pitch - x.preferred_depth_to_pitch)
    depth_utilization = 1.0 / (1.0 + x.lower_half_blinding_strength * excess_ratio ** 2)
    depth_utilization = max(x.min_depth_utilization, min(1.0, depth_utilization))

    # Narrow channel / maldistribution penalty. This adds pressure drop.
    channel_factor = 1.0 + x.channel_penalty_strength * excess_ratio ** 2

    q_max_m3_s = x.q_max_m3_h / 3600.0

    # Collapse is iterative: more dp causes more collapse, which reduces area and increases dp.
    collapse_utilization = 1.0
    predicted_dp_pa = 0.0
    effective_media_area_m2 = 0.0
    media_velocity_m_s = 0.0
    media_dp_pa = 0.0

    for _ in range(10):
        effective_media_area_m2 = (
            gross_media_area_m2
            * x.base_media_utilization
            * depth_utilization
            * collapse_utilization
        )
        if effective_media_area_m2 <= 0:
            return None

        media_velocity_m_s = q_max_m3_s / effective_media_area_m2
        media_dp_pa = K * (media_velocity_m_s ** x.flow_exponent) * x.media_safety_factor
        predicted_dp_pa = media_dp_pa * channel_factor

        collapse_index = (
            (predicted_dp_pa / x.collapse_reference_dp_pa)
            * (depth_to_pitch / x.preferred_depth_to_pitch) ** 2
            / x.collapse_support_factor
        )

        if collapse_index <= x.collapse_onset_index:
            new_collapse_utilization = 1.0
        else:
            overload = collapse_index - x.collapse_onset_index
            new_collapse_utilization = 1.0 / (1.0 + x.collapse_strength * overload ** 2)
            new_collapse_utilization = max(
                x.min_collapse_utilization,
                min(1.0, new_collapse_utilization),
            )

        # Damping prevents oscillation close to the onset threshold.
        collapse_utilization = 0.6 * collapse_utilization + 0.4 * new_collapse_utilization

    feasible = predicted_dp_pa <= x.dp_limit_pa

    collapse_index = (
        (predicted_dp_pa / x.collapse_reference_dp_pa)
        * (depth_to_pitch / x.preferred_depth_to_pitch) ** 2
        / x.collapse_support_factor
    )

    blinding_risk = min(
        1.0,
        max(
            0.0,
            (depth_to_pitch - x.preferred_depth_to_pitch)
            / max(1e-9, x.severe_depth_to_pitch - x.preferred_depth_to_pitch),
        ),
    )

    collapse_risk = min(
        1.0,
        max(0.0, (collapse_index - x.collapse_onset_index) / 4.0),
    )

    return {
        "feasible": feasible,
        "pleat_count": pleat_count,
        "pleat_depth_mm": pleat_depth_mm,
        "pleat_pitch_mm": pleat_pitch_mm,
        "open_channel_mm": open_channel_mm,
        "depth_to_pitch": depth_to_pitch,
        "gross_media_area_m2": gross_media_area_m2,
        "effective_media_area_m2": effective_media_area_m2,
        "area_utilization_pct": 100.0 * effective_media_area_m2 / gross_media_area_m2,
        "media_velocity_m_s": media_velocity_m_s,
        "media_dp_pa": media_dp_pa,
        "channel_factor": channel_factor,
        "predicted_dp_pa": predicted_dp_pa,
        "dp_margin_pa": x.dp_limit_pa - predicted_dp_pa,
        "dp_margin_pct": 100.0 * (x.dp_limit_pa - predicted_dp_pa) / x.dp_limit_pa,
        "depth_utilization_pct": 100.0 * depth_utilization,
        "collapse_utilization_pct": 100.0 * collapse_utilization,
        "blinding_risk_0_to_1": blinding_risk,
        "collapse_risk_0_to_1": collapse_risk,
    }


def optimize_filter(x: FilterDesignInputs) -> List[Dict[str, float]]:
    validate_inputs(x)
    K = media_resistance_constant(x)

    usable_width_mm = x.frame_width_mm - 2.0 * x.edge_dead_zone_mm
    max_pleat_count = int(floor(usable_width_mm / x.min_pleat_pitch_mm))

    candidates: List[Dict[str, float]] = []

    for pleat_count in range(1, max_pleat_count + 1):
        for pleat_depth_mm in _frange(
            x.min_pleat_depth_mm,
            x.max_pleat_depth_mm,
            x.pleat_depth_step_mm,
        ):
            candidate = evaluate_candidate(x, pleat_count, pleat_depth_mm, K)
            if candidate:
                candidates.append(candidate)

    if not candidates:
        raise RuntimeError("No candidates were possible. Check frame dimensions and pitch/channel limits.")

    max_area = max(c["gross_media_area_m2"] for c in candidates)

    for c in candidates:
        # Feasibility penalty makes non-compliant candidates sink to the bottom.
        fail_penalty = 10.0 if not c["feasible"] else 0.0
        c["balanced_score"] = (
            fail_penalty
            + c["predicted_dp_pa"] / x.dp_limit_pa
            + 0.25 * c["blinding_risk_0_to_1"]
            + 0.35 * c["collapse_risk_0_to_1"]
            + 0.04 * c["gross_media_area_m2"] / max_area
        )

    if x.objective == "lowest_dp":
        candidates.sort(key=lambda c: (not c["feasible"], c["predicted_dp_pa"], c["gross_media_area_m2"]))
    elif x.objective == "least_media_meeting_dp":
        candidates.sort(key=lambda c: (not c["feasible"], c["gross_media_area_m2"], c["predicted_dp_pa"]))
    else:
        candidates.sort(key=lambda c: (not c["feasible"], c["balanced_score"], c["predicted_dp_pa"]))

    return candidates[:x.top_n]


def print_results(x: FilterDesignInputs, results: List[Dict[str, float]]) -> None:
    K = media_resistance_constant(x)
    required_area = required_effective_area_no_pleat_penalty_m2(x)

    print("\nFILTER PLEAT OPTIMIZATION")
    print("-" * 72)
    print(f"Maximum airflow:                 {x.q_max_m3_h:.1f} m^3/h")
    print(f"Maximum allowed restriction:     {x.dp_limit_pa:.1f} Pa")
    print(f"Test restriction:                {x.dp_test_pa:.1f} Pa at {x.q_test_m3_h:.1f} m^3/h")
    print(f"Test media area:                 {x.test_media_area_m2:.4f} m^2")
    print(f"Calibrated media K:              {K:.3f}")
    print(f"Lower-bound required eff. area:  {required_area:.4f} m^2")
    print(f"Objective:                       {x.objective}")
    print("-" * 72)

    if not results:
        print("No results.")
        return

    columns: Tuple[Tuple[str, str, int], ...] = (
        ("feasible", "OK", 5),
        ("pleat_count", "Pleats", 7),
        ("pleat_depth_mm", "Depth mm", 9),
        ("pleat_pitch_mm", "Pitch mm", 9),
        ("depth_to_pitch", "D/P", 6),
        ("gross_media_area_m2", "Gross m2", 9),
        ("effective_media_area_m2", "Eff m2", 8),
        ("area_utilization_pct", "Util %", 8),
        ("predicted_dp_pa", "DP Pa", 8),
        ("dp_margin_pct", "Margin %", 9),
        ("blinding_risk_0_to_1", "Blind", 7),
        ("collapse_risk_0_to_1", "Collapse", 9),
    )

    header = " ".join(label.rjust(width) for _, label, width in columns)
    print(header)
    print("-" * len(header))

    for r in results:
        cells = []
        for key, _, width in columns:
            value = r[key]
            if isinstance(value, bool):
                text = "YES" if value else "NO"
            elif isinstance(value, int):
                text = f"{value:d}"
            else:
                text = f"{value:.2f}"
            cells.append(text.rjust(width))
        print(" ".join(cells))

    best = results[0]
    print("-" * 72)
    print("Recommended starting design:")
    print(f"  Pleat count:       {best['pleat_count']}")
    print(f"  Pleat depth:       {best['pleat_depth_mm']:.1f} mm")
    print(f"  Gross media area:  {best['gross_media_area_m2']:.4f} m^2")
    print(f"  Effective area:    {best['effective_media_area_m2']:.4f} m^2")
    print(f"  Predicted dp:      {best['predicted_dp_pa']:.1f} Pa")
    print(f"  DP margin:         {best['dp_margin_pa']:.1f} Pa")
    print(f"  Blinding risk:     {best['blinding_risk_0_to_1']:.2f} / 1.00")
    print(f"  Collapse risk:     {best['collapse_risk_0_to_1']:.2f} / 1.00")


def write_results_csv(path: str, results: List[Dict[str, float]]) -> None:
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)



# -----------------------------------------------------------------------------
# Streamlit web user interface
# -----------------------------------------------------------------------------
# Run locally with:
#     streamlit run filter_pleat_optimizer_streamlit.py
#
# Free hosting option:
#     1) Put this file in a GitHub repository.
#     2) Add a requirements.txt file containing: streamlit pandas
#     3) Deploy the repo at https://share.streamlit.io/

from io import StringIO
import pandas as pd
import streamlit as st


FIELD_GROUPS = (
    (
        "Required operating and test data",
        (
            ("q_max_m3_h", "Maximum airflow", "m^3/h", "Required maximum airflow through the finished filter."),
            ("dp_limit_pa", "Maximum allowed restriction", "Pa", "Maximum clean-filter restriction at maximum airflow."),
            ("dp_test_pa", "Bench / Palas test restriction", "Pa", "Measured restriction from the media or filter test."),
            ("q_test_m3_h", "Bench / Palas test airflow", "m^3/h", "Airflow used for the bench or Palas test."),
            ("test_media_area_m2", "Test media area", "m^2", "Media area used in the test."),
            ("frame_height_mm", "Frame height", "mm", "Overall filter frame height."),
            ("frame_width_mm", "Frame width", "mm", "Overall filter frame width."),
        ),
    ),
    (
        "Package and manufacturing limits",
        (
            ("max_pleat_depth_mm", "Maximum pleat depth", "mm", "Physical package depth available for the pleat."),
            ("min_pleat_depth_mm", "Minimum pleat depth", "mm", "Smallest depth to include in the search."),
            ("pleat_depth_step_mm", "Pleat depth step", "mm", "Increment used while searching pleat depths."),
            ("edge_dead_zone_mm", "Edge dead zone", "mm", "Area lost to potting, gasket, frame edges, adhesive, etc."),
            ("tip_loss_mm", "Tip / fold loss", "mm", "Ineffective media at each pleat tip or fold."),
            ("min_pleat_pitch_mm", "Minimum pleat pitch", "mm", "Minimum center-to-center pleat spacing."),
            ("min_open_channel_mm", "Minimum open channel", "mm", "Remaining open channel after fold/tip allowance."),
        ),
    ),
    (
        "Media and pleat behavior model",
        (
            ("flow_exponent", "Flow exponent", "", "Pressure drop scales with media velocity^n."),
            ("media_safety_factor", "Media safety factor", "", "Covers lot variation, adhesive, humidity, screen, and installation losses."),
            ("base_media_utilization", "Base media utilization", "", "Starting usable area fraction before pleat penalties."),
            ("preferred_depth_to_pitch", "Preferred depth-to-pitch", "", "Target depth/pitch ratio before penalties increase."),
            ("severe_depth_to_pitch", "Severe depth-to-pitch", "", "Depth/pitch ratio considered high risk."),
            ("lower_half_blinding_strength", "Lower-half blinding strength", "", "Penalty strength for deep/tight pleats."),
            ("min_depth_utilization", "Minimum depth utilization", "", "Lower bound for depth utilization."),
            ("channel_penalty_strength", "Channel penalty strength", "", "Added pressure-drop penalty for tight channels."),
        ),
    ),
    (
        "Collapse / distortion model and output",
        (
            ("collapse_support_factor", "Collapse support factor", "", "Raise if separators, hot-melt beads, backing mesh, or stiff media are used."),
            ("collapse_reference_dp_pa", "Collapse reference DP", "Pa", "Reference pressure for collapse risk model."),
            ("collapse_onset_index", "Collapse onset index", "", "Index where collapse utilization begins to reduce."),
            ("collapse_strength", "Collapse strength", "", "Penalty strength beyond collapse onset."),
            ("min_collapse_utilization", "Minimum collapse utilization", "", "Lower bound for collapse utilization."),
            ("top_n", "Number of results", "", "How many ranked designs to show."),
        ),
    ),
)

DEFAULT_INPUTS = FilterDesignInputs(
    q_max_m3_h=250.0,
    dp_limit_pa=500.0,
    dp_test_pa=40.0,
    q_test_m3_h=100.0,
    test_media_area_m2=1.0,
    frame_height_mm=300.0,
    frame_width_mm=250.0,
    max_pleat_depth_mm=55.0,
    min_pleat_depth_mm=10.0,
    pleat_depth_step_mm=1.0,
    objective="balanced",
    top_n=10,
)

RESULT_COLUMNS = {
    "feasible": "OK",
    "pleat_count": "Pleats",
    "pleat_depth_mm": "Depth mm",
    "pleat_pitch_mm": "Pitch mm",
    "open_channel_mm": "Open Channel mm",
    "depth_to_pitch": "D/P",
    "gross_media_area_m2": "Gross m^2",
    "effective_media_area_m2": "Eff. m^2",
    "area_utilization_pct": "Util %",
    "media_velocity_m_s": "Media Velocity m/s",
    "media_dp_pa": "Media DP Pa",
    "channel_factor": "Channel Factor",
    "predicted_dp_pa": "DP Pa",
    "dp_margin_pa": "DP Margin Pa",
    "dp_margin_pct": "Margin %",
    "depth_utilization_pct": "Depth Util %",
    "collapse_utilization_pct": "Collapse Util %",
    "blinding_risk_0_to_1": "Blind Risk",
    "collapse_risk_0_to_1": "Collapse Risk",
    "balanced_score": "Balanced Score",
}


def _number_input(key: str, label: str, unit: str, help_text: str, default_value):
    full_label = f"{label} ({unit})" if unit else label
    if key == "top_n":
        return st.number_input(
            full_label,
            value=int(default_value),
            step=1,
            min_value=1,
            help=help_text,
            key=key,
        )
    return st.number_input(
        full_label,
        value=float(default_value),
        step=1.0 if "mm" in unit or "Pa" in unit or "m^3/h" in unit else 0.01,
        format="%.6f",
        help=help_text,
        key=key,
    )


def read_streamlit_inputs() -> FilterDesignInputs:
    defaults = DEFAULT_INPUTS.__dict__
    values = {}
    for _, fields in FIELD_GROUPS:
        for key, _, _, _ in fields:
            values[key] = st.session_state[key]
    values["top_n"] = int(values["top_n"])
    values["objective"] = st.session_state["objective"]
    return FilterDesignInputs(**values)


def results_to_dataframe(results: List[Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    if df.empty:
        return df
    df = df.rename(columns=RESULT_COLUMNS)
    for col in df.select_dtypes(include=["float"]).columns:
        df[col] = df[col].round(4)
    if "OK" in df.columns:
        df["OK"] = df["OK"].map({True: "YES", False: "NO"})
    return df


def dataframe_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def main():
    st.set_page_config(page_title="Filter Pleat Optimizer", layout="wide")

    st.title("Filter Pleat Optimizer")
    st.write(
        "Enter the required variables, choose an objective, then calculate recommended pleat designs. "
        "This is an engineering screening tool; final values should be verified by prototype testing."
    )

    defaults = DEFAULT_INPUTS.__dict__

    with st.sidebar:
        st.header("Inputs")
        st.selectbox(
            "Optimization objective",
            options=("balanced", "lowest_dp", "least_media_meeting_dp"),
            index=("balanced", "lowest_dp", "least_media_meeting_dp").index(DEFAULT_INPUTS.objective),
            help="balanced is recommended for restriction, media efficiency, and pleat risk.",
            key="objective",
        )

        for group_title, fields in FIELD_GROUPS:
            with st.expander(group_title, expanded=(group_title == "Required operating and test data")):
                for key, label, unit, help_text in fields:
                    _number_input(key, label, unit, help_text, defaults[key])

    st.divider()
    st.write("Review the inputs in the sidebar, then click **Calculate** to run the optimizer.")

    if st.button("Calculate", type="primary"):
        try:
            with st.spinner("Optimizing pleat geometry..."):
                inputs = read_streamlit_inputs()
                results = optimize_filter(inputs)
                df = results_to_dataframe(results)
                best = results[0] if results else None

            if best:
                st.success("Calculation complete")
                st.subheader("Recommended starting design")
                metric_cols = st.columns(5)
                metric_cols[0].metric("Pleat count", f"{best['pleat_count']}")
                metric_cols[1].metric("Pleat depth", f"{best['pleat_depth_mm']:.1f} mm")
                metric_cols[2].metric("Predicted DP", f"{best['predicted_dp_pa']:.1f} Pa")
                metric_cols[3].metric("DP margin", f"{best['dp_margin_pa']:.1f} Pa")
                metric_cols[4].metric("Gross media area", f"{best['gross_media_area_m2']:.4f} m^2")

                K = media_resistance_constant(inputs)
                required_area = required_effective_area_no_pleat_penalty_m2(inputs)
                st.info(
                    f"Effective area: {best['effective_media_area_m2']:.4f} m^2 | "
                    f"Lower-bound required effective area: {required_area:.4f} m^2 | "
                    f"Blinding risk: {best['blinding_risk_0_to_1']:.2f} / 1.00 | "
                    f"Collapse risk: {best['collapse_risk_0_to_1']:.2f} / 1.00 | "
                    f"Calibrated media K: {K:.3f}"
                )

            st.subheader("Ranked designs")
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.download_button(
                label="Download results as CSV",
                data=dataframe_to_csv(df),
                file_name="filter_pleat_optimizer_results.csv",
                mime="text/csv",
            )

            with st.expander("Notes and assumptions"):
                st.markdown(
                    """
- Units are metric: airflow in m^3/h, pressure in Pa, dimensions in mm, area in m^2.
- Gross media area is approximated as `2 * pleat_depth * usable_height * pleat_count`, adjusted for fold-tip loss.
- Media pressure drop follows `dp = K * media_velocity^n`, calibrated from the bench or Palas test.
- The model includes empirical penalties for deep/tight pleat blinding, maldistribution, and collapse/distortion under vacuum.
                    """
                )

        except Exception as exc:
            st.error(f"Calculation problem: {exc}")
    else:
        st.info("The optimizer will not run until you click Calculate.")


if __name__ == "__main__":
    main()
