
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
# Simple graphical user interface
# -----------------------------------------------------------------------------
# Run this file with:
#     python filter_pleat_optimizer_ui.py
# The UI lets users enter the optimizer variables without editing this code.

import tkinter as tk
from tkinter import ttk, messagebox, filedialog


FIELD_GROUPS = (
    (
        "Required operating and test data",
        (
            ("q_max_m3_h", "Maximum airflow", "m³/h", "Required maximum airflow through the finished filter."),
            ("dp_limit_pa", "Maximum allowed restriction", "Pa", "Maximum clean-filter restriction at maximum airflow."),
            ("dp_test_pa", "Bench / Palas test restriction", "Pa", "Measured restriction from the media or filter test."),
            ("q_test_m3_h", "Bench / Palas test airflow", "m³/h", "Airflow used for the bench or Palas test."),
            ("test_media_area_m2", "Test media area", "m²", "Media area used in the test."),
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

RESULT_COLUMNS = (
    ("feasible", "OK", 52),
    ("pleat_count", "Pleats", 70),
    ("pleat_depth_mm", "Depth mm", 85),
    ("pleat_pitch_mm", "Pitch mm", 85),
    ("depth_to_pitch", "D/P", 70),
    ("gross_media_area_m2", "Gross m²", 85),
    ("effective_media_area_m2", "Eff. m²", 85),
    ("area_utilization_pct", "Util %", 80),
    ("predicted_dp_pa", "DP Pa", 80),
    ("dp_margin_pct", "Margin %", 80),
    ("blinding_risk_0_to_1", "Blind Risk", 85),
    ("collapse_risk_0_to_1", "Collapse Risk", 105),
)


class FilterPleatOptimizerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Filter Pleat Optimizer")
        self.geometry("1180x780")
        self.minsize(1020, 680)
        self.entries = {}
        self.current_results = []
        self.current_inputs = None

        self._configure_style()
        self._build_layout()
        self._populate_defaults()

    def _configure_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10))
        style.configure("Section.TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        style.configure("Card.TFrame", relief="solid", borderwidth=1)
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=(14, 8))
        style.configure("Treeview", rowheight=26)
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))

    def _build_layout(self):
        header = ttk.Frame(self, padding=(18, 14, 18, 8))
        header.pack(fill="x")
        ttk.Label(header, text="Filter Pleat Optimizer", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Enter the required variables, choose an objective, then calculate recommended pleat designs.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        main = ttk.PanedWindow(self, orient="horizontal")
        main.pack(fill="both", expand=True, padx=18, pady=(6, 18))

        input_outer = ttk.Frame(main, padding=(0, 0, 10, 0))
        main.add(input_outer, weight=1)

        canvas = tk.Canvas(input_outer, highlightthickness=0)
        scrollbar = ttk.Scrollbar(input_outer, orient="vertical", command=canvas.yview)
        self.input_frame = ttk.Frame(canvas)
        self.input_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.input_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self._build_input_fields()

        right = ttk.Frame(main)
        main.add(right, weight=2)

        controls = ttk.Frame(right, padding=(0, 0, 0, 10))
        controls.pack(fill="x")
        ttk.Label(controls, text="Optimization objective:").pack(side="left")
        self.objective_var = tk.StringVar(value="balanced")
        objective = ttk.Combobox(
            controls,
            textvariable=self.objective_var,
            state="readonly",
            width=28,
            values=("balanced", "lowest_dp", "least_media_meeting_dp"),
        )
        objective.pack(side="left", padx=(8, 16))
        ttk.Button(controls, text="Calculate", style="Accent.TButton", command=self.calculate).pack(side="left")
        ttk.Button(controls, text="Export CSV", command=self.export_csv).pack(side="left", padx=(8, 0))
        ttk.Button(controls, text="Reset", command=self._populate_defaults).pack(side="left", padx=(8, 0))

        self.summary = tk.Text(right, height=8, wrap="word", relief="solid", borderwidth=1, padx=12, pady=10)
        self.summary.pack(fill="x", pady=(0, 12))
        self.summary.configure(state="disabled")

        table_frame = ttk.Frame(right)
        table_frame.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(
            table_frame,
            columns=[key for key, _, _ in RESULT_COLUMNS],
            show="headings",
            selectmode="browse",
        )
        for key, label, width in RESULT_COLUMNS:
            self.tree.heading(key, text=label)
            self.tree.column(key, width=width, anchor="center", stretch=False)
        ybar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        xbar = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

    def _build_input_fields(self):
        for title, fields in FIELD_GROUPS:
            group = ttk.LabelFrame(self.input_frame, text=title, style="Section.TLabelframe", padding=12)
            group.pack(fill="x", pady=(0, 10))
            for row, (key, label, unit, help_text) in enumerate(fields):
                ttk.Label(group, text=label).grid(row=row, column=0, sticky="w", pady=4)
                var = tk.StringVar()
                entry = ttk.Entry(group, textvariable=var, width=14)
                entry.grid(row=row, column=1, sticky="ew", padx=(8, 6), pady=4)
                ttk.Label(group, text=unit, width=5).grid(row=row, column=2, sticky="w", pady=4)
                hint = ttk.Label(group, text=help_text, wraplength=260, foreground="#555")
                hint.grid(row=row, column=3, sticky="w", padx=(6, 0), pady=4)
                self.entries[key] = var
            group.columnconfigure(1, weight=1)

    def _populate_defaults(self):
        defaults = DEFAULT_INPUTS.__dict__
        for key, var in self.entries.items():
            var.set(str(defaults[key]))
        self.objective_var.set(DEFAULT_INPUTS.objective)
        self.current_results = []
        self.current_inputs = None
        self._clear_tree()
        self._set_summary("Enter values and click Calculate. Defaults match the original example values from the script.")

    def _read_inputs(self):
        values = {}
        for key, var in self.entries.items():
            raw = var.get().strip()
            if raw == "":
                raise ValueError(f"{key} is required")
            if key == "top_n":
                values[key] = int(float(raw))
            else:
                values[key] = float(raw)
        values["objective"] = self.objective_var.get()
        return FilterDesignInputs(**values)

    def calculate(self):
        try:
            inputs = self._read_inputs()
            results = optimize_filter(inputs)
            self.current_inputs = inputs
            self.current_results = results
            self._show_results(inputs, results)
        except Exception as exc:
            messagebox.showerror("Calculation problem", str(exc))

    def _clear_tree(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

    def _show_results(self, inputs, results):
        self._clear_tree()
        for r in results:
            values = []
            for key, _, _ in RESULT_COLUMNS:
                value = r[key]
                if isinstance(value, bool):
                    values.append("YES" if value else "NO")
                elif isinstance(value, int):
                    values.append(f"{value:d}")
                else:
                    values.append(f"{value:.2f}")
            self.tree.insert("", "end", values=values)

        if not results:
            self._set_summary("No results were returned.")
            return

        best = results[0]
        K = media_resistance_constant(inputs)
        required_area = required_effective_area_no_pleat_penalty_m2(inputs)
        summary = (
            "Recommended starting design\n"
            f"Pleat count: {best['pleat_count']}    "
            f"Pleat depth: {best['pleat_depth_mm']:.1f} mm    "
            f"Predicted DP: {best['predicted_dp_pa']:.1f} Pa    "
            f"DP margin: {best['dp_margin_pa']:.1f} Pa\n"
            f"Gross media area: {best['gross_media_area_m2']:.4f} m²    "
            f"Effective area: {best['effective_media_area_m2']:.4f} m²    "
            f"Lower-bound required effective area: {required_area:.4f} m²\n"
            f"Blinding risk: {best['blinding_risk_0_to_1']:.2f} / 1.00    "
            f"Collapse risk: {best['collapse_risk_0_to_1']:.2f} / 1.00    "
            f"Calibrated media K: {K:.3f}"
        )
        self._set_summary(summary)

    def _set_summary(self, text):
        self.summary.configure(state="normal")
        self.summary.delete("1.0", "end")
        self.summary.insert("1.0", text)
        self.summary.configure(state="disabled")

    def export_csv(self):
        if not self.current_results:
            messagebox.showinfo("Nothing to export", "Calculate results before exporting a CSV file.")
            return
        path = filedialog.asksaveasfilename(
            title="Save optimization results",
            defaultextension=".csv",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
            initialfile="filter_pleat_optimizer_results.csv",
        )
        if not path:
            return
        try:
            write_results_csv(path, self.current_results)
            messagebox.showinfo("Export complete", f"Results saved to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Export problem", str(exc))


if __name__ == "__main__":
    app = FilterPleatOptimizerApp()
    app.mainloop()
