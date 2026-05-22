"""
============================================================
Knowledge Graph Failure Localization in 5G Core Networks
============================================================

This script constructs and visualizes a heterogeneous knowledge graph
for failure localization in 5G core networks using telemetry data.

The framework models:
    - Network Functions (AMF, SMF, UPF, AUSF, UDM)
    - Metrics / telemetry features
    - Symptoms / anomalies
    - Failure events
    - Structural and failure propagation relationships

The graph supports:
    - Correlation-aware failure propagation
    - Root-cause visualization
    - Knowledge graph explainability

------------------------------------------------------------
Author: 238061
Project: KG-CAGNN for 5G Core Failure Localization
License: HiIoP Lab
------------------------------------------------------------
"""

# ============================================================
# Imports
# ============================================================

import os
import re
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ============================================================
# Configuration
# ============================================================

CSV_PATH = "training-data_c.csv"
OUTPUT_DIR = "output"

LABEL_COLUMN = "y_true(fc)"

RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)


# ============================================================
# Load Dataset
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(CSV_PATH)

print(f"Dataset shape: {df.shape}")

normal_mask = (
    df[LABEL_COLUMN]
    .astype(str)
    .str.lower()
    .eq("normal")
)

print(f"Normal samples : {normal_mask.sum()}")
print(f"Failure samples: {(~normal_mask).sum()}")


# ============================================================
# Helper Functions
# ============================================================

def component_from_name(name):
    """
    Extract 5G component name from metric/failure label.
    """

    prefix = str(name).split("_")[0].lower()

    if prefix.startswith("amf"):
        return "AMF"

    elif prefix.startswith("ausf"):
        return "AUSF"

    elif prefix.startswith("udm"):
        return "UDM"

    elif prefix.startswith("smf"):
        return "SMF"

    elif prefix.startswith("upf"):
        return "UPF"

    return prefix.upper()


def normalize_metric_name(col):
    """
    Clean and normalize telemetry metric names.
    """

    name = str(col)

    name = re.sub(r"_value$", "", name)

    parts = name.split("_")

    if len(parts) >= 2:
        name = "_".join(parts[1:])

    name = name.replace("statistics.", "")
    name = name.replace("per-core-stats.per-core-stat.", "")
    name = name.replace("load-average.", "load-avg-")

    name = name.replace(".", "-")
    name = name.replace("_", "-")

    return name


def symptom_from_label(label):
    """
    Convert failure labels into symptom categories.
    """

    label = str(label).lower()

    if "memory" in label:
        return "memory_stress"

    elif "vcpu" in label or "cpu" in label:
        return "cpu_overload"

    elif "interface-down" in label:
        return "interface_down"

    elif "interface-loss" in label:
        return "interface_loss"

    elif "bridge" in label:
        return "bridge_down"

    elif "packet" in label:
        return "packet_congestion"

    return "performance_degradation"


def compact_failure_name(label):
    """
    Generate compact failure name.
    """

    comp = component_from_name(label)
    symptom = symptom_from_label(label)

    return f"{comp}_{symptom}"


def short_label(name, max_len=18):
    """
    Format node labels for visualization.
    """

    name = str(name)

    name = name.replace("_", "\n")
    name = name.replace("-", "\n")

    if len(name) > max_len:
        return name[:max_len] + "..."

    return name


# ============================================================
# Select Representative Failure Classes
# ============================================================

print("\nSelecting representative failures...")

failure_counts = (
    df.loc[~normal_mask, LABEL_COLUMN]
    .value_counts()
)

selected_failures = []

seen = set()

for label in failure_counts.index:

    key = (
        component_from_name(label),
        symptom_from_label(label)
    )

    if key not in seen:
        selected_failures.append(label)
        seen.add(key)

    if len(selected_failures) >= 6:
        break

print("\nSelected Failures:")

for failure in selected_failures:
    print(f"  - {failure}")

main_root = (
    compact_failure_name(selected_failures[0])
    if selected_failures
    else None
)


# ============================================================
# Select Important Metrics
# ============================================================

print("\nSelecting important telemetry metrics...")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

numeric_cols = [
    c for c in numeric_cols
    if c != LABEL_COLUMN
]

value_cols = [
    c for c in numeric_cols
    if c.endswith("_value")
]

if not value_cols:
    value_cols = numeric_cols

normal_df = df.loc[normal_mask, value_cols]
failure_df = df.loc[~normal_mask, value_cols]

scores = {}

for col in value_cols:

    normal_std = normal_df[col].std(skipna=True)

    if pd.isna(normal_std) or normal_std == 0:
        normal_std = df[col].std(skipna=True)

    if pd.isna(normal_std) or normal_std == 0:
        continue

    score = abs(
        failure_df[col].mean(skipna=True)
        - normal_df[col].mean(skipna=True)
    ) / (normal_std + 1e-9)

    if np.isfinite(score):
        scores[col] = score

top_metrics = []

for component in ["amf", "ausf", "udm", "smf", "upf"]:

    component_cols = [
        c for c in scores
        if c.lower().startswith(component)
    ]

    component_cols = sorted(
        component_cols,
        key=lambda x: scores[x],
        reverse=True
    )[:4]

    top_metrics.extend(component_cols)

if len(top_metrics) < 12:

    top_metrics = sorted(
        scores,
        key=scores.get,
        reverse=True
    )[:20]

print(f"Selected {len(top_metrics)} important metrics.")


# ============================================================
# Build Knowledge Graph
# ============================================================

print("\nBuilding knowledge graph...")

G = nx.DiGraph()

node_type = {}


def add_node(node, node_category):
    """
    Add node with node type.
    """

    G.add_node(node)

    node_type[node] = node_category


# ------------------------------------------------------------
# Network Function Nodes
# ------------------------------------------------------------

for component in ["AMF", "AUSF", "UDM", "SMF", "UPF"]:
    add_node(component, "component")


# ------------------------------------------------------------
# Structural Dependencies
# ------------------------------------------------------------

structural_edges = [
    ("AMF", "AUSF"),
    ("AMF", "UDM"),
    ("AMF", "SMF"),
    ("SMF", "UPF"),
    ("AUSF", "UDM"),
    ("UDM", "AMF"),
]

for source, target in structural_edges:

    G.add_edge(
        source,
        target,
        relation="depends_on",
        edge_type="structural"
    )


# ------------------------------------------------------------
# Metric Nodes
# ------------------------------------------------------------

metric_nodes = []

for col in top_metrics:

    component = component_from_name(col)

    metric = normalize_metric_name(col)

    if len(metric) > 28:
        metric = metric[:25] + "..."

    metric_node = f"{component}:{metric}"

    if component not in G:
        add_node(component, "component")

    add_node(metric_node, "metric")

    metric_nodes.append(metric_node)

    G.add_edge(
        component,
        metric_node,
        relation="has_metric",
        edge_type="semantic"
    )


# ------------------------------------------------------------
# Failure and Symptom Nodes
# ------------------------------------------------------------

for label in selected_failures:

    component = component_from_name(label)

    symptom = symptom_from_label(label)

    failure = compact_failure_name(label)

    add_node(symptom, "symptom")

    add_node(failure, "failure")

    G.add_edge(
        failure,
        component,
        relation="affects",
        edge_type="failure"
    )

    G.add_edge(
        failure,
        symptom,
        relation="causes",
        edge_type="failure"
    )

    G.add_edge(
        symptom,
        component,
        relation="observed_at",
        edge_type="semantic"
    )


# ------------------------------------------------------------
# Metric-to-Symptom Relations
# ------------------------------------------------------------

for metric in metric_nodes:

    metric_lower = metric.lower()

    if any(k in metric_lower for k in ["memory", "available", "used"]):
        symptom = "memory_stress"

    elif any(k in metric_lower for k in [
        "cpu", "load", "user",
        "system", "idle", "iowait"
    ]):
        symptom = "cpu_overload"

    elif any(k in metric_lower for k in [
        "interface", "oper-status"
    ]):
        symptom = "interface_down"

    elif any(k in metric_lower for k in [
        "octets", "pkts", "packet"
    ]):
        symptom = "interface_loss"

    elif "bridge" in metric_lower:
        symptom = "bridge_down"

    else:
        symptom = "performance_degradation"

    if symptom in G:

        G.add_edge(
            metric,
            symptom,
            relation="indicates",
            edge_type="semantic"
        )

print(f"Graph nodes: {G.number_of_nodes()}")
print(f"Graph edges: {G.number_of_edges()}")


# ============================================================
# Graph Layout
# ============================================================

print("\nGenerating graph layout...")

rng = np.random.default_rng(RANDOM_SEED)

group_centers = {
    "component": np.array([0.0, 0.5]),
    "metric": np.array([2.5, 0.5]),
    "symptom": np.array([1.2, -2.0]),
    "failure": np.array([-1.6, -2.0]),
}

pos_init = {}

for node in G.nodes():

    node_group = node_type.get(node, "metric")

    jitter = rng.normal(0, 0.35, size=2)

    pos_init[node] = (
        group_centers[node_group] + jitter
    )

pos = nx.spring_layout(
    G,
    pos=pos_init,
    k=1.35,
    iterations=700,
    seed=RANDOM_SEED,
    weight=None
)

for node in pos:

    x, y = pos[node]

    node_group = node_type[node]

    if node_group == "component":
        pos[node] = (x * 1.8, y * 1.8 + 0.4)

    elif node_group == "metric":
        pos[node] = (x * 2.6 + 0.8, y * 2.4 + 0.2)

    elif node_group == "symptom":
        pos[node] = (x * 2.5 - 0.2, y * 2.5 - 0.8)

    elif node_group == "failure":
        pos[node] = (x * 2.8 - 0.8, y * 2.8 - 1.2)

for node in pos:

    pos[node] = (
        pos[node][0] + rng.uniform(-0.15, 0.38),
        pos[node][1] + rng.uniform(-0.28, 0.18)
    )


# ============================================================
# Visualization
# ============================================================

print("\nRendering graph...")

fig, ax = plt.subplots(figsize=(15, 9))

ax.set_title(
    "Knowledge Graph Failure Localization in 5G Core Network",
    fontsize=17,
    fontweight="bold",
    pad=18
)


# ------------------------------------------------------------
# Edge Styles
# ------------------------------------------------------------

edge_styles = {
    "structural": {
        "color": "#1d4f79",
        "style": "solid",
        "width": 1.8,
        "alpha": 0.85,
        "edges": []
    },
    "semantic": {
        "color": "#6f757c",
        "style": "solid",
        "width": 0.9,
        "alpha": 0.45,
        "edges": []
    },
    "failure": {
        "color": "#c62728",
        "style": "dashed",
        "width": 1.8,
        "alpha": 0.9,
        "edges": []
    }
}

for u, v, data in G.edges(data=True):

    edge_type = data.get("edge_type", "semantic")

    edge_styles[edge_type]["edges"].append((u, v))

for edge_type, style in edge_styles.items():

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=style["edges"],
        edge_color=style["color"],
        style=style["style"],
        width=style["width"],
        alpha=style["alpha"],
        arrows=True,
        arrowsize=12,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.10",
        ax=ax
    )


# ------------------------------------------------------------
# Node Styles
# ------------------------------------------------------------

node_styles = {
    "component": {
        "color": "#8ecae6",
        "edge": "#1f4e79",
        "size": 1500
    },
    "metric": {
        "color": "#d9d9d9",
        "edge": "#555555",
        "size": 900
    },
    "symptom": {
        "color": "#fdae61",
        "edge": "#b35806",
        "size": 1100
    },
    "failure": {
        "color": "#ef5350",
        "edge": "#a50f15",
        "size": 1250
    }
}

for node_group, style in node_styles.items():

    nodes = [
        n for n in G.nodes()
        if node_type.get(n) == node_group
    ]

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=nodes,
        node_color=style["color"],
        edgecolors=style["edge"],
        node_size=style["size"],
        linewidths=1.3,
        alpha=0.96,
        ax=ax
    )


# ------------------------------------------------------------
# Highlight Root Cause
# ------------------------------------------------------------

if main_root in G:

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=[main_root],
        node_color="#ef5350",
        edgecolors="#7f0000",
        node_size=2300,
        linewidths=3.0,
        alpha=1.0,
        ax=ax
    )

    x, y = pos[main_root]

    ax.text(
        x,
        y + 0.30,
        "★",
        fontsize=28,
        ha="center",
        va="center",
        color="#f6c400",
        fontweight="bold"
    )


# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------

labels = {}

for node in G.nodes():

    if node_type[node] == "metric":

        labels[node] = short_label(
            node.split(":", 1)[-1],
            max_len=16
        )

    else:
        labels[node] = node.replace("_", "\n")

nx.draw_networkx_labels(
    G,
    pos,
    labels=labels,
    font_size=7.5,
    font_weight="bold",
    ax=ax
)


# ------------------------------------------------------------
# Equation Annotation
# ------------------------------------------------------------

ax.text(
    0.01,
    0.98,
    r"Correlation-aware KG:  $A = A_s + A_f$     $X' = X + A_f\Delta X$",
    transform=ax.transAxes,
    fontsize=13,
    fontweight="bold",
    va="top",
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        edgecolor="#444444",
        alpha=0.95
    )
)


# ------------------------------------------------------------
# Legend
# ------------------------------------------------------------

legend_elements = [

    Patch(
        facecolor="#8ecae6",
        edgecolor="#1f4e79",
        label="Network function"
    ),

    Patch(
        facecolor="#d9d9d9",
        edgecolor="#555555",
        label="Metric / interface"
    ),

    Patch(
        facecolor="#fdae61",
        edgecolor="#b35806",
        label="Symptom / anomaly"
    ),

    Patch(
        facecolor="#ef5350",
        edgecolor="#a50f15",
        label="Failure event"
    ),

    Line2D(
        [0],
        [0],
        color="#1f4e79",
        lw=2,
        label=r"Structural dependency ($A_s$)"
    ),

    Line2D(
        [0],
        [0],
        color="#6c757d",
        lw=1.5,
        label="Semantic relation"
    ),

    Line2D(
        [0],
        [0],
        color="#d62728",
        lw=2,
        linestyle="--",
        label=r"Failure propagation ($A_f$)"
    ),

    Line2D(
        [0],
        [0],
        marker="*",
        color="w",
        markerfacecolor="#f6c400",
        markeredgecolor="#7f0000",
        markersize=15,
        label="Predicted root cause"
    )
]

ax.legend(
    handles=legend_elements,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.10),
    ncol=4,
    frameon=True,
    fontsize=10
)

ax.axis("off")

plt.tight_layout()


# ============================================================
# Save Outputs
# ============================================================

png_path = os.path.join(
    OUTPUT_DIR,
    "kg_failure_localization_clear.png"
)

pdf_path = os.path.join(
    OUTPUT_DIR,
    "kg_failure_localization_clear.pdf"
)

plt.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    pdf_path,
    bbox_inches="tight"
)

plt.show()

print("\nSaved outputs:")
print(f"  PNG : {png_path}")
print(f"  PDF : {pdf_path}")

print("\nDone.")
