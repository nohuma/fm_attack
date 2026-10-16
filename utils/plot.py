"""Curves from the metrics written during training."""

import json
from pathlib import Path


def read_metrics(path: str | Path) -> list[dict]:
    """Rows of ``metrics.jsonl``. Blank lines are skipped."""
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def plot_curves(steps: list[int], series: dict[str, list[float]], save_path: str | Path) -> Path:
    """Save one figure. ``lr`` is drawn on its own axis. Other series share the top axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loss_keys = [key for key in series if key != "lr"]
    panels = 2 if "lr" in series else 1
    fig, axes = plt.subplots(panels, 1, figsize=(8, 3 * panels), sharex=True, squeeze=False)
    top = axes[0, 0]
    for key in loss_keys:
        top.plot(steps, series[key], label=key)
    top.set_ylabel("value")
    top.legend()
    top.grid(True, alpha=0.3)
    if "lr" in series:
        axes[1, 0].plot(steps, series["lr"], color="tab:orange", label="lr")
        axes[1, 0].set_ylabel("lr")
        axes[1, 0].grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("step")
    fig.tight_layout()
    destination = Path(save_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination


def plot_metrics(metrics_path: str | Path, save_path: str | Path | None = None) -> Path | None:
    """Plot every column in a metrics file. Returns the image path, or None if there are no rows."""
    path = Path(metrics_path)
    if not path.is_file():
        return None
    rows = read_metrics(path)
    if not rows:
        return None
    steps = [int(row["step"]) for row in rows]
    keys = [key for row in rows for key in row if key != "step"]
    ordered = list(dict.fromkeys(keys))
    series = {
        key: [float("nan") if row.get(key) is None else float(row[key]) for row in rows]
        for key in ordered
    }
    destination = Path(save_path) if save_path else path.with_name("curves.png")
    return plot_curves(steps, series, destination)
