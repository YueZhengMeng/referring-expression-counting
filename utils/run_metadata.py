import json
import subprocess
from pathlib import Path


def criterion_metadata(criterion):
    return {
        "cost_class": criterion.matcher.cost_class,
        "cost_point": criterion.matcher.cost_point,
        "cost_rep": criterion.matcher.cost_rep,
        "loss_weights": dict(criterion.weight_dict),
        "contrast_modes": {
            "rec": criterion.use_contrast_rec,
            "attribute": criterion.use_contrast_attr,
            "asd": criterion.use_contrast_asd,
            "image_hm": criterion.use_contrast_img_hm,
            "image": criterion.use_contrast_img,
        },
    }


def write_run_metadata(stats_dir, args, criterion):
    """Persist the exact objective and training settings beside run statistics."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"

    metadata = {
        "schema_version": 1,
        "git_revision": revision,
        "arguments": vars(args),
        "criterion": criterion_metadata(criterion),
    }
    path = Path(stats_dir) / "run_config.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)


def update_loss_stats(stats, criterion):
    """Accumulate detached raw and weighted loss components for one epoch."""
    for key, value in criterion.last_loss_breakdown.items():
        stats[key] = stats.get(key, 0.0) + float(value.item())


def format_loss_stats(stats, count):
    if not stats or count == 0:
        return ""
    means = ", ".join(
        f"{key}={value / count:.6f}"
        for key, value in sorted(stats.items())
    )
    return f"loss[{means}]"
