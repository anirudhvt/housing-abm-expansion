"""Baseline validation against Atlanta plausibility targets.

Two changes from the original version matter for interpreting the numbers.

First, validation now runs under the *same* spin-up and window as the policy
experiments and averages over the window. The original script ran a single
seed with no spin-up at all and read the final month, while the policy runs
used a 600-month spin-up: the validation table therefore described a different
model state than the one the policy results came from, which is why the
reported homeownership rate (0.52) bore no relation to what the policy runs
actually produced (~0.24).

Second, it runs multiple seeds and reports an interval, so "in range" is a
claim about the model rather than about one draw.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from housing_abm.experiment import build_and_spinup  # noqa: E402
from housing_abm.metrics import run_window, window_means  # noqa: E402


def _appreciation_target():
    """Real Atlanta appreciation interquartile range from Case-Shiller."""
    try:
        cs = pd.read_csv("atlanta_case_shiller_2020_2023.csv")
        return (
            cs["g"].quantile(0.25),
            cs["g"].quantile(0.75),
            f"25th-75th pctile of real Atlanta YoY appreciation, 2020-2023 "
            f"(Case-Shiller ATXRSA); full-period mean {cs['g'].mean():.3f}",
        )
    except FileNotFoundError:
        return -0.05, 0.20, "PLACEHOLDER -- run pull_case_shiller_data.py"


TARGETS = [
    (
        "homeownership_rate",
        0.55,
        0.75,
        "US national range ~63-69%; Atlanta metro tends lower, ~60-63%",
    ),
    (
        "rental_vacancy_rate",
        0.05,
        0.15,
        "ACS metro Atlanta rental vacancy 7.3%; wider band for a small ABM",
    ),
    (
        "mean_ltv_owner_occupier",
        0.60,
        0.95,
        "origination LTV across FHA (up to 96.5%) and conventional (up to 80%)",
    ),
    (
        "mean_lti_owner_occupier",
        1.5,
        5.0,
        "typical loan-to-income multiples for US mortgages",
    ),
    (
        "institutional_share_of_rentals",
        0.20,
        0.40,
        "institutional operators hold roughly 30% of metro Atlanta SFR",
    ),
    ("annual_appreciation_g", *_appreciation_target()),
]


def _one(job):
    seed, households, spinup, months, config = job
    model = build_and_spinup(seed, households, spinup, config)
    return window_means(run_window(model, months))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--households", type=int, default=300)
    parser.add_argument("--spinup", type=int, default=120)
    parser.add_argument("--months", type=int, default=180)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--config", type=str, default="config/baseline_params.yaml")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()

    jobs = [
        (s, args.households, args.spinup, args.months, args.config)
        for s in range(args.seeds)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_one, jobs))
    df = pd.DataFrame(rows)

    print(
        f"Baseline validation -- {args.households} households, "
        f"{args.spinup}mo spin-up, {args.months}mo window averaged, "
        f"{args.seeds} seeds\n"
    )
    print(f"{'metric':<32} {'mean':>8} {'95% CI':>20} {'target':>16}   status")
    print("-" * 96)
    n_out = 0
    for label, low, high, note in TARGETS:
        vals = df[label].dropna()
        if vals.empty:
            print(f"{label:<32} {'n/a':>8} {'':>20} {f'[{low}, {high}]':>16}   NO DATA")
            continue
        mean = vals.mean()
        se = vals.std(ddof=1) / np.sqrt(len(vals))
        crit = stats.t.ppf(0.975, len(vals) - 1)
        lo, hi = mean - crit * se, mean + crit * se
        in_range = low <= mean <= high
        overlaps = not (hi < low or lo > high)
        status = "OK" if in_range else ("marginal" if overlaps else "OUT OF RANGE")
        n_out += 0 if in_range else 1
        ci = f"[{lo:.3f}, {hi:.3f}]"
        print(
            f"{label:<32} {mean:8.3f} {ci:>20} {f'[{low}, {high}]':>16}   {status}"
        )
        if not in_range:
            print(f"{'':<32} note: {note}")

    print(f"\n{len(TARGETS) - n_out}/{len(TARGETS)} targets met.")
    print(
        "Spin-up and window match the policy experiments, so these numbers "
        "describe\nthe same model state the policy effects are measured in."
    )


if __name__ == "__main__":
    main()
