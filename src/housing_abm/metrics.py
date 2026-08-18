"""Outcome metrics for the policy experiments.

The original harness read every outcome from a single terminal snapshot: one
month's homeownership rate, one month's rental vacancy rate. With a few hundred
agents that snapshot is close to a single binomial draw, so its cross-seed
standard deviation is dominated by sampling noise that has nothing to do with
the policy under test. Averaging the same quantity over the measurement window
uses all the information the run already produced, and cuts the standard error
by roughly the square root of the number of effectively independent months.

Everything here reads from a per-month record collected during the run, so a
single simulation yields both the window mean and the within-run series needed
for stationarity diagnostics.
"""

from __future__ import annotations

import numpy as np

from housing_abm.agents.first_time_buyer import FirstTimeBuyer
from housing_abm.agents.institutional_investor import InstitutionalInvestor
from housing_abm.agents.repeat_buyer import RepeatBuyer
from housing_abm.agents.small_landlord import SmallLandlord

# outcomes reported for every policy comparison
PRIMARY_METRICS = (
    "homeownership_rate",
    "rental_vacancy_rate",
    "annual_appreciation_g",
)

# additional series used for validation and for reading the mechanism
SECONDARY_METRICS = (
    "mean_ltv_owner_occupier",
    "mean_lti_owner_occupier",
    "institutional_share_of_rentals",
    "investor_share_of_stock",
    "ftb_purchase_share",
    "median_price",
    "median_rent",
    "n_households",
)

ALL_METRICS = PRIMARY_METRICS + SECONDARY_METRICS


def observe(model) -> dict:
    """Snapshot every tracked outcome for the current month."""
    owner_occupiers = [
        a
        for a in model.agents
        if isinstance(a, (RepeatBuyer, FirstTimeBuyer))
        and a.house is not None
        and a.house.mortgage_principal > 0
    ]
    ltvs, ltis = [], []
    for a in owner_occupiers:
        price = a.house.price
        if price and price > 0:
            ltvs.append(a.house.mortgage_principal / price)
        annual_income = a.income * 12
        if annual_income > 0:
            ltis.append(a.house.mortgage_principal / annual_income)

    rental_stock = len(model.rental_units)
    institutional_units = sum(
        len(a.properties) for a in model.agents if isinstance(a, InstitutionalInvestor)
    )
    investor_units = institutional_units + sum(
        len(a.properties) for a in model.agents if isinstance(a, SmallLandlord)
    )
    stock = model.total_housing_stock()

    prices = [u.price for u in model.housing_units if u.price]
    rents = [u.rent for u in model.rental_units if u.rent]

    return {
        "homeownership_rate": model._homeownership_rate(),
        "rental_vacancy_rate": model._rental_vacancy_rate(),
        "annual_appreciation_g": model._appreciation_g(),
        "mean_ltv_owner_occupier": float(np.mean(ltvs)) if ltvs else None,
        "mean_lti_owner_occupier": float(np.mean(ltis)) if ltis else None,
        "institutional_share_of_rentals": (
            institutional_units / rental_stock if rental_stock else None
        ),
        "investor_share_of_stock": investor_units / stock if stock else None,
        "ftb_purchase_share": model.ftb_purchase_share_this_month(),
        "median_price": float(np.median(prices)) if prices else None,
        "median_rent": float(np.median(rents)) if rents else None,
        "n_households": float(model.n_household_agents()),
    }


def run_window(model, months: int) -> dict[str, list]:
    """Step the model `months` times, recording every outcome each month."""
    series: dict[str, list] = {name: [] for name in ALL_METRICS}
    for _ in range(months):
        model.step()
        row = observe(model)
        for name in ALL_METRICS:
            series[name].append(row[name])
    return series


def window_means(series: dict[str, list]) -> dict[str, float | None]:
    """Mean of each series over the window, ignoring months with no data."""
    out: dict[str, float | None] = {}
    for name, values in series.items():
        clean = [v for v in values if v is not None and not np.isnan(v)]
        out[name] = float(np.mean(clean)) if clean else None
    return out


def effective_sample_size(values) -> float:
    """Months of independent information in an autocorrelated series.

    n_eff = n * (1 - rho) / (1 + rho) for an AR(1) with lag-1 autocorrelation
    rho. Used to report honestly how much a window average actually buys over
    a single snapshot, rather than claiming the full month count.
    """
    x = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    n = x.size
    if n < 3:
        return float(n)
    x = x - x.mean()
    denom = float(x @ x)
    if denom <= 0:
        return float(n)
    rho = float(x[:-1] @ x[1:]) / denom
    rho = min(max(rho, -0.99), 0.99)
    return float(n * (1.0 - rho) / (1.0 + rho))
