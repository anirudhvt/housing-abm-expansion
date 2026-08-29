"""Section 3.1 step 2 + Appendix A1
Adds new for sale stock whenever house to household ratio falls below target

Placeholder: new units priced at tract current pricing, rather than house price datasets
"""

from housing_abm.agents.first_time_buyer import FirstTimeBuyer
from housing_abm.agents.housing_unit import HousingUnit
from housing_abm.agents.institutional_investor import InstitutionalInvestor
from housing_abm.agents.renter import Renter
from housing_abm.agents.repeat_buyer import RepeatBuyer
from housing_abm.agents.small_landlord import SmallLandlord
from housing_abm.equations.investor_propensity import sample_landlord_incomes

HOUSEHOLD_TYPES = (Renter, FirstTimeBuyer, RepeatBuyer, SmallLandlord)


def run_construction(model):
    cfg = model.params["construction"]
    n_households = sum(1 for a in model.agents if isinstance(a, HOUSEHOLD_TYPES))
    total_units = model.total_housing_stock()
    target_units = cfg["target_house_to_household_ratio"] * n_households
    deficit = int(round(target_units - total_units))
    if deficit <= 0:
        return

    tract = model.tracts["tract_001"]
    for _ in range(deficit):  # create new houses to fill the gap, put on sale market
        unit = HousingUnit(model=model, tract_id="tract_001", quality=1.0)
        unit.price = tract.price_per_quality
        model.list_for_sale(unit)

def run_investor_replenishment(model):
    """Keeps investor population proportional to household base"""
    sim_cfg = model.params.get("simulation", {})
    n_households = sum(1 for a in model.agents if isinstance(a, HOUSEHOLD_TYPES))
 
    n_small_landlords = sum(1 for a in model.agents if isinstance(a, SmallLandlord))
    target_small_landlords = round(n_households * sim_cfg.get("small_landlord_fraction", 0.0))
    gap = target_small_landlords - n_small_landlords
    if gap > 0:
        income_cfg = model.params.get("income_distribution", {})
        propensity_cfg = model.params.get("investor_propensity_eq", {})
        landlord_incomes = sample_landlord_incomes(
            model.random_gen,
            n_landlords=gap,
            pool_size=n_households,
            decile_probs=propensity_cfg.get(
                "small_landlord_income_decile_probs", [0.1] * 10
            ),
            income_lognormal_mean=income_cfg.get("household_lognormal_mean", 8.6),
            income_lognormal_sigma=income_cfg.get("household_lognormal_sigma", 0.65),
        )
        for income in landlord_incomes:  # fill in gap with realistically-selected landlords
            age = int(model.random_gen.integers(30, 70))
            landlord = SmallLandlord(model=model, income=float(income), age=age, tract_id="tract_001")
            landlord.bank_balance = float(model.random_gen.lognormal(mean=11.5, sigma=0.6))
 
    n_institutional_investors = sum(
        1 for a in model.agents if isinstance(a, InstitutionalInvestor)
    )
    target_institutional_investors = round(
        n_households * sim_cfg.get("institutional_investor_fraction", 0.0)
    )
    for _ in range(target_institutional_investors - n_institutional_investors): #fill in gap with normal investors
        # same distribution as the initial cohort in AtlantaHousingModel.__init__;
        # these previously differed by a factor of ~5, so every replenished
        # investor entered far richer than the ones it replaced
        available_capital = float(model.random_gen.lognormal(mean=13.0, sigma=0.5))
        InstitutionalInvestor(
            model=model, available_capital=available_capital, tract_id="tract_001"
        )
