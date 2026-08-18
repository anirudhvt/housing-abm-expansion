"""Section 3.7 + 6.2: Hard LTV limits (overriding mortgage_terms.yaml)
 and 'soft' LTI limits (a hard cap with some allowance of share of new mortgages permitted above it

"""
import yaml


def load_policies(model, policy_paths):  # grab policies from config
    model.policies = []
    for path in policy_paths or []:
        with open(path) as f:
            model.policies.extend(yaml.safe_load(f))
    _resolve_policy_scale(model)
    # invalidate the by-type index built in investor_restrictions
    model._policy_type_index = None
    _apply_hard_ltv_overrides(model)


def _resolve_policy_scale(model):
    """Convert scale-relative thresholds into unit counts for this run size.

    Unit thresholds written as absolute counts silently change what policy is
    being tested when the population changes: a 5-unit ownership cap is a
    severe restriction against 300 households and a nearly vacuous one against
    3,000, because institutional portfolios scale with the market. Since
    increasing the population is the main lever for reducing Monte Carlo
    noise, thresholds have to be expressed per unit of market size or the
    variance reduction comes at the cost of changing the intervention.

    A policy may specify `*_per_1000_households` instead of an absolute count;
    it is resolved here against the configured population. Absolute values are
    left untouched, so existing configs keep working.
    """
    households = model.params.get("simulation", {}).get("n_households")
    households = getattr(model, "_configured_n_households", None) or households
    if not households:
        return
    per_1000 = households / 1000.0

    def _resolve(container, relative_key, absolute_key, floor=1):
        if relative_key in container:
            container[absolute_key] = max(
                floor, int(round(container[relative_key] * per_1000))
            )

    for policy in model.policies:
        _resolve(policy, "max_units_per_1000_households", "max_units_absolute")
        for bracket in policy.get("brackets", []) or []:
            # a bracket may legitimately start at zero units
            _resolve(bracket, "min_units_per_1000_households", "min_units", floor=0)
            if bracket.get("max_units_per_1000_households") is None and (
                "max_units_per_1000_households" in bracket
            ):
                bracket["max_units"] = None
            else:
                _resolve(bracket, "max_units_per_1000_households", "max_units")


def _apply_hard_ltv_overrides(model):
    for policy in model.policies:
        if policy["type"] == "ltv_limit":
            loan_type = policy["loan_type"]
            model.mortgage_terms[loan_type]["max_ltv"] = policy[
                "hard_limit"
            ]  # edit params


def enforce_lti_policies(model):
    """
    run before ownership market
    clamp worst offenders down to hard limit
    exempt some random bids - clear at original LTI (soft_allowance)"""
    lti_policies = [p for p in model.policies if p["type"] == "lti_limit"]
    if not lti_policies:  # no lti policies
        return

    for policy in lti_policies:
        # grab details about the policy
        loan_type = policy["loan_type"]
        hard_limit = policy["hard_limit"]
        allowance = policy.get("soft_allowance", 0.0)

        # only affects certain loans
        affected = [
            b for b in model._ownership_bid_queue if b["agent"].LOAN_TYPE == loan_type
        ]
        if not affected:
            continue

        def implied_lti(bid):
            """Grab LTI of agent"""
            loan = bid["max_price"] - bid["down_payment"]
            annual_income = bid["agent"].income * 12.0
            return loan / annual_income if annual_income > 0 else 0.0

        over_limit = [b for b in affected if implied_lti(b) > hard_limit]
        if not over_limit:  # not affected
            continue

        # affected by policy
        model.random_gen.shuffle(over_limit)
        n_exempt = int(round(len(affected) * allowance))
        capped = over_limit[
            n_exempt:
        ]  # first n_exempt keep their original (higher) bid

        for bid in capped:
            max_loan = hard_limit * bid["agent"].income * 12.0
            bid["max_price"] = bid["down_payment"] + max_loan
