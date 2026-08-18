# Why the original confidence intervals crossed zero

The first version of this study reported paired differences with 95% intervals
that spanned zero on almost every outcome. Three things were responsible, in
roughly this order of magnitude:

1. **Model bugs that severed the channels the policies act through.** Several
   policies could not have produced an effect no matter how many seeds were
   run.
2. **An experiment design that discarded most of the information each run
   produced**, and in particular a pairing scheme that did not actually pair.
3. **A baseline that was still drifting** when the outcomes were measured, so
   seeds were compared at different points on their own trajectories.

This document records what each of those was, how it was diagnosed, and what
replaced it.

---

## 1. Bugs that removed the policy transmission channels

### Resale listings never reached the market

`AtlantaHousingModel.queue_listing` recorded the seller of a resale listing in
`_resale_sellers` but never added the unit to `for_sale_units`. Since
`run_ownership_market` only ever iterates that registry, every resale listing
was flagged for sale and then never offered to a buyer.

Measured on the original code at 300 households over 300 months: **316 of 363
queued resale listings were orphaned, and 291 of 369 repeat buyers (79%) ended
the run permanently stuck holding a home they had listed and could not sell.**

This is the single most consequential defect, because the supply-release
channel is the entire mechanism by which the three financial penalties are
supposed to work: reduce effective yield -> raise the investor's probability of
selling -> release units to owner-occupiers. Step three did not exist. It also
disabled forced divestiture under the hard ownership cap, which lists excess
units through the same path.

Listing and rental-stock membership now go through single entry points
(`list_for_sale`, `add_rental_unit`, `drop_rental_unit`, `register_unit`) so
the registries cannot drift out of sync with unit state again.

### The investor decision logistic was fully saturated

EQ 9/12 compute the expected and effective yield; EQ 10/13 turn them into
monthly probabilities via `sigma(beta * Omega)^(1/12)`. Baptista et al.
calibrate `beta = 50`, and their `Omega` is monthly: `g` is a monthly
house-price growth expectation and `m/d` is a monthly mortgage payment over the
down payment.

This model computed `g`, `kappa` and `r_bar` as *annual* rates (EQ 4 compares a
three-month price average against the same average twelve months earlier) while
passing a monthly `m/d` alongside them. `Omega` came out roughly an order of
magnitude too large, and internally inconsistent.

At `beta = 50` and annual-scale `Omega`, across every market state tested:

| `g` (annual) | `Omega` | `P(buy)` | `P(buy)` with 5% purchase tax | `P(sell)` |
|---|---|---|---|---|
| -0.05 | 0.190 | 1.0000 | 0.9999 | 0.00022 |
|  0.00 | 0.250 | 1.0000 | 1.0000 | 0.00003 |
|  0.02 | 0.274 | 1.0000 | 1.0000 | 0.00002 |
|  0.05 | 0.310 | 1.0000 | 1.0000 | 0.00001 |
|  0.10 | 0.370 | 1.0000 | 1.0000 | 0.00000 |

Investors bought with probability 1.0000 every month regardless of market
conditions, essentially never sold, and **no financial penalty of any plausible
size could move them** -- a 5% purchase tax shifted `P(buy)` by less than
1e-4. Landlord portfolios could therefore only ever grow.

The paper attributed the null result on financial penalties to Atlanta's gross
rental yield providing "a cushion against tax surcharges". The actual cause was
a units mismatch that placed the logistic in a region where its derivative is
about 1e-5. On the monthly basis the equation is defined on, the same 5%
purchase tax moves `P(buy)` by ~0.007 per month.

`tests/test_investor_yield.py::test_omega_stays_inside_the_logistic_response_region`
now pins `|beta * Omega| < 2` across the realistic range and requires a visible
`P(buy)` response to the purchase tax, so this cannot regress silently.

### The purchase tax was charged to existing holdings

`compute_policy_cost` summed all three taxes and was passed to both
`expected_yield_buy` (EQ 9) and `effective_yield_sell` (EQ 12). A tax on
*acquiring* a home therefore also reduced the effective yield on homes an
investor already owned, raising their probability of selling. The scenario made
investors churn their portfolios rather than deterring acquisition, and
investor purchases went *up* under the tax.

Acquisition costs and holding costs are now separate. The purchase tax is
applied as the paper specifies it -- as an increase in the cash the investor
must put up for the same asset -- and is carried on the bid separately from the
down payment, since it buys no equity. (Folding it into the down payment would
shrink the mortgage principal, so the "tax" would have reduced the buyer's
borrowing instead of costing them anything.)

### The vacancy tax grace period was 90 months

`HousingUnit.day_vacant` is incremented once per model step, and a step is one
month, so it counts months. It was compared directly against
`vacancy_threshold_days: 90`, enforcing the 90-day grace period as 90 months --
seven and a half years. The tax essentially never bound.

### The progressive portfolio tax was a tax of zero

Brackets ran 10 / 50 / 100 / 350 units. The realised institutional portfolio
distribution is roughly 0-18 units, so nearly every holding sat in the 0%
bracket. The scenario nominally tested a progressive tax and actually tested no
tax, which is why it produced "the smallest effect ... not statistically
distinguishable from zero". Brackets now span the distribution the model
produces.

### The waiting period restricted twice as many agents as the other policies

`waiting_period.yaml` had `applies_to: [small_landlord, institutional]` while
all five other scenarios were institutional-only. The waiting period's larger
measured effect partly reflected a broader treatment rather than a better
mechanism. All six now target the same population.

### Registry and accounting defects

- `rental_units` retained units sold to owner-occupiers, so the denominator of
  the rental vacancy rate accumulated owner-occupied homes and the measured
  vacancy rate fell steadily toward zero for purely mechanical reasons.
- `for_sale_units` grew monotonically because sold units were never removed, so
  `houses_per_capita` -- the market-supply signal in EQ 6 -- measured
  cumulative construction rather than live listings.
- The initial rental stock was created with `owner = None` and never assigned.
  At the default configuration that left **~35% of the housing stock inert**:
  no agent could sell it, it never reached the sale market, and no investor
  policy could touch it. It is now distributed across small landlords and
  institutional investors to hit two calibration targets at once -- a ~30%
  institutional share of the rental stock, and 2-10 units per mom-and-pop
  landlord.
- `_liquidate_estate` demolished units a deceased agent merely rented.

---

## 2. The pairing did not pair

The original harness built two independent models per seed and ran each through
its own spin-up with the policy active from month one, then differenced them.
Passing the same seed to both does not make them comparable: the policy changes
an agent decision in the first month, every later random draw shifts, and after
a 600-month spin-up the two arms are two unrelated realisations that happen to
share a seed.

The diagnostic is the correlation between arms across seeds. Under working
common random numbers it should be close to 1. Measured on the original
results:

| policy | corr(baseline, policy) on homeownership |
|---|---|
| ownership_cap_hard | +0.11 |
| ownership_cap_soft | +0.27 |
| portfolio_tax | +0.59 |
| purchase_tax | +0.25 |
| vacancy_tax | +0.14 |
| waiting_period | **-0.06** |

With correlation near zero, `sd(difference) = sqrt(2) x sd(one arm)`:
differencing *added* variance rather than removing it. The waiting-period
figures show exactly this -- `sd(base) = 0.033`, `sd(diff) = 0.040`, against
`sqrt(2) x 0.033 = 0.046`.

### Shared spin-up, then fork

Each seed is now spun up **once**, with no policy, and the resulting model is
deep-copied into each arm. Both begin from a bit-identical microstate: same
agents, same balances, same houses, same prices. The fork was verified to
produce bit-identical trajectories when both copies are stepped without an
intervention.

This also halves the compute, since the spin-up runs once per seed instead of
once per arm.

### Split RNG streams

Even after forking, the arms desynchronise as soon as the policy changes how
many random numbers are consumed -- so the two arms of a seed would see
different interest-rate paths and different birth/death sequences, noise
unrelated to the policy that enters the paired difference in full.

The model now spawns independent substreams from one `SeedSequence`:

- `random_gen` -- market and behavioural draws
- `rng_macro` -- the exogenous interest-rate path
- `rng_demography` -- births, deaths, heir selection

The macro and demographic paths are therefore bit-identical across arms for a
given seed whatever the policy does.

Arm correlation after these changes: **+0.5 to +0.99**, with achieved variance
reduction up to 50x versus unpaired sampling. Both numbers are reported for
every effect so the reader can check the pairing is working rather than assume
it.

---

## 3. Estimating outcomes from a single month

Every outcome was read from one terminal snapshot. Over a few hundred agents,
one month's homeownership rate is close to a single binomial draw: at
`p = 0.5` and `n = 300`, its standard deviation is ~2.9 percentage points,
against policy effects of 1-3 points.

Outcomes are now averaged over the measurement window. How much this buys
depends on the autocorrelation of the series, which the design diagnostics
report as an effective sample size:

| outcome | `n_eff` of 240 months | SE reduction vs a snapshot |
|---|---|---|
| rental vacancy rate | 59 | ~7.7x |
| first-time-buyer purchase share | 32 | ~5.6x |
| annual appreciation | 4.7 | ~2.2x |
| homeownership rate | 2.7 | ~1.6x |
| institutional share of rentals | 3.2 | ~1.8x |

The stocks are highly persistent and gain little; the flows are nearly white
month to month and gain a great deal. This is worth knowing before choosing
which outcome to build an argument on.

### Measure the flow, not only the stock

`ftb_purchase_share` -- first-time buyers as a share of completed purchases --
was added because it is the quantity every one of these policies acts on most
directly: who wins the bidding on a given listing. The homeownership *stock*
only moves as fast as that flow accumulates, so it responds later and more
weakly to the same intervention.

It is computed by pooling numerator and denominator over the window, not by
averaging monthly ratios. A handful of homes change hands per month, so the
monthly share is a ratio of very small integers -- 0/2, 1/3, undefined -- and
its mean is both noisy and biased toward low-transaction months. Pooling first
raised the arm correlation on this metric from a value that decayed to **-0.53**
at long windows to a stable **+0.84 to +0.92**.

### Choosing the measurement window

Longer windows average away more within-run noise, but the two arms drift apart
as they run, so the CRN correlation decays. The standard error of the paired
effect is minimised somewhere in between rather than at the longest window.
`scripts/diagnose_design.py` sweeps window length and reports both, so the
choice is made from measurement rather than convention.

---

## 4. The baseline was not stationary

If the model is still drifting when outcomes are measured, each seed is sampled
at a different point on its own trajectory and that spread enters the
cross-seed standard deviation as noise. It also means a validation table
computed at one spin-up length says nothing about the model state a policy
experiment at a different spin-up length is measuring -- which is why the
original paper reported a validated homeownership rate of 0.52 while the policy
runs were producing ~0.24.

Four sources of drift were found and removed.

**The initial age distribution was wrong.** Ages were drawn `U(22, 65)` against
a stationary distribution -- the one implied by the model's own mortality
hazard -- with median 54 and a tail past 88. The initial population had nobody
near the mortality midpoint, so deaths stayed near zero for several decades
while births continued; the population grew, hit the hazard wall together as one
cohort, and collapsed. That transient is longer than the whole simulation, so
**no spin-up length reached a steady state.** Ages are now drawn from the
stationary distribution, which starts the model where the spin-up was supposed
to end.

Note that the mortality curve was already correctly calibrated: at
`mortality_scale = 1.0` the implied mean household lifetime is 53.3 years,
giving a death rate of 1.876%/yr, exactly the configured birth rate. Baptista
et al. enforce the same condition by scaling the mortality pdf. Only the
*initial condition* was inconsistent with it.

**Rent-burden exits were an uncompensated population sink.** Households pushed
out by rent burden left permanently, at ~0.4%/yr on top of mortality, tipping
the birth/death balance negative. Real metros replace out-migrants with
in-migrants; they are now replaced one-for-one.

**Institutional investors had no capital outflow at all.** Household agents are
drained by EQ 2 consumption and small landlords by the same rule, but every
dollar of institutional net rental income compounded into buying power forever.
Their share of the rental stock climbed monotonically -- 30% to ~60% over a long
run -- so the ownership distribution never settled. Real single-family rental
operators are largely REIT-structured and must distribute the bulk of taxable
income, so a payout ratio is both the realistic and the stabilising choice.

**Entrants were drawn richer than the initial population.** The initial cohort
came from `lognormal(8.6, 0.65)` (median ~$5,430/mo, matching the ACS metro
Atlanta median) while later entrants came from `lognormal(8.9, 0.55)` (median
~$7,330/mo). As the initial cohort was replaced, the population drifted steadily
richer and homeownership drifted up with it -- a trend in every reported outcome
that had nothing to do with any policy. One distribution now serves both.

A fifth change was needed for a different reason: vacated units were re-let in
the same month they were vacated, so turnover produced no vacancy at all and
the rental vacancy rate sat at ~0 in every run. That both mis-stated the level
against the ACS figure and **censored one of the three reported outcomes at a
floor**, where a policy can only move it in one direction. A one-month
frictional void between tenancies against an ~18-month average lease puts
turnover vacancy in the ACS range.

`scripts/diagnose_design.py` regresses each monthly series on the month index
and reports which outcomes are still drifting, so any residual trend is stated
rather than assumed away.

---

## 5. Population size, and why the matching loop had to be rewritten

Population is the most direct lever on sampling noise, and the study was pinned
at 300 households by runtime: the model scaled roughly quadratically, so 1200
households cost about 11x what 300 did.

The cost was concentrated in Phase 1 of the double auction, where every
remaining bid scanned every remaining offer -- and ran the per-unit policy
predicate for each `(bid, offer)` pair, tens of millions of times per run.
Offers are now sorted by price once per round with a running argmax of the
preference key, so a bid's preferred offer is a binary search plus one array
lookup (`housing_abm/markets/offer_book.py`). Ties in the preference key are
common, because the construction sector emits units at exactly quality 1.0, and
are broken by a per-round random jitter that reproduces the uniform random
tie-break the original did explicitly. Unit-level policy eligibility depends
only on the agent's *class*, so it is resolved once per class per round rather
than once per pair.

At 1200 households this cut a 30-step profile from 125s to 10s.

### Policy thresholds must be absolute, not per-capita

An early attempt expressed ownership caps and tax brackets per 1000 households,
on the assumption that thresholds should scale with market size. Measurement
showed the opposite: because `institutional_investor_fraction` ties the *number*
of investors to the household count, the rental stock and the investor count
grow together and the mean institutional portfolio stays near 7-12 units at
every population from 300 to 2400 households.

A per-capita cap therefore gets looser as the population grows. At 1200
households it stopped binding entirely, and the measured effect was **exactly
zero on every outcome** -- a result that would have looked like a substantive
finding rather than a specification error. Absolute unit counts are the
scale-invariant form here, and the configs say so.

---

## 6. Reporting

Every effect is reported with:

- the paired 95% t interval **and** a bootstrap interval over the same paired
  differences; material disagreement means the normal approximation is not safe
  for that outcome
- the arm-to-arm correlation and achieved variance reduction, so the reader can
  verify the pairing worked
- **Holm-Bonferroni adjusted p-values across the full grid** of six policies x
  five outcomes. Thirty tests against one baseline need family-wise error
  control before any single result is called significant; Holm is uniformly
  more powerful than Bonferroni and assumes nothing about independence, which
  matters because the arms share a spun-up state
- the **minimum detectable effect** at 80% power, so a null reads as a bound on
  the effect rather than as evidence of its absence, and the seed count that
  would be needed to resolve the observed point estimate

Purchases are also broken out by buyer class, so the incidence question -- who
actually buys the homes a policy diverts from institutional investors -- is
read directly from the data rather than inferred from the homeownership stock.
