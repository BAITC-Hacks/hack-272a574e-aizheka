# Data and Campaign Economics

Measured 2026-09-23 on `dev/data-economics`. The reproducible profiler is
`analysis/profile_data.py`; its deterministic prior table is
`analysis/transition_priors.csv`. No API calls were made.

## Target audience

- `customer_profile.csv` has 23,441 rows and 23,441 unique customer IDs.
- `predicted_arpu` totals 150,641,084.25, matching the evaluation baseline.
- ARPU segments contain 2,738 LOW, 6,780 MID, 13,918 HIGH, and 5 missing.
- Median predicted ARPU is 6,237.89. There are 103 zero values and no negatives.
- All nonmissing segment labels match the documented thresholds. `call_segment`
  remains populated for 110 rows whose source call fields are missing, so the
  supplied segment should be treated as authoritative for those rows.
- Missingness is limited but decision-relevant: current tariff 95 rows (0.405%),
  data fields 110 (0.469%), contact fields 470 (2.005%), and base stations 162
  (0.691%). Paid campaigns should avoid rows whose expected lift cannot be
  computed from a known current tariff; zero predicted ARPU cannot offset cost.
- The largest current-tariff/ARPU cell has 4,726 customers, below the 5,000
  campaign cap. Coarser filters can exceed it: the largest tariff has 6,904
  customers and HIGH has 13,918. The scorer keeps the lowest sorted IDs when it
  caps a segment, so candidate value must be evaluated on the actual capped set.

## Population separation and joins

The target profile shares zero IDs with `change_tariff.csv`, `traffic.csv`, and
`arpu_monthly.csv`. This confirms that historical outcomes cannot be joined to
target customers and must be used as population-level priors.

Historical coverage:

- Tariff changes: 14,823 rows, 14,817 customers, one event date (`2026-10-01`).
- Traffic: 75,736 customer-month rows, 14,875 customers, April-September 2026.
- Monthly ARPU: 78,798 customer-month rows, 14,991 customers, April-September 2026.
- Every change customer appears in monthly ARPU; 14,727 appear in traffic.
- All 14,875 traffic customers appear in monthly ARPU.

Data-quality handling:

- Tariff changes contain six exact duplicate rows. The profiler removes them
  before estimating transition priors.
- Traffic has no duplicate customer-month keys.
- Monthly ARPU has 350 repeated customer-month keys: 84 exact duplicate extra
  rows and 266 key groups with conflicting values. A future monthly-ARPU feature
  must define an aggregation rule; this milestone does not silently choose one.
- All tariff codes found in the target and history exist in the tariff dictionary.

## Tariffs

The compact and descriptive tariff dictionaries agree on all five core fields.
There are 21 codes. Prices range from 0 to 12,528.6, with median 5,934.6.
`tariff_5`, `tariff_6`, `tariff_7`, and `tariff_8` have identical published
price, data, and minute allowances. These codes cannot be distinguished using
the supplied tariff attributes alone.

## Historical transition priors

The prior builder follows the mock environment's comparability rules while
retaining uncertainty:

1. Remove exact duplicate transition rows.
2. Exclude previous ARPU below 100 to avoid unstable percentage denominators.
3. Compute `(next_3m - previous_3m) / previous_3m` and clip it to `[-1, 3]`.
4. Group by origin tariff, destination tariff, and previous-ARPU segment.
5. Estimate a normal empirical-Bayes prior per ARPU segment and shrink each
   group mean according to its sampling variance and between-group variance.
6. Preserve sample size, raw mean and standard deviation, posterior mean and
   standard deviation, 95% interval, direction rates, and shrinkage weight.

After exact deduplication, 12,699 of 14,817 rows have previous ARPU of at least
100. They form 451 observed transition groups out of 1,260 possible ordered
cross-tariff/ARPU groups (35.8% coverage). Median support is 11 observations;
121 groups have fewer than 5 and 300 have fewer than 20. Shrinkage is therefore
necessary. All posterior values are finite and the output is deterministic.
The generated CSV SHA-256 is
`afedde753d76d12eff6072d753499547dbab6396fab9312dfe1a72649405a97e`.

Observed direction shares after filtering are 44.33% upsell, 43.17% downsell,
and 12.50% flat using the documented +/-10% boundary. These are descriptive
associations from historical changers, not causal campaign effects.

`destination_share_among_changers` is the fraction of observed tariff changes
from an origin/ARPU segment that went to a destination. The source has no
nonchanger denominator, exposure/control group, or campaign response label, so
this value must not be interpreted as campaign conversion probability. The mock
environment names the analogous quantity `conversion_rate`, but production logic
should learn response from pilots instead.

## Public pilot contract

`env.run_pilot` accepts valid public filters, target tariff, channel, and a
requested sample size. Requested size is clipped to 10-200 and may be reduced by
segment size, remaining contacts, or remaining budget. It returns actual sample
size, cost, noisy observed lift ratio and total, and remaining resources. The
agent does not receive sampled customer IDs.

The mock's standard deviation for a mean lift observation is `0.804 / sqrt(n)`:

| Pilot size | Standard deviation |
| ---: | ---: |
| 10 | 0.2542 |
| 30 | 0.1468 |
| 50 | 0.1137 |
| 100 | 0.0804 |
| 150 | 0.0656 |
| 200 | 0.0569 |

This known mock value is useful for testing Bayesian mechanics. Judging effects
differ, so later code should keep observation uncertainty configurable and avoid
tuning decisions to the mock effect table.

## Campaign economics

| Channel | Cost/contact | Conversion multiplier | Budget-only contact ceiling |
| --- | ---: | ---: | ---: |
| push | 0 | 0.50 | 15,000 contact cap |
| sms | 4 | 0.65 | 15,000 contact cap |
| digital_ads | 22 | 0.85 | 4,545 |
| call | 160 | 1.20, capped at probability 1 | 625 |

The scorer processes campaigns in order. Each campaign is capped at 5,000, then
by remaining total contacts, then by remaining money. Repeated contacts consume
cost and reach again. Customer uplift is deduplicated afterward by retaining that
customer's best campaign effect. Portfolio order and overlap therefore affect net
value even when campaign definitions are unchanged.

Net value is `predicted_arpu * realized lift ratio - contact cost`. Paid channels
need enough expected lift per contacted customer to clear that direct cost.
Push has no monetary cost but still consumes contact capacity and can crowd out a
better campaign. Campaigns with a target equal to the current tariff should be
excluded before piloting because the public contract does not prohibit them.

## Implications for later milestones

- Use the generated posterior means and uncertainty as priors, then let pilots
  update them. Do not treat historical destination shares as conversion rates.
- Exclude invalid/no-op transitions and calculate segment membership exactly as
  the scorer does. Compute value after all caps and prior campaign allocations.
- Penalize or avoid paid contact for unknown current tariff and zero baseline ARPU.
- Carry missingness indicators into semantic state instead of filling unknown
  consumption with zero.
- Keep the monthly ARPU duplicate-key issue explicit until an aggregation rule is
  justified. It is not needed for the first deterministic shortlist.
