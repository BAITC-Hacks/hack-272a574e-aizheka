# Bayesian Adaptive Pilots

Measured 2026-09-23 on `dev/bayesian-adaptive-pilots`. The implementation is
`adaptive_pilots.py`, called by `agent.py`. No API calls or cloud compute were
used.

## Before record

Milestone 4 produced 32 deterministic candidate hypotheses, but `agent.py` still
used the starter policy: six fixed SMS pilots, raw observed ratios, and no
uncertainty-aware updates. On seeds 0-9 it had median net value `-357,948`, worst
value `-1,019,431`, and `0/10` positive runs. Seed 42 was `-1,035,279`.

The hypothesis for this milestone was that weak historical priors, exact Bayesian
updates from public pilot results, and a value-of-information pilot policy would
avoid large deployments caused by noisy point estimates while preserving enough
budget and contacts for final campaigns.

## Observation model

Each candidate's campaign lift ratio is represented by a normal belief. Historical
transition lift is only a prior signal because the source contains changers but
no campaign conversion denominator. The builder therefore multiplies the
historical signal by a conservative response scale of 0.10 and applies a standard
deviation floor of 0.12. Sparse historical groups receive wider priors.

The public mock contract models a pilot mean with standard deviation
`0.804 / sqrt(n)`. For observed ratio `y`, prior mean `m`, prior variance `v`, and
pilot variance `r`, the update is the normal-normal posterior:

```text
posterior variance = 1 / (1/v + 1/r)
posterior mean     = posterior variance * (m/v + y/r)
```

The model consumes only public `observed_lift_ratio` and actual `n_customers`.
It never reads sampled IDs, hidden effects, or evaluator internals.

## Pilot policy

For each feasible candidate, the planner computes the expected value of deciding
after another pilot versus deciding now. The calculation integrates the positive
part of a normal posterior around the channel-specific break-even lift. Pilot net
value is included because pilot contacts count in the score.

Sample size targets posterior standard deviation 0.06, then is clipped by:

- the public 10-200 pilot range;
- actual audience size and current environment resources;
- 1,600 exploratory contacts and 20,000 exploratory budget total;
- 8,000 maximum spend for one pilot, limiting a call pilot to 50 contacts;
- reserves of 10,000 contacts and 60,000 budget for deployment;
- at most eight attempts, two pilots per candidate, and two per audience cell.

After every accepted observation, all candidate utilities are recomputed. A
candidate rejection is isolated and planning continues. Attempt count advances
even when the environment rejects a call, preventing an outage loop. If all
pilots fail, the agent still returns one valid free push fallback.

The provisional final selector uses only measured candidates, permits one final
campaign per audience cell, and simulates remaining resource consumption. An
ordinary deployment must have positive net value at the lower edge of a central
90% credible interval. A campaign that would consume more than half the remaining
budget must clear the lower edge of a central 95% interval. Full overlap-aware
portfolio optimization remains Milestone 6.

## Before and after results

Fixed seeds 0-9:

| Metric | Starter | Adaptive |
| --- | ---: | ---: |
| Median net value | -357,948 | 488,485 |
| Worst net value | -1,019,431 | 28,584 |
| Best net value | -76,493 | 489,719 |
| Positive runs | 0/10 | 10/10 |

Adaptive seed results were `488,503`, `489,719`, `196,745`, `489,119`,
`488,889`, `28,707`, `488,467`, `489,527`, `28,584`, and `487,626`.

Held-out seeds 10-29 were frozen until after the commitment-sensitive risk gate:

- median: `194,221`;
- minimum: `319`;
- maximum: `637,120`;
- positive runs: `20/20`.

Seed 42 finished at `+489,248`, with eight pilots, one final SMS campaign, 2,914
total contacts, 24,200 communication cost, and 82,680 budget plus 13,806 contacts
remaining after exploration. The final submission campaign was
`tariff_8` MID -> `tariff_10` by SMS.

These scores are evidence about mechanics on the supplied mock only. Hidden
effects differ, so no mock tariff or observed outcome is hardcoded.

## Rejected variant

A controlled variant forced small confirmation pilots whenever a measured
candidate looked profitable but had not cleared its confidence bound. It made
pilot sequences visibly differ by seed, but displaced broader exploration and
fell to `3/10` positive runs with median `-25,172`. The variant was removed. The
retained planner still recomputes decisions after every update; in these mock
runs, fresh candidate information consistently exceeded the value of revisits.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe local_eval.py --runs 10
.\.venv\Scripts\python.exe local_eval.py
.\.venv\Scripts\python.exe make_submission.py
```

Thirteen tests cover posterior movement, uncertainty reduction, information value,
observation-sensitive final decisions, resource reserves, small audiences,
call-cost caps, rejected pilots, universal rejection fallback, real mock
integration, and the deterministic shortlist. `submission.csv` was generated
twice with identical SHA-256:
`2c38b5a2e2b2ba26cffa8744de434f34555f346e1bd4c5b2164705ca74b2f53d`.

No GCP compute is warranted: the complete tests and fixed-seed runs finish locally
in seconds on the available 6-core CPU. Large robustness sweeps can be revisited
in Milestone 10 if local runtime becomes material.
