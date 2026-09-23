# Deterministic Candidate Shortlist

Measured 2026-09-23 on `dev/deterministic-shortlist`. The reproducible builder is
`analysis/shortlist_candidates.py`; its review artifact is
`analysis/candidate_shortlist.csv`. No API calls or hidden evaluator data were used.

## Before record

Inputs and SHA-256 hashes:

- `customer_profile.csv`: `e377baa2553a790f030a1d3411b3b5c5ca6316ccc7485badc9b08afa6da17d3f`
- `analysis/transition_priors.csv`: `e139beffbe97e20da36f7bb69da8da24cfb65caf8bd0eb8a7697b3a679372f2c`
- `data/dict_tariff.csv`: `74ef4242c9a6c46bd391d78251d6e3b9a408d913590c6efad7aa3957018be74c`

Configuration: shortlist size 32, no random seed, one current tariff and one ARPU
segment per candidate, all 21 published target tariffs, and the four public
channels. The hypothesis was that an evidence-backed, diversified 20-40 item set
would make later pilot selection tractable without prematurely selecting a final
portfolio.

The starting point was 63 populated current-tariff/ARPU audience cells and 451
observed historical transition priors. There was no reusable candidate generator,
resource-aware channel comparison, or validation layer before this milestone.

## Method

The builder enumerates every populated audience cell against every different
target tariff and every channel. It keeps unsupported combinations visible in the
universe but does not rank them until they have an evidence model. Each audience
is sorted by `ID_NUMBER` before applying the same per-campaign, contact, and
monetary ceilings as the public scorer. This matters for `digital_ads` (at most
4,545 contacts under the full budget) and `call` (at most 625).

For observed transitions, the builder combines the empirical-Bayes posterior
mean and standard deviation with the contacted customers' predicted ARPU,
channel multiplier, and contact cost. These fields are named value `proxy`
throughout the output. They are not campaign forecasts: the historical source
contains tariff changers but no exposure/nonchanger denominator, so it cannot
identify conversion probability or causal uplift.

Selection has four deterministic passes:

1. Seed coverage for each ARPU segment.
2. Seed coverage for each economically plausible channel.
3. Fill 24 exploitation positions by prior net value proxy under tariff,
   segment, and channel concentration caps.
4. Fill eight exploration positions by uncertainty value proxy.

Only one channel represents an origin/target/ARPU transition. Every candidate
must have a historical prior, at least 10 modeled contacts, a positive one-standard-
deviation upper net proxy, valid public filters, and a non-no-op target tariff.
Concentration caps are preferences; the selector can relax them deterministically
if needed to fill the requested review size.

## After measurements

- Candidate universe: 5,040 combinations.
- Combinations with historical priors: 1,804, exactly four channels for each of
  the 451 observed transition groups.
- Plausibly positive channel combinations: 1,338.
- Review shortlist: 32 distinct transitions.
- Coverage: 9 origin tariffs, 8 target tariffs, all 3 ARPU segments, all 4 channels.
- ARPU distribution: 8 HIGH, 12 MID, and 12 LOW candidates.
- Channel distribution: 2 push, 6 SMS, 12 digital ads, and 12 call candidates.
- Selection reasons: 3 ARPU coverage, 4 channel coverage, 17 expected-value,
  and 8 uncertainty candidates.
- Audience sizes range from 11 to 4,726; modeled contacts range from 11 to 4,545.
- Historical support ranges from 1 to 408 observations; median support is 6.5.

The low support is deliberate evidence for Milestone 5, not a claim of certainty.
Sparse and negative-mean exploration candidates remain hypotheses only when their
uncertainty interval leaves material upside. Bayesian pilot updates must decide
whether to keep them. This CSV must not be submitted as a 32-campaign strategy;
the final environment permits at most 10 campaigns and shares budget/contact
limits across the portfolio.

## Reproduction and checks

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m analysis.shortlist_candidates
```

Four tests cover resource ceilings, unsupported-prior handling, deterministic
selection, diversity, validation, and agreement between generated filters and
actual profile audiences. The generator was run twice and produced identical
bytes. Output SHA-256:
`05b04ab5cf84a40e97811fe5312a4bdd866d46dee3f2948e92f454447087badf`.

The unchanged starter agent was also rerun at seed 42 in UTF-8 console mode. Its
net result remained `-1,035,279`, matching the recorded baseline and confirming
that this milestone did not alter execution behavior in `agent.py`.

No baseline score comparison is claimed in this milestone because the shortlist
does not yet execute pilots or replace `agent.py`. The next controlled change is
to update these priors from public pilot observations and compare decisions under
fixed seeds.
