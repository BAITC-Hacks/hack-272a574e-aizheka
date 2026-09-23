# Beeline Agent: Game Plan

Updated 2026-09-23. Milestone 5 is merged. The user authorized publishing this
snapshot to main. README.md contains current deployment instructions and results.
The portfolio experiment regressed and remains disabled; the tested Bayesian
strategy stays the default. Vercel AI helpers and offline tests exist but are
not connected to Agent.act. Corrected live requests still receive a billing 403.
Milestones 6-11 remain incomplete. The checklist below retains the original plan;
older provider and branch notes are historical, not current deployment guidance.

## Workspace

- Repository: `D:\hackalem\hack-272a574e-aizheka`
- Origin: `https://github.com/BAITC-Hacks/hack-272a574e-aizheka.git`
- Current branch: `dev/bayesian-adaptive-pilots`, created from updated `main`.
- Source package: `D:\hackalem\beeline_case_participants`
- Preserve the source package and copy working files into the repository.
- Put the Python environment and dependency caches on D: to conserve C: space.
- Use one `dev/NAME-OF-THE-CHANGE` branch per milestone. Merge or push only when
  the user explicitly requests it; never modify `main` implicitly.
- Requested development model: GPT-6 Astra. Application semantic provider: Jev.

## Review of completed work

- D: clone and milestone branches verified. Completed milestones 1-3 are on `main`.
- Participant guide, starter template, evaluation runner, and submission generator
  reviewed. Original participant files remain in their source folder.
- TypeSafe skill and the four user-provided OpenRouter pages reviewed.
- This plan, `.gitignore`, and an empty-key `.env.example` are committed on
  `game-plan` and pushed to `origin/game-plan` in commit `42f5cb5`.
- `.env.local` created with empty credential fields and excluded from Git.
  The credential pasted in chat was neither stored nor used; rotate it and
  enter its replacement locally. No API calls have been performed.
- All 16 participant-package files were copied into the repository without
  changing the source package. Their SHA-256 hashes match the originals.
- A Python 3.11 virtual environment was created at `.venv` on D:. The recorded
  dependencies `numpy==2.3.3` and `pandas==2.3.2` are installed and importable.
- The unchanged starter agent was measured on seed 42 and seeds 0-9. It was
  unprofitable in every run; full results are in `docs/BASELINE_REPORT.md`.
- `submission.csv` was generated twice with identical bytes and SHA-256.
- Public data, joins, missingness, tariff consistency, pilot observations, and
  scoring economics were profiled in `docs/DATA_ECONOMICS.md`.
- Sparse historical transition effects were converted into a deterministic,
  uncertainty-aware prior table in `analysis/transition_priors.csv`.
- A 5,040-row deterministic candidate universe now produces a validated,
  diversified 32-item review shortlist in `analysis/candidate_shortlist.csv`.
- A normal-normal Bayesian learner now updates candidate lift beliefs from public
  pilot outputs and preserves deployment resources with bounded exploration.

## Credentials and test spending

Store replacement credentials only in the ignored repository-root `.env.local`:
`OPENROUTER_API_KEY` for OpenRouter and, when needed, `OPENAI_API_KEY` for direct
OpenAI tests. `.env.example` contains empty fields and nonsecret defaults only.
The application loader is still to be implemented; it must preserve environment
variables supplied by the judging process. Git ignore rules are not encryption.

Never force-add credential files or include keys in logs, cache entries, reports,
request dumps, or submission archives. Check exclusion and staged files before
every commit or push. Build submission archives from an explicit file allowlist.
Keep raw provider payloads and local caches ignored; export sanitized metrics.

Default AI mode is offline with a zero API-spend allowance until live testing is
configured. Start with one small OpenRouter contract test, then a bounded fixture
set. Ask the user for the direct OpenAI key and an explicit test budget before
using their OpenAI balance. The stated $60 is available credit, not a spend target.
Enforce request/token caps, bounded retries and deadlines, reserve estimated
in-flight costs before parallel calls, and reconcile reported usage afterward.
Use provider-side limits where available; local estimates alone cannot guarantee
an exact dollar ceiling. Offline tests require neither key.

## Before and after method

Before each milestone, record the implementation, input hashes, configuration,
seeds, hypothesis, and measurements. Change one meaningful component, rerun the
same checks, and compare. Keep changes supported by correctness and evidence.

Compare the starter template, deterministic strategy, adaptive Bayesian strategy,
direct Jev features, and the verified cascade. Record per-seed net gain, median, worst result,
positive-run count, runtime, costs, contacts, pilots, and rejected campaigns.
Track API cost separately from the simulated campaign budget. Use held-out seeds
after tuning. Mock scores demonstrate mechanics, not future judging performance.

For the cascade, also compare cheap-model-only and stronger-model-only outputs
on the same labeled evidence fixtures. Measure unsupported claims accepted,
supported claims rejected, abstention, escalation rate, latency, and actual spend.
Do not use Jev as the sole judge of its own success. Include human-reviewed gold
labels and deterministic checks. Freeze model IDs, inputs, and prompts during a
comparison and preserve sanitized response snapshots for replay.

## Constraints

The guide requires `agent.py` with `Agent.act(env)`, 1-10 valid final campaigns,
at most 5,000 customers per campaign, 15,000 total contacts, a budget of 100,000,
and at most 20 pilots of 10-200 customers. Pilot costs and contacts count.
Uplift is counted once per customer using the best campaign; repeated contacts
still cost money. Optimize incremental ARPU minus communication costs.

Use public environment methods and supplied historical data. Agent code must
never access hidden effects or organizer-only state. Produce reproducible
`submission.csv` using the supplied generator.

The guide allows model calls within 10 minutes; the template says offline within
five minutes. Target under five minutes with bounded API calls and a complete
offline path until the organizers clarify the rule.

## Ordered checklist

### 1. Import and establish the environment

- [x] Clone onto D:, create `game-plan`, verify source folder, and read the guide.
- [x] Copy supplied Python files, guide, dictionaries, profile, and `data/` into
  the repository with relative paths preserved and harness files unchanged.
- [x] Record source hashes, create a D: Python environment, record dependencies,
  and verify the existing ignore rules cover generated files.

Done when the imported package is traceable and runners resolve their inputs.
The input CSV files total about 22 MB. Preserve the originals in the source folder.

### 2. Measure the baseline

- [x] Preserve the template and copy its initial behavior into `agent.py`.
- [x] Run `python local_eval.py`, `python local_eval.py --runs 10`, and
  `python make_submission.py` from the repository root; save outputs and metrics.
- [x] Repeat submission generation under the same configuration and compare.

Done: seed 42 net was `-1,035,279`; the 10-seed median was `-357,948`,
with `0/10` positive runs. The submission outputs were byte-identical.

### 3. Validate public data and economics

- [x] Check schemas, units, missing values, identifiers, joins, and tariff fields.
- [x] Estimate historical transition lift and uncertainty with shrinkage for
  sparse groups. History describes a different population and is only a prior.
- [x] Verify public pilot result fields and scoring semantics, including channel
  effects, eligibility, overlap, and budget allocation, before modeling them.

Done: the target and historical populations are disjoint; 451 of 1,260 transition
groups have evidence, with median support 11. Priors retain uncertainty, exact
duplicate transitions are removed, and campaign-conversion limitations are explicit.

### 4. Build the deterministic shortlist

- [x] Generate segment / target tariff / channel combinations in Python.
- [x] Filter invalid combinations, respect campaign sizes, and preserve diversity.
- [x] Start with 20-40 candidates ranked by expected net value and uncertainty;
  evaluate coverage rather than treating this shortlist size as optimal.

Done: 5,040 valid combinations reduce to 32 distinct evidence-backed transitions
covering all ARPU segments and channels. Resource ceilings and filters are tested;
the output is byte-reproducible. Value fields are historical ranking proxies, not
campaign-conversion forecasts. See `docs/SHORTLIST_REPORT.md`.

### 5. Add Bayesian learning and adaptive pilots

- [x] Choose an observation model supported by public pilot outputs; update lift
  estimates and uncertainty after each actual pilot result.
- [x] Select the next pilot and sample size by expected decision benefit,
  uncertainty, costs, and remaining resources. Reserve capacity for deployment.
- [x] Recheck limits before calls; handle small audiences and rejected pilots.
  Stop exploration when further information is unlikely to improve the decision.

Done: exact Gaussian updates change posterior means, uncertainty, and final
decisions. Exploration is capped at eight attempts, 1,600 contacts, and 20,000
budget while reserving deployment capacity. Rejections and complete pilot outage
have tested fallbacks. See `docs/BAYESIAN_PILOTS_REPORT.md`.

### 6. Optimize the final portfolio

- [ ] Select up to 10 campaigns using posterior value, uncertainty, overlap,
  channel costs, and resources remaining after pilots.
- [ ] Account for marginal uplift and repeated contact costs. Validate sizes,
  identifiers, filters, and deterministic ordering before returning campaigns.
- [ ] Define a conservative valid fallback for weak evidence. The guide requires
  at least one final campaign even when predicted gains are poor.

Done when the offline adaptive agent evaluates and produces valid submissions.

### 7. Integrate Jev through OpenRouter

Use the TypeSafe Python SDK with explicit `api_key` from `OPENROUTER_API_KEY`
and `base_url="https://openrouter.ai/api"`; the SDK targets `/v1/systemone`.
The native Decisions route used by the cookbooks is `/api/alpha/decisions`;
do not mix its path with the SDK base URL. Start with model `typesafe/jev-1.13`,
record the response model/provider and usage, and pin the tested SDK version.
`~typesafe/jev-latest` changes over time and is only for exploratory comparisons.
The SDK model-listing helper is incompatible with OpenRouter's model-list shape;
use its public Models API if discovery is needed.
[SDK contract](https://openrouter.ai/docs/guides/community/typesafe-sdk)
[Model alias](https://openrouter.ai/~typesafe/jev-latest/)

- [ ] Read current primitive/API contracts and define compact `CandidateState`
  with aggregate needs and tariff attributes, without raw customer identifiers.
- [ ] Ask independent questions together per candidate: `Choice` for offer-fit
  category, `Score` for consumption fit, `Score` for potential negative reaction,
  and `Noul` for evident needs mismatch. Bound concurrency across candidates.
- [ ] Keep these as auxiliary features, not measured conversion or ARPU effects.
  Retain distributions and Choice/Score confidence; Noul has no separate confidence.
  Calibrate influence on examples; low confidence reduces reliance rather than
  automatically rejecting candidates.
- [ ] Cache by semantic inputs, questions, model, and schema version. Reuse after
  pilots only when those inputs are unchanged. Python updates posteriors and plans.
- [ ] Implement missing-key, timeout, service-error, and invalid-response fallback.
  Validate expected answer IDs, types, finite scores/probabilities, allowed labels,
  and required confidence; missing verification fields must not imply approval.
  Never silently send an OpenRouter key to the direct TypeSafe or OpenAI endpoint.
  The guide only promises `OPENAI_API_KEY`; OpenRouter access at judging still
  requires confirmation even though a personal OpenRouter account is available.
- [ ] Make submission replay independent of changing live responses with an
  allowed versioned feature snapshot or documented deterministic offline mode.

Done when ablations establish whether Jev helps, outage tests pass, and replay is
reproducible. Live integration verification on this route requires an OpenRouter
key, not a separate direct TypeSafe key.

### 8. Add the evidence-verified cascade as a measured experiment

Apply the cookbook's draft -> verify -> escalate -> verify flow to campaign
hypotheses and analyst explanations. A cheaper generative model drafts; Jev
classifies evidence support; a stronger model gets at most one repair attempt.
Verify that attempt too. Missing evidence, abstention, errors, or a second failure
return to the deterministic path. The cookbook's confidence threshold is an
example to calibrate, not a correctness guarantee or a promised saving.
[Cascade recipe](https://openrouter.ai/docs/cookbook/evaluate-and-optimize/jev-verified-cascade)

- [ ] Assemble evidence packets from tariff descriptions, aggregate consumption,
  historical estimates, and public pilot results, with stable source IDs.
- [ ] Request a bounded proposal schema: supported filter values, target tariff,
  channel, evidence references, and a rationale. A proposal cannot execute tools.
- [ ] Check schemas, identifiers, numeric claims, and limits in Python before
  semantic verification. Known arithmetic should not consume inference calls.
- [ ] Ask Jev whether the factual rationale addresses the task and is supported,
  unsupported, or explicitly abstains. Future uplift remains a hypothesis that
  pilots must test, even when the supporting facts pass verification.
- [ ] Escalate only a potentially valuable unsupported or ambiguous proposal
  when time, evidence, and the configured API budget permit. Missing evidence
  alone is a reason to gather evidence or abstain, not to buy a larger model.
- [ ] Bound proposed additions to the shortlist. Python recomputes their value
  and chooses pilots; generated suggestions never become commands directly.
- [ ] Choose and pin draft/escalation models at the test stage. Direct OpenAI
  testing waits for the user's key and budget. No automatic cross-provider billing.

Keep direct Jev features separate from proposal verification so we can measure
each component. Do not add a generator call merely to reproduce computed facts.
If tariff text adds little information, retain the offline agent and template
explanations. Done when labeled fixtures and paired runs justify the added cost.

### 9. Validate proposed actions before pilot execution

Adapt the tool-gating recipe only where a generated proposal needs semantic
grounding. Code always checks budgets, contacts, valid tariffs/channels, campaign
size, and public method arguments immediately before `env.run_pilot`.
Jev may assess evidence consistency but cannot override a failed hard constraint.
An ambiguous proposal is deferred or discarded while the numerical strategy
continues; the timed agent must not wait for human approval mid-run.
Recheck evidence freshness and resource state before executing an accepted action.
[Tool-gating recipe](https://openrouter.ai/docs/cookbook/building-agents/gate-tool-calls-with-jev)

- [ ] Log sanitized reason codes, evidence hashes, model versions, and verdicts.
- [ ] Test invalid arguments, stale evidence, misleading rationales, and a Jev
  outage; assert that no invalid pilot executes and the fallback still progresses.

Done when model-generated suggestions cannot bypass deterministic constraints.

### 10. Test robustness and measure the after results

- [ ] Test budget/contact boundaries, overlaps, sparse groups, noisy observations,
  exhausted resources, cache invalidation, and service unavailability.
- [ ] Compare strategy variants on the same seeds and then held-out seeds.
- [ ] Document improvements, regressions, runtime, and remaining uncertainty.

Done when the before/after report is reproducible and claims match measurements.

### 11. Package and review

- [ ] Generate `submission.csv` twice under the documented mode and compare.
- [ ] Package `agent.py`, required local modules, submission, and dependencies;
  verify local-module and feature-cache submission rules with the organizers.
- [ ] Write setup instructions and architecture notes; review the final diff.
- [ ] Check the staged diff and submission allowlist contain no credentials,
  `.env.local`, raw API payloads, or local caches. Verify credentials stay untracked.
- [ ] Await explicit product-ready approval before merging/pushing to `main`.

Done when the exact submission package reproduces and meets public constraints.

## Architecture and scope

Public inputs -> deterministic shortlist -> optional semantic features ->
adaptive pilot learning -> constrained portfolio -> submission.

Optional proposal path: evidence -> cheap draft -> code validation -> Jev check ->
accept, one verified repair, or fallback -> numerical ranking and pilot selection.
The learner and optimizer remain Python; the cascade does not replace them.

Use the TypeSafe AI skill for integration. The first delivery is the working
Python agent and evaluation report; a frontend or deployment is not required.
Keep `Agent.act(env)` stable and introduce modules as the code needs them.

## Sources and unresolved deployment details

Reviewed local sources: `PARTICIPANT_GUIDE.md`, `agent_template.py`,
`local_eval.py`, and `make_submission.py` in the participant package.
Live TypeSafe references checked on 2026-09-23:

- [Composite scoring](https://docs.typesafe.ai/patterns/composite-scoring)
- [Confidence](https://docs.typesafe.ai/confidence)
- [Python SDK](https://docs.typesafe.ai/sdk/python)
- [Reranking cookbook](https://docs.typesafe.ai/cookbooks/rerank_typesafe)

Additional OpenRouter sources reviewed: the model alias, SDK, cascade, and gating
pages linked above. No cookbook benchmark is treated as evidence for this case.

Clarify runtime/network rules, judging OpenRouter credentials, feature-cache
packaging, and local-module submission before release. These do not block offline
work. Next implementation step after review: constrained final portfolio optimization.
