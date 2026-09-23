# Starter Baseline Report

Measured 2026-09-23 on branch `dev/baseline-evaluation` before any strategy
changes. No API calls were made.

## Configuration

- Python 3.11.3
- NumPy 2.3.3
- pandas 2.3.2
- Agent: exact working-tree copy of `agent_template.py`
- Agent/template SHA-256:
  `739ec30073deef0767e64b814043dd6190ed85ed0eb70ec15dfe461cbeedf62f`
- Evaluation commands used `PYTHONUTF8=1` because the first attempt inherited
  Windows `cp1252` and raised `UnicodeEncodeError` while printing Russian text.
  The supplied evaluation harness was not modified.

## Seed 42

Command: `python local_eval.py`

| Metric | Result |
| --- | ---: |
| Status | FAIL |
| Baseline ARPU | 150,641,084 |
| Gross ARPU lift | -1,012,999 |
| Communication cost | 22,280 |
| Net result | -1,035,279 |
| Change from baseline | -0.687% |
| Campaigns including pilots | 8 |
| Pilots | 6 of 20 |
| Contacts | 5,570 |
| Unique customers | 5,111 |
| Risk score | 66.3% |
| Budget used | 22,280 of 100,000 |

Final campaigns selected by the starter agent:

| Campaign | Segment | Transition | Channel |
| --- | --- | --- | --- |
| `main_tariff_10_tariff_8` | HIGH | `tariff_10` -> `tariff_8` | sms |
| `main_tariff_8_tariff_9` | MID | `tariff_8` -> `tariff_9` | sms |

The first final campaign had gross lift `-1,007,245` across 2,950 contacts. The
second had gross lift `34,920` across 1,720 contacts. This explains most of the
negative seed-42 result and gives later work a concrete regression target.

## Ten-seed robustness run

Command: `python local_eval.py --runs 10`

| Seed | Net result |
| ---: | ---: |
| 0 | -1,019,431 |
| 1 | -576,204 |
| 2 | -338,971 |
| 3 | -140,281 |
| 4 | -1,019,237 |
| 5 | -366,964 |
| 6 | -320,312 |
| 7 | -348,932 |
| 8 | -76,493 |
| 9 | -601,538 |

- Median: `-357,948`
- Minimum: `-1,019,431`
- Maximum: `-76,493`
- Positive runs: `0/10`

The starter agent is consistently unprofitable on the mock seeds. Its sign does
not fluctuate, but that is failure stability rather than acceptable robustness.

## Submission reproducibility

Command: `python make_submission.py`, executed twice with the supplied seed 42.

- Rows: 2 campaigns
- File size: 226 bytes
- First SHA-256:
  `613d7c12899e1f42b70a7def168fdacb01a3bccf75c21539ce549be36d529420`
- Second SHA-256:
  `613d7c12899e1f42b70a7def168fdacb01a3bccf75c21539ce549be36d529420`
- Result: byte-identical

## Baseline conclusion

The starter meets the mechanical pilot and submission flow on the mock runner,
but fails the economic objective on every measured seed. Future milestones should
compare against these exact seeds and preserve a held-out set for final validation.
