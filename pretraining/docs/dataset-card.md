# Ticket-v2 dataset card

**Origin:** deterministic synthetic English helpdesk tickets, authored templates,
no scraped or customer data. Generated data and original code retain the imported
MIT license. Product names are fictional. Seed: 2026.

| Split | Examples | Base wording templates | Purpose |
| --- | ---: | ---: | --- |
| Training | 3,600 | 60 | Fit vocabulary, Transformer weights and classical baselines |
| Validation | 612 | 24 | Checkpoint, configuration and inference-mode selection |
| Test | 612 | 24 | Frozen final comparison |
| Challenge | 180 | 18 | Additional authored phrasing stress test |

Each split is balanced over six intents × three priorities. Every full dataset
text is unique. Wording templates are disjoint across all four splits in v2.
Each ticket combines a base wording template, product, context, explicit urgency
and a randomized ordering. The manifest records split hashes and generator hash.

## Labels

- `access_reset`: an individual's credentials, authentication, verification or
  account recovery prevents access.
- `billing_issue`: invoice amounts, charges, payments, tax discrepancies or refunds.
- `bug_report`: malfunction of a particular feature while the service remains usable.
- `feature_request`: an explicitly proposed new capability or enhancement.
- `service_outage`: widespread unavailability affecting the team/company/service.
- `account_cancellation`: a request to end a paid account, plan or renewal.
- `priority`: `low`, `normal`, `high`, supplied by the explicit urgency phrase.

The generator's labels describe authored single-intent examples. They are not
human-reviewed annotations of genuine customer messages. Ambiguous real messages
may not fit these six categories.

## Evaluation contract

The v2 templates and three-seed matrix were frozen before training. Validation
selects checkpoints by answer-token cross-entropy. Configuration/mode selection
uses mean validation joint accuracy, intent macro-F1, mean answer CE and a fixed
tie-break. Test and challenge predictions are generated after selection is saved.

Accuracies and macro-F1 are scored for both fields. Joint accuracy requires both.
Invalid/rejected predictions count as incorrect, and macro-F1 retains all classes.
The paired bootstrap resamples template groups, not individual paraphrase variants.

## Limits

All splits share products, context phrases, urgency vocabulary and generation
assumptions. Different template strings are not independent real-world domains;
many contain related vocabulary. Explicit urgency makes priority much easier than
intent. Report both fields and the number of template groups. This benchmark
establishes neither real-ticket accuracy nor broad language understanding.

The original **ticket-v1** dataset is preserved by a byte-compatible generator.
Its training and validation templates overlap; test/challenge were repeatedly
examined in historical tuning. V1 scores remain development evidence and are not
combined with v2 metrics or represented as an untouched benchmark.

Full generated CSVs are ignored. Regenerate them with `pretraining data prepare`.
Compact [v2 manifest](../reports/ticket-v2/data-manifest.json) and
[legacy manifest](../reports/legacy/data-manifest.json) identify the exact data.
