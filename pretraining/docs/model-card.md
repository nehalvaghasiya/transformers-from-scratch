---
license: mit
language: en
tags: [from-scratch, synthetic-data, text-classification]
---
# Ticket Transformer trained from scratch

Run `ticket-v2-base-42`; seed 42; 655,872 Transformer
parameters; 655,872 total trainable parameters.
Word-level tokenizer fitted on training text. Answer-token causal cross-entropy,
with optional auxiliary pooled classifiers. Label tokens are constraint-decoded
and mapped to JSON; this is a task-specific model, not free-form JSON generation.

## Benchmark

| System | Test joint accuracy | Challenge joint accuracy | Test intent macro-F1 |
| --- | ---: | ---: | ---: |
| base-generative | 92.37% ± 1.43 | 60.37% ± 4.17 | 0.924 |
| char_tfidf | 99.18% | 93.89% | 0.992 |
| cosine-generative | 87.85% ± 9.05 | 57.04% ± 9.88 | 0.874 |
| majority | 5.56% | 5.56% | 0.048 |
| pool-dual | 96.51% ± 2.98 | 73.15% ± 6.14 | 0.966 |
| pool-generative | 94.34% ± 4.19 | 71.11% ± 5.85 | 0.943 |
| random-init | 5.23% ± 0.28 | 5.19% ± 1.28 | 0.076 |
| word_tfidf | 99.51% | 78.89% | 0.995 |

± denotes sample standard deviation across three fixed seeds. Classical baselines are single fitted systems.


These aggregate results describe a three-seed experiment, not just this artifact.
See the accompanying benchmark report for individual seeds, selection, uncertainty
and failures. All data is synthetic; no real-ticket accuracy is established.

## Loading

Install this repository's locked CPU environment and use
`pretraining predict --model ARTIFACT --text 'My invoice is wrong. No rush.'`.
The `run_info.json` records source, data and artifact hashes. Architecture and
vocabulary are stored with weights. Weights were trained from random initialization.

## Limitations

Six fixed intents, three urgency classes, English synthetic templates, short inputs.
Urgency phrases are shared across splits. Unknown words become UNK. Multiple intents,
implicit urgency, typos, long tickets and real data can cause failures. Confidence
is not calibrated. Training and inference CUDA memory has not been validated here.
