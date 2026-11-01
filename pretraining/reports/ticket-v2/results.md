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
