# Evaluation results

## Retrieval (top_k=4, 20 lookup + 5 out-of-scope questions)

| Metric | Value |
| --- | --- |
| Hit rate @4 | 1.00 |
| MRR | 0.97 |
| Contamination rate | 0.00 |

## Relevance gate sweep

| MIN_RELEVANCE_SCORE | Abstention accuracy (out-of-scope blocked) | False abstention rate (answerable blocked) |
| --- | --- | --- |
| 0.15 | 0.60 | 0.00 |
| 0.25 | 0.80 | 0.00 |
| 0.30 | 1.00 | 0.00 |
| 0.35 | 1.00 | 0.00 |
| 0.45 | 1.00 | 0.10 |

## Per-question retrieval

| id | kind | rank of first correct chunk | top relevance |
| --- | --- | --- | --- |
| t2d-age | lookup | 1 | 0.518 |
| t2d-hba1c | lookup | 1 | 0.821 |
| t2d-egfr | lookup | 1 | 0.474 |
| t2d-kidney | lookup | 1 | 0.435 |
| t2d-pancreatitis | lookup | 1 | 0.770 |
| t2d-thyroid | lookup | 1 | 0.603 |
| t2d-glp1 | lookup | 1 | 0.614 |
| t2d-metformin | lookup | 1 | 0.827 |
| t2d-liver | lookup | 3 | 0.356 |
| t2d-pregnancy | lookup | 1 | 0.504 |
| t2d-bmi | lookup | 1 | 0.566 |
| t2d-heart | lookup | 1 | 0.538 |
| nsclc-ecog | lookup | 1 | 0.725 |
| nsclc-egfr-mut | lookup | 1 | 0.663 |
| nsclc-pd1 | lookup | 1 | 0.647 |
| nsclc-cns | lookup | 1 | 0.578 |
| nsclc-qtc | lookup | 1 | 0.789 |
| nsclc-organ | lookup | 1 | 0.609 |
| nsclc-chemo | lookup | 1 | 0.621 |
| nsclc-autoimmune | lookup | 1 | 0.781 |
| abstain-parking | abstain | - | 0.146 |
| abstain-payment | abstain | - | 0.187 |
| abstain-weather | abstain | - | 0.107 |
| abstain-recipe | abstain | - | 0.117 |
| abstain-tattoo | abstain | - | 0.295 |
