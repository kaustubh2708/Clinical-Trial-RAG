# Evaluation results

## Retrieval (top_k=4; 20 paraphrased, 10 exact-term, 5 out-of-scope questions)

| Mode | Hit rate @k (all) | MRR (all) | MRR paraphrased | MRR exact-term | Contamination |
| --- | --- | --- | --- | --- | --- |
| dense | 0.97 | 0.94 | 0.97 | 0.90 | 0.00 |
| hybrid | 0.97 | 0.94 | 0.97 | 0.90 | 0.00 |

## Relevance gate sweep

Blocked = best dense relevance below the threshold (hybrid also lets through queries naming a code found verbatim in the protocol).

| MIN_RELEVANCE_SCORE | Mode | Abstention accuracy (out-of-scope blocked) | False abstention rate (answerable blocked) |
| --- | --- | --- | --- |
| 0.15 | dense | 0.60 | 0.00 |
| 0.15 | hybrid | 0.60 | 0.00 |
| 0.25 | dense | 0.80 | 0.00 |
| 0.25 | hybrid | 0.80 | 0.00 |
| 0.30 | dense | 1.00 | 0.07 |
| 0.30 | hybrid | 1.00 | 0.03 |
| 0.35 | dense | 1.00 | 0.07 |
| 0.35 | hybrid | 1.00 | 0.03 |
| 0.45 | dense | 1.00 | 0.17 |
| 0.45 | hybrid | 1.00 | 0.10 |

## Per-question retrieval

| id | kind | rank (dense) | rank (hybrid) | top dense relevance |
| --- | --- | --- | --- | --- |
| t2d-age | lookup | 1 | 3 | 0.518 |
| t2d-hba1c | lookup | 1 | 1 | 0.821 |
| t2d-egfr | lookup | 1 | 1 | 0.474 |
| t2d-kidney | lookup | 1 | 1 | 0.435 |
| t2d-pancreatitis | lookup | 1 | 1 | 0.770 |
| t2d-thyroid | lookup | 1 | 1 | 0.603 |
| t2d-glp1 | lookup | 1 | 1 | 0.614 |
| t2d-metformin | lookup | 1 | 1 | 0.827 |
| t2d-liver | lookup | 3 | 1 | 0.356 |
| t2d-pregnancy | lookup | 1 | 1 | 0.504 |
| t2d-bmi | lookup | 1 | 1 | 0.566 |
| t2d-heart | lookup | 1 | 1 | 0.538 |
| nsclc-ecog | lookup | 1 | 1 | 0.725 |
| nsclc-egfr-mut | lookup | 1 | 1 | 0.663 |
| nsclc-pd1 | lookup | 1 | 1 | 0.647 |
| nsclc-cns | lookup | 1 | 1 | 0.578 |
| nsclc-qtc | lookup | 1 | 1 | 0.789 |
| nsclc-organ | lookup | 1 | 1 | 0.609 |
| nsclc-chemo | lookup | 1 | 1 | 0.621 |
| nsclc-autoimmune | lookup | 1 | 1 | 0.781 |
| abstain-parking | abstain | - | - | 0.146 |
| abstain-payment | abstain | - | - | 0.187 |
| abstain-weather | abstain | - | - | 0.107 |
| abstain-recipe | abstain | - | - | 0.117 |
| abstain-tattoo | abstain | - | - | 0.295 |
| exact-cyp3a4 | exact | 1 | 1 | 0.574 |
| exact-men2 | exact | 1 | 1 | 0.291 |
| exact-ctcae | exact | 1 | 1 | 0.692 |
| exact-calcitonin | exact | 1 | 1 | 0.684 |
| exact-sglt2 | exact | 1 | 1 | 0.491 |
| exact-alk | exact | 1 | 1 | 0.430 |
| exact-anc | exact | - | - | 0.267 |
| exact-pembro | exact | 1 | 1 | 0.708 |
| exact-uln | exact | 1 | 1 | 0.537 |
| exact-recist | exact | 1 | 1 | 0.760 |

## End-to-end (LLM: ollama/qwen2.5:7b, mode=hybrid, MIN_RELEVANCE_SCORE=0.3)

| Metric | Value |
| --- | --- |
| Status accuracy | 0.72 (13/18) |
| Unsafe answers (ELIGIBLE when NOT_ELIGIBLE expected) | 0 |
| Citation verification rate | 1.00 (28/28) |
| Grounded rate | 0.80 |

| id | expected | predicted | grounded |
| --- | --- | --- | --- |
| t2d-age | INDETERMINATE | INDETERMINATE | True |
| t2d-hba1c | INDETERMINATE | INDETERMINATE | True |
| t2d-egfr | NOT_ELIGIBLE | NOT_ELIGIBLE | True |
| t2d-kidney | - | INDETERMINATE | True |
| t2d-pancreatitis | INDETERMINATE | INDETERMINATE | True |
| t2d-thyroid | NOT_ELIGIBLE | INDETERMINATE | False |
| t2d-glp1 | NOT_ELIGIBLE | INDETERMINATE | True |
| t2d-metformin | INDETERMINATE | INDETERMINATE | True |
| t2d-liver | - | INDETERMINATE | True |
| t2d-pregnancy | NOT_ELIGIBLE | NOT_ELIGIBLE | True |
| t2d-bmi | - | INDETERMINATE | True |
| t2d-heart | NOT_ELIGIBLE | INDETERMINATE | True |
| nsclc-ecog | - | INDETERMINATE | True |
| nsclc-egfr-mut | NOT_ELIGIBLE | INDETERMINATE | True |
| nsclc-pd1 | NOT_ELIGIBLE | NOT_ELIGIBLE | True |
| nsclc-cns | - | INDETERMINATE | True |
| nsclc-qtc | - | INDETERMINATE | True |
| nsclc-organ | - | INDETERMINATE | True |
| nsclc-chemo | NOT_ELIGIBLE | INDETERMINATE | True |
| nsclc-autoimmune | INDETERMINATE | INDETERMINATE | True |
| abstain-parking | CRITERIA_NOT_SPECIFIED | CRITERIA_NOT_SPECIFIED | False |
| abstain-payment | CRITERIA_NOT_SPECIFIED | CRITERIA_NOT_SPECIFIED | False |
| abstain-weather | CRITERIA_NOT_SPECIFIED | CRITERIA_NOT_SPECIFIED | False |
| abstain-recipe | CRITERIA_NOT_SPECIFIED | CRITERIA_NOT_SPECIFIED | False |
| abstain-tattoo | CRITERIA_NOT_SPECIFIED | CRITERIA_NOT_SPECIFIED | False |
| exact-cyp3a4 | - | INDETERMINATE | True |
| exact-men2 | - | INDETERMINATE | True |
| exact-ctcae | - | INDETERMINATE | True |
| exact-calcitonin | - | INDETERMINATE | True |
| exact-sglt2 | - | INDETERMINATE | True |
| exact-alk | - | INDETERMINATE | True |
| exact-anc | - | CRITERIA_NOT_SPECIFIED | False |
| exact-pembro | - | INDETERMINATE | True |
| exact-uln | - | INDETERMINATE | True |
| exact-recist | - | INDETERMINATE | True |
