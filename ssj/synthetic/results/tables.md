## Document-grouped stratified 5-fold out-of-fold CV

| Model | Macro-F1 | Weighted-F1 | Accuracy | commercial | narrative | report | statute |
|---|---|---|---|---|---|---|---|
| Naive Bayes | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Linear SVM | 0.99 | 0.99 | 0.99 | 1.00 | 0.99 | 0.99 | 1.00 |
| Random forest | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Majority-class baseline | 0.19 | 0.19 | 0.24 | 0.34 | 0.21 | 0.21 | 0.00 |
| Random-feature control | 0.27 | 0.27 | 0.27 | 0.37 | 0.33 | 0.11 | 0.28 |

## Feature importance

| Feature | RF impurity | SVM |coef| | Permutation (macro-F1 drop) |
|---|---|---|---|
| prop_ctx_monetary | 0.121 | 0.143 | 0.010 |
| ord_per_1k | 0.114 | 0.066 | 0.000 |
| high_per_1k | 0.109 | 0.058 | 0.000 |
| prop_ctx_temporal | 0.104 | 0.074 | 0.002 |
| num_per_1k | 0.101 | 0.026 | 0.000 |
| prop_digit | 0.099 | 0.139 | 0.000 |
| card_ord_ratio | 0.085 | 0.031 | 0.000 |
| card_per_1k | 0.085 | 0.087 | 0.000 |
| prop_ctx_enum | 0.051 | 0.054 | 0.000 |
| prop_case_gen | 0.039 | 0.027 | 0.000 |
| prop_gender_agr | 0.026 | 0.016 | 0.000 |
| prop_case_dat | 0.014 | 0.031 | 0.000 |
| low_per_1k | 0.009 | 0.038 | 0.000 |
| prop_ctx_other | 0.009 | 0.045 | 0.000 |
| prop_case_acc | 0.008 | 0.018 | 0.000 |
| prop_case_loc | 0.005 | 0.028 | 0.000 |
| prop_case_nom | 0.005 | 0.024 | 0.000 |
| prop_pos_medial | 0.004 | 0.009 | 0.000 |
| prop_case_ins | 0.003 | 0.031 | 0.000 |
| prop_pos_initial | 0.003 | 0.012 | 0.000 |
| prop_pos_final | 0.002 | 0.011 | 0.000 |
| prop_compound | 0.002 | 0.023 | 0.000 |
| dual_per_1k | 0.001 | 0.009 | 0.000 |
| prop_ctx_measure | 0.000 | 0.000 | 0.000 |

## Register profile (means over windows)

| register   |   num_per_1k |   card_per_1k |   ord_per_1k |   card_ord_ratio |   low_per_1k |   high_per_1k |   prop_digit |   prop_compound |   prop_case_nom |   prop_case_gen |   prop_case_dat |   prop_case_acc |   prop_case_loc |   prop_case_ins |   prop_gender_agr |   dual_per_1k |   prop_pos_initial |   prop_pos_medial |   prop_pos_final |   prop_ctx_temporal |   prop_ctx_measure |   prop_ctx_monetary |   prop_ctx_enum |   prop_ctx_other |
|:-----------|-------------:|--------------:|-------------:|-----------------:|-------------:|--------------:|-------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|------------------:|--------------:|-------------------:|------------------:|-----------------:|--------------------:|-------------------:|--------------------:|----------------:|-----------------:|
| commercial |       57.619 |        55     |        2.619 |           23.103 |        8.81  |        46.032 |        0.805 |           0.162 |           0.236 |           0.266 |           0.122 |           0.199 |           0.085 |           0.093 |             0.059 |         2.976 |              0.08  |             0.834 |            0.086 |               0     |                  0 |               0.569 |           0.482 |            0.03  |
| narrative  |       13.056 |        12.54  |        0.516 |            7.524 |        9.127 |         0.437 |        0.03  |           0.106 |           0.355 |           0.063 |           0     |           0.384 |           0.114 |           0.084 |             0.396 |         4.563 |              0.103 |             0.799 |            0.098 |               0.499 |                  0 |               0     |           0.443 |            0.128 |
| report     |       42.222 |        36.389 |        5.833 |            7.972 |       13.968 |        21.667 |        0.513 |           0.177 |           0.237 |           0.198 |           0     |           0.285 |           0.207 |           0.074 |             0.16  |         4.048 |              0.064 |             0.859 |            0.077 |               0.55  |                  0 |               0     |           0.475 |            0.036 |
| statute    |       78.492 |        33.214 |       45.278 |            0.773 |       21.468 |        48.175 |        0.613 |           0.187 |           0.098 |           0.561 |           0     |           0.112 |           0.23  |           0     |             0.243 |         1.865 |              0.094 |             0.809 |            0.098 |               0     |                  0 |               0     |           0.818 |            0.182 |
