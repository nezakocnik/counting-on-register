## Protocol A: document-level stratified 5-fold out-of-fold CV

| Model | Macro-F1 | Weighted-F1 | Accuracy |
|---|---|---|---|
| Naive Bayes | 0.35 | 0.43 | 0.44 |
| Linear SVM | 0.30 | 0.50 | 0.61 |
| Random forest | 0.74 | 0.84 | 0.85 |
| Majority-class baseline | 0.12 | 0.45 | 0.60 |
| Random-feature control | 0.12 | 0.45 | 0.60 |
| Linear SVM (class-weighted) | 0.46 | 0.58 | 0.59 |
| Random forest (class-weighted) | 0.74 | 0.83 | 0.84 |

| Register | Naive Bayes | Linear SVM | Random forest | Linear SVM (class-weighted) | Random forest (class-weighted) |
|---|---|---|---|---|---|
| fiction | 0.36 | 0.01 | 0.83 | 0.42 | 0.83 |
| non-fiction | 0.14 | 0.05 | 0.44 | 0.29 | 0.46 |
| legislative | 0.38 | 0.72 | 0.76 | 0.58 | 0.76 |
| magazine | 0.26 | 0.22 | 0.65 | 0.37 | 0.63 |
| newspaper | 0.49 | 0.75 | 0.90 | 0.72 | 0.89 |
| internet | 0.46 | 0.03 | 0.86 | 0.38 | 0.86 |

## Protocol B: title-grouped stratified 5-fold out-of-fold CV

| Model | Macro-F1 | Weighted-F1 | Accuracy |
|---|---|---|---|
| Naive Bayes | 0.30 | 0.33 | 0.35 |
| Linear SVM | 0.28 | 0.48 | 0.58 |
| Random forest | 0.66 | 0.73 | 0.73 |
| Majority-class baseline | 0.12 | 0.45 | 0.60 |
| Random-feature control | 0.12 | 0.45 | 0.60 |
| Linear SVM (class-weighted) | 0.43 | 0.53 | 0.52 |
| Random forest (class-weighted) | 0.66 | 0.74 | 0.74 |

| Register | Naive Bayes | Linear SVM | Random forest | Linear SVM (class-weighted) | Random forest (class-weighted) |
|---|---|---|---|---|---|
| fiction | 0.36 | 0.01 | 0.79 | 0.39 | 0.79 |
| non-fiction | 0.13 | 0.04 | 0.42 | 0.28 | 0.43 |
| legislative | 0.35 | 0.70 | 0.73 | 0.58 | 0.71 |
| magazine | 0.18 | 0.19 | 0.40 | 0.30 | 0.39 |
| newspaper | 0.35 | 0.73 | 0.81 | 0.65 | 0.81 |
| internet | 0.44 | 0.03 | 0.80 | 0.37 | 0.81 |

## Protocol C: balanced subsample, title-grouped CV (mean over reps)

| Model | Macro-F1 (sd) | Weighted-F1 | Accuracy |
|---|---|---|---|
| Naive Bayes | 0.48 (0.01) | 0.48 | 0.52 |
| Linear SVM | 0.57 (0.01) | 0.57 | 0.60 |
| Random forest | 0.67 (0.01) | 0.67 | 0.68 |
| Linear SVM (class-weighted) | 0.58 (0.01) | 0.58 | 0.61 |
| Random forest (class-weighted) | 0.67 (0.01) | 0.67 | 0.68 |
| Majority-class baseline | 0.06 (0.00) | 0.06 | 0.12 |
| Random-feature control | 0.19 (0.05) | 0.19 | 0.19 |

## Feature importance

| Feature | RF impurity | SVM |coef| | Permutation (macro-F1 drop) |
|---|---|---|---|
| prop_compound | 0.066 | 0.011 | 0.024 |
| prop_ctx_enum | 0.056 | 0.046 | 0.064 |
| prop_pos_final | 0.055 | 0.005 | 0.041 |
| ord_per_1k | 0.053 | 0.049 | 0.029 |
| prop_ctx_other | 0.053 | 0.095 | 0.017 |
| prop_digit | 0.052 | 0.044 | 0.016 |
| prop_ctx_temporal | 0.050 | 0.065 | 0.026 |
| prop_case_dat | 0.049 | 0.003 | 0.034 |
| prop_ctx_monetary | 0.044 | 0.067 | 0.024 |
| prop_pos_medial | 0.043 | 0.027 | 0.022 |
| prop_gender_agr | 0.043 | 0.026 | 0.009 |
| num_per_1k | 0.042 | 0.207 | 0.010 |
| high_per_1k | 0.041 | 0.086 | 0.004 |
| prop_pos_initial | 0.040 | 0.015 | 0.015 |
| card_ord_ratio | 0.039 | 0.035 | 0.008 |
| card_per_1k | 0.037 | 0.076 | 0.003 |
| low_per_1k | 0.035 | 0.040 | 0.004 |
| prop_case_acc | 0.034 | 0.016 | 0.010 |
| prop_ctx_measure | 0.033 | 0.039 | 0.006 |
| prop_case_gen | 0.032 | 0.011 | 0.024 |
| prop_case_loc | 0.029 | 0.008 | 0.003 |
| prop_case_ins | 0.026 | 0.007 | 0.005 |
| dual_per_1k | 0.024 | 0.007 | 0.003 |
| prop_case_nom | 0.024 | 0.014 | 0.002 |

## Register profile (means)

| register    |   num_per_1k |   card_per_1k |   ord_per_1k |   card_ord_ratio |   low_per_1k |   high_per_1k |   prop_digit |   prop_compound |   prop_case_nom |   prop_case_gen |   prop_case_dat |   prop_case_acc |   prop_case_loc |   prop_case_ins |   prop_gender_agr |   dual_per_1k |   prop_pos_initial |   prop_pos_medial |   prop_pos_final |   prop_ctx_temporal |   prop_ctx_measure |   prop_ctx_monetary |   prop_ctx_enum |   prop_ctx_other |
|:------------|-------------:|--------------:|-------------:|-----------------:|-------------:|--------------:|-------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|------------------:|--------------:|-------------------:|------------------:|-----------------:|--------------------:|-------------------:|--------------------:|----------------:|-----------------:|
| fiction     |        9.416 |         4.227 |        1.264 |            3.439 |        3.484 |         2.007 |        0.086 |           0.068 |           0.278 |           0.134 |           0.024 |           0.378 |           0.12  |           0.064 |             0.75  |         0.878 |              0.061 |             0.936 |            0.003 |               0.188 |              0.027 |               0.019 |           0.085 |            0.681 |
| non-fiction |       27.966 |        19.343 |        4.006 |            9.274 |       10.155 |        13.195 |        0.527 |           0.027 |           0.269 |           0.168 |           0.022 |           0.3   |           0.166 |           0.072 |             0.419 |         1.106 |              0.077 |             0.897 |            0.028 |               0.186 |              0.065 |               0.017 |           0.103 |            0.629 |
| legislative |       51.363 |        22.963 |       20.132 |            1.58  |       22.309 |        20.786 |        0.636 |           0.01  |           0.174 |           0.387 |           0.023 |           0.182 |           0.172 |           0.053 |             0.329 |         0.714 |              0.051 |             0.923 |            0.027 |               0.152 |              0.04  |               0.054 |           0.385 |            0.374 |
| magazine    |       24.187 |        16.196 |        3.753 |            5.626 |        8.767 |        11.182 |        0.507 |           0.032 |           0.264 |           0.142 |           0.017 |           0.338 |           0.154 |           0.079 |             0.409 |         1.241 |              0.057 |             0.929 |            0.015 |               0.225 |              0.103 |               0.039 |           0.082 |            0.551 |
| newspaper   |       29.899 |        19.732 |        6.155 |            3.793 |       10.828 |        15.059 |        0.535 |           0.035 |           0.245 |           0.143 |           0.016 |           0.339 |           0.169 |           0.082 |             0.372 |         1.31  |              0.034 |             0.963 |            0.003 |               0.249 |              0.074 |               0.075 |           0.087 |            0.514 |
| internet    |       34     |        22.954 |        6.937 |            3.66  |       12.498 |        17.392 |        0.621 |           0.016 |           0.242 |           0.138 |           0.012 |           0.328 |           0.187 |           0.09  |             0.313 |         1.333 |              0.029 |             0.968 |            0.002 |               0.275 |              0.083 |               0.069 |           0.116 |            0.458 |
