#!/usr/bin/env python3
"""Register classification from numeral features on ssj500k windows (output of extract_features.py).

Models, baselines and importance estimators are those of ../gigafida/analyze_gigafida.py.
Evaluation: stratified five-fold out-of-fold cross-validation grouped by document, so that no
window of a test document is seen in training and every window is predicted exactly once.
Registers with fewer than --min-class windows are dropped.

Output (in --outdir): summary.json, tables.md, register_profile_means.csv, confusion_rf.png/.pdf
"""
import argparse, json, os
import numpy as np, pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix, accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.dummy import DummyClassifier

FEATURES = ['num_per_1k', 'card_per_1k', 'ord_per_1k', 'card_ord_ratio', 'low_per_1k', 'high_per_1k',
            'prop_digit', 'prop_compound',
            'prop_case_nom', 'prop_case_gen', 'prop_case_dat', 'prop_case_acc', 'prop_case_loc', 'prop_case_ins',
            'prop_gender_agr', 'dual_per_1k',
            'prop_pos_initial', 'prop_pos_medial', 'prop_pos_final',
            'prop_ctx_temporal', 'prop_ctx_measure', 'prop_ctx_monetary', 'prop_ctx_enum', 'prop_ctx_other']
ORDER = ['newspaper', 'fiction', 'wikipedia', 'magazine', 'professional', 'internet', 'other']


def models(seed):
    return {
        'Naive Bayes': GaussianNB(),
        'Linear SVM': make_pipeline(StandardScaler(), LinearSVC(C=1.0, max_iter=20000, random_state=seed)),
        'Random forest': RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
    }


def oof(model, X, y, splits):
    pred = np.empty(len(y), dtype=object)
    for tr, te in splits:
        model.fit(X[tr], y[tr]); pred[te] = model.predict(X[te])
    return pred


def scores(y, pred, labels):
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    return {'macro_f1': float(f1_score(y, pred, labels=labels, average='macro', zero_division=0)),
            'weighted_f1': float(f1_score(y, pred, labels=labels, average='weighted', zero_division=0)),
            'accuracy': float(accuracy_score(y, pred)),
            'per_class': {l: {'precision': float(a), 'recall': float(b), 'f1': float(c), 'support': int(d)}
                          for l, a, b, c, d in zip(labels, p, r, f, s)},
            'confusion': confusion_matrix(y, pred, labels=labels).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='features.csv')
    ap.add_argument('--outdir', default='results')
    ap.add_argument('--min-class', type=int, default=20)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    data = pd.read_csv(a.features, dtype={'document_id': str})
    n_all = len(data)
    counts = data['register'].value_counts()
    keep = counts[counts >= a.min_class].index
    data = data[data['register'].isin(keep)].reset_index(drop=True)
    labels = [l for l in ORDER if l in set(data['register'])] + sorted(set(data['register']) - set(ORDER))
    X = data[FEATURES].values.astype(float)
    y = data['register'].values.astype(object)
    groups = data['document_id'].values

    splits = list(StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=a.seed).split(X, y, groups))
    res = {name: scores(y, oof(m, X, y, splits), labels) for name, m in models(a.seed).items()}
    res['Majority-class baseline'] = scores(y, oof(DummyClassifier(strategy='most_frequent'), X, y, splits), labels)
    y_sh = y.copy(); np.random.RandomState(a.seed).shuffle(y_sh)       # random-feature control
    pred = np.empty(len(y), dtype=object); m = models(a.seed)['Linear SVM']
    for tr, te in splits:
        m.fit(X[tr], y_sh[tr]); pred[te] = m.predict(X[te])
    res['Random-feature control'] = scores(y, pred, labels)

    # Importance: RF impurity, |SVM coef|, permutation (RF on held-out folds, macro-F1 drop)
    rf = RandomForestClassifier(n_estimators=200, random_state=a.seed, n_jobs=-1).fit(X, y)
    svm = make_pipeline(StandardScaler(), LinearSVC(C=1.0, max_iter=20000, random_state=a.seed)).fit(X, y)
    coef = np.abs(svm[-1].coef_).mean(axis=0)
    perm = np.zeros(len(FEATURES))
    for tr, te in splits:
        mm = RandomForestClassifier(n_estimators=100, random_state=a.seed, n_jobs=-1).fit(X[tr], y[tr])
        perm += permutation_importance(mm, X[te], y[te], scoring='f1_macro', n_repeats=3,
                                       random_state=a.seed, n_jobs=-1).importances_mean
    imp = {'rf_impurity': dict(zip(FEATURES, rf.feature_importances_.round(5).tolist())),
           'svm_abs_coef': dict(zip(FEATURES, (coef / coef.sum()).round(5).tolist())),
           'permutation_grouped': dict(zip(FEATURES, (perm / len(splits)).round(5).tolist()))}

    prof = data.groupby('register')[FEATURES].mean().loc[labels]
    prof.to_csv(os.path.join(a.outdir, 'register_profile_means.csv'))
    summary = {'n_windows_input': int(n_all), 'n_windows': int(len(data)),
               'n_documents': int(data['document_id'].nunique()),
               'windows_per_register': data['register'].value_counts().to_dict(),
               'documents_per_register': data.groupby('register')['document_id'].nunique().to_dict(),
               'dropped_registers': {k: int(v) for k, v in counts.items() if v < a.min_class},
               'folds': a.folds, 'seed': a.seed, 'results': res, 'importance': imp,
               'profile_means': prof.round(4).to_dict(orient='index')}
    with open(os.path.join(a.outdir, 'summary.json'), 'w', encoding='utf8') as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    cm = np.array(res['Random forest']['confusion'], dtype=float); cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    im = ax.imshow(cmn, cmap='Greys', vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha='right'); ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted register'); ax.set_ylabel('True register')
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f'{cmn[i, j]:.2f}', ha='center', va='center', fontsize=8.5, color='white' if cmn[i, j] > 0.55 else 'black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(a.outdir, f'confusion_rf.{ext}'), dpi=300)
    plt.close(fig)

    with open(os.path.join(a.outdir, 'tables.md'), 'w', encoding='utf8') as fh:
        fh.write('## Document-grouped stratified 5-fold out-of-fold CV\n\n| Model | Macro-F1 | Weighted-F1 | Accuracy | '
                 + ' | '.join(labels) + ' |\n|---|---|---|---|' + '---|' * len(labels) + '\n')
        for name, r in res.items():
            fh.write(f"| {name} | {r['macro_f1']:.2f} | {r['weighted_f1']:.2f} | {r['accuracy']:.2f} | "
                     + ' | '.join(f"{r['per_class'][l]['f1']:.2f}" for l in labels) + ' |\n')
        fh.write('\n## Feature importance\n\n| Feature | RF impurity | SVM |coef| | Permutation (macro-F1 drop) |\n|---|---|---|---|\n')
        for f in sorted(FEATURES, key=lambda k: -imp['rf_impurity'][k]):
            fh.write(f"| {f} | {imp['rf_impurity'][f]:.3f} | {imp['svm_abs_coef'][f]:.3f} | {imp['permutation_grouped'][f]:.3f} |\n")
        fh.write('\n## Register profile (means over windows)\n\n' + prof.round(3).to_markdown() + '\n')
    print(json.dumps({n: [round(r['macro_f1'], 2), round(r['weighted_f1'], 2)] for n, r in res.items()}))
    for n in ['Naive Bayes', 'Linear SVM', 'Random forest']:
        print(n, {l: round(res[n]['per_class'][l]['f1'], 2) for l in labels})


if __name__ == '__main__':
    main()
