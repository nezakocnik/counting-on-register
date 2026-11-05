#!/usr/bin/env python3
"""Register classification from numeral features, Gigafida 2.1 document level.

Input : gigafida_numeral_counts.tsv  (one row per Gigafida document; raw counts
        obtained from CLARIN.SI KonText frequency distributions over text.id)
Output: results_gigafida/  (feature matrix, summary.json, tables, figures)
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix, accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.dummy import DummyClassifier

CLASS_MAP = {
    'tisk/periodično/časopis': 'newspaper',
    'tisk/periodično/revija': 'magazine',
    'internet': 'internet',
    'tisk/knjižno/leposlovno': 'fiction',
    'tisk/knjižno/strokovno': 'non-fiction',
}
LEGAL_SOURCE = 'Državni zbor Republike Slovenije'   # tisk/drugo documents from the National Assembly = legislative register
ORDER = ['fiction', 'non-fiction', 'legislative', 'magazine', 'newspaper', 'internet']

FEATURES = ['num_per_1k', 'card_per_1k', 'ord_per_1k', 'card_ord_ratio', 'low_per_1k', 'high_per_1k',
            'prop_digit', 'prop_compound',
            'prop_case_nom', 'prop_case_gen', 'prop_case_dat', 'prop_case_acc', 'prop_case_loc', 'prop_case_ins',
            'prop_gender_agr', 'dual_per_1k',
            'prop_pos_initial', 'prop_pos_medial', 'prop_pos_final',
            'prop_ctx_temporal', 'prop_ctx_measure', 'prop_ctx_monetary', 'prop_ctx_enum', 'prop_ctx_other']


def build_features(df):
    f = pd.DataFrame(index=df.index)
    ntok = df['ntok'].astype(float)
    num = df[['Kag', 'Kav', 'Kbg', 'Kbv', 'Kbz', 'Kbd', 'Krg', 'Krv']].sum(axis=1).astype(float)
    card = df[['Kag', 'Kbg', 'Krg']].sum(axis=1).astype(float)
    ordn = df[['Kav', 'Kbv', 'Krv']].sum(axis=1).astype(float)
    digit = df[['Kag', 'Kav']].sum(axis=1).astype(float)
    word = df[['Kbg', 'Kbv', 'Kbz', 'Kbd']].sum(axis=1).astype(float)
    low = df['low'].astype(float)
    high = (card + ordn - low).clip(lower=0)
    cases = df[['case_i', 'case_r', 'case_d', 'case_t', 'case_m', 'case_o']].astype(float)
    ncase = cases.sum(axis=1)
    safe_num = num.replace(0, np.nan)
    f['num_per_1k'] = 1000 * num / ntok
    f['card_per_1k'] = 1000 * card / ntok
    f['ord_per_1k'] = 1000 * ordn / ntok
    f['card_ord_ratio'] = (card / (ordn + 1.0))
    f['low_per_1k'] = 1000 * low / ntok
    f['high_per_1k'] = 1000 * high / ntok
    f['prop_digit'] = (digit / (digit + word).replace(0, np.nan)).fillna(0)
    f['prop_compound'] = (df['compound'] / df['Kbg'].replace(0, np.nan)).fillna(0)
    for col, lab in zip(['case_i', 'case_r', 'case_d', 'case_t', 'case_m', 'case_o'],
                        ['nom', 'gen', 'dat', 'acc', 'loc', 'ins']):
        f['prop_case_' + lab] = (cases[col] / ncase.replace(0, np.nan)).fillna(0)
    f['prop_gender_agr'] = (df['gender'] / safe_num).fillna(0)
    f['dual_per_1k'] = 1000 * df['dual'] / ntok
    f['prop_pos_initial'] = (df['sinit'] / safe_num).fillna(0)
    f['prop_pos_final'] = (df['sfinal'] / safe_num).fillna(0)
    f['prop_pos_medial'] = (1 - f['prop_pos_initial'] - f['prop_pos_final']).clip(lower=0)
    for k in ['temporal', 'measure', 'monetary', 'enum']:
        f['prop_ctx_' + k] = (df['ctx_' + k] / safe_num).fillna(0).clip(upper=1)
    f['prop_ctx_other'] = (1 - f[['prop_ctx_temporal', 'prop_ctx_measure', 'prop_ctx_monetary', 'prop_ctx_enum']].sum(axis=1)).clip(lower=0)
    return f[FEATURES]


def models(seed, subset=None):
    m = {
        'Naive Bayes': GaussianNB(),
        'Linear SVM': make_pipeline(StandardScaler(), LinearSVC(C=1.0, max_iter=20000, random_state=seed)),
        'Random forest': RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
        'Linear SVM (class-weighted)': make_pipeline(StandardScaler(), LinearSVC(C=1.0, max_iter=20000, random_state=seed, class_weight='balanced')),
        'Random forest (class-weighted)': RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1, class_weight='balanced'),
    }
    if subset: m = {k: v for k, v in m.items() if k in subset}
    return m


def oof_predict(model, X, y, splits):
    pred = np.empty(len(y), dtype=object)
    for tr, te in splits:
        m = model
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def scores(y, pred, labels):
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    return {
        'macro_f1': float(f1_score(y, pred, labels=labels, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y, pred, labels=labels, average='weighted', zero_division=0)),
        'accuracy': float(accuracy_score(y, pred)),
        'per_class': {lab: {'precision': float(pp), 'recall': float(rr), 'f1': float(ff), 'support': int(ss)}
                      for lab, pp, rr, ff, ss in zip(labels, p, r, f, s)},
        'confusion': confusion_matrix(y, pred, labels=labels).tolist(),
    }


def run_protocol(X, y, groups, labels, seed, folds, grouped, subset=None, baselines=True):
    rng = np.random.RandomState(seed)
    if grouped:
        cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = list(cv.split(X, y, groups))
    else:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = list(cv.split(X, y))
    out = {}
    for name, m in models(seed, subset).items():
        pred = oof_predict(m, X, y, splits)
        out[name] = scores(y, pred, labels)
    if not baselines:
        return out, splits
    # majority baseline
    pred = oof_predict(DummyClassifier(strategy='most_frequent'), X, y, splits)
    out['Majority-class baseline'] = scores(y, pred, labels)
    # random-feature control: linear SVM trained on shuffled labels (label-feature link destroyed)
    y_sh = y.copy()
    rng.shuffle(y_sh)
    pred = np.empty(len(y), dtype=object)
    m = models(seed)['Linear SVM']
    for tr, te in splits:
        m.fit(X[tr], y_sh[tr]); pred[te] = m.predict(X[te])
    out['Random-feature control'] = scores(y, pred, labels)
    return out, splits


def importances(X, y, feats, splits, seed, labels):
    rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1).fit(X, y)
    rf_imp = dict(zip(feats, rf.feature_importances_.round(5).tolist()))
    svm = make_pipeline(StandardScaler(), LinearSVC(C=1.0, max_iter=20000, random_state=seed)).fit(X, y)
    coef = np.abs(svm[-1].coef_).mean(axis=0)
    svm_imp = dict(zip(feats, (coef / coef.sum()).round(5).tolist()))
    # grouped permutation importance on held-out folds (RF), averaged over folds
    perm = np.zeros(len(feats))
    for tr, te in splits:
        m = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1).fit(X[tr], y[tr])
        pi = permutation_importance(m, X[te], y[te], scoring='f1_macro', n_repeats=3, random_state=seed, n_jobs=-1)
        perm += pi.importances_mean
    perm /= len(splits)
    perm_imp = dict(zip(feats, perm.round(5).tolist()))
    return rf_imp, svm_imp, perm_imp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--counts', default='gigafida_numeral_counts.tsv')
    ap.add_argument('--outdir', default='results_gigafida')
    ap.add_argument('--min-tokens', type=int, default=300)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--balanced-reps', type=int, default=3)
    ap.add_argument('--stage', default='all', choices=['A', 'B', 'C', 'imp', 'report', 'all'])
    ap.add_argument('--models', default='')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    df = pd.read_csv(a.counts, sep='\t', dtype={'id': str, 'title': str, 'source': str, 'year': str, 'publisher': str})
    df['register'] = df['cls'].map(CLASS_MAP)
    df.loc[(df['cls'] == 'tisk/drugo') & (df['source'] == LEGAL_SOURCE), 'register'] = 'legislative'
    n_all = len(df)
    df = df[df['register'].notna()].copy()
    n_cls = len(df)
    df = df[df['ntok'] >= a.min_tokens].copy()
    n_len = len(df)
    feats = build_features(df)
    data = pd.concat([df[['id', 'register', 'cls', 'title', 'source', 'year', 'ntok']].reset_index(drop=True),
                      feats.reset_index(drop=True)], axis=1)
    data.to_csv(os.path.join(a.outdir, 'gigafida_features.csv'), index=False)

    labels = [l for l in ORDER if l in set(data['register'])]
    X = data[FEATURES].values.astype(float)
    y = data['register'].values.astype(object)
    groups = data['title'].fillna('').values

    summary = {'n_documents_corpus': int(n_all), 'n_documents_with_register': int(n_cls),
               'n_documents_after_min_tokens': int(n_len), 'min_tokens': a.min_tokens,
               'tokens_analysed': int(data['ntok'].sum()),
               'class_counts': data['register'].value_counts().to_dict(),
               'class_tokens': data.groupby('register')['ntok'].sum().to_dict(),
               'class_titles': data.groupby('register')['title'].nunique().to_dict(),
               'features': FEATURES, 'folds': a.folds, 'seed': a.seed}

    stage = a.stage
    def save(name, obj):
        with open(os.path.join(a.outdir, name), 'w', encoding='utf8') as fh: json.dump(obj, fh, ensure_ascii=False)
    def load(name):
        with open(os.path.join(a.outdir, name), encoding='utf8') as fh: return json.load(fh)

    subset = a.models.split(',') if a.models else None
    for st in [s for s in ('A', 'B') if stage in (s, 'all')]:
        res, _ = run_protocol(X, y, groups, labels, a.seed, a.folds, grouped=(st == 'B'), subset=subset, baselines=not a.models)
        fn = 'stage_%s.json' % st
        prev = load(fn) if os.path.exists(os.path.join(a.outdir, fn)) else {}
        prev.update(res); save(fn, prev)
        print('protocol', st, 'done', flush=True)
    if stage in ('C', 'all'):
        nmin = int(data['register'].value_counts().min())
        reps = []
        rng = np.random.RandomState(a.seed)
        for r in range(a.balanced_reps):
            idx = np.concatenate([rng.choice(np.where(y == lab)[0], nmin, replace=False) for lab in labels])
            res, _ = run_protocol(X[idx], y[idx], groups[idx], labels, a.seed + r, a.folds, grouped=True)
            reps.append(res)
        agg = {}
        for name in reps[0]:
            agg[name] = {'macro_f1_mean': float(np.mean([r[name]['macro_f1'] for r in reps])),
                         'macro_f1_sd': float(np.std([r[name]['macro_f1'] for r in reps])),
                         'weighted_f1_mean': float(np.mean([r[name]['weighted_f1'] for r in reps])),
                         'accuracy_mean': float(np.mean([r[name]['accuracy'] for r in reps])),
                         'per_class_f1_mean': {lab: float(np.mean([r[name]['per_class'][lab]['f1'] for r in reps])) for lab in labels},
                         'confusion_sum': np.sum([np.array(r[name]['confusion']) for r in reps], axis=0).tolist()}
        save('stage_C.json', {'n_per_class': nmin, 'reps': a.balanced_reps, 'results': agg})
    if stage in ('imp', 'all'):
        cv = StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=a.seed)
        splitsB = list(cv.split(X, y, groups))
        rf_imp, svm_imp, perm_imp = importances(X, y, FEATURES, splitsB, a.seed, labels)
        save('stage_imp.json', {'rf_impurity': rf_imp, 'svm_abs_coef': svm_imp, 'permutation_grouped': perm_imp})
    if stage != 'report' and stage != 'all':
        print('stage', stage, 'done'); return

    resA = load('stage_A.json'); resB = load('stage_B.json'); C = load('stage_C.json'); agg = C['results']
    imp = load('stage_imp.json'); rf_imp, svm_imp, perm_imp = imp['rf_impurity'], imp['svm_abs_coef'], imp['permutation_grouped']
    summary['protocol_A_document_cv'] = resA
    summary['protocol_B_title_grouped_cv'] = resB
    summary['protocol_C_balanced_grouped_cv'] = C
    summary['importance'] = imp

    # Descriptive profile
    prof = data.groupby('register')[FEATURES].mean().loc[labels]
    prof.to_csv(os.path.join(a.outdir, 'register_profile_means.csv'))
    data.groupby('register')[FEATURES].median().loc[labels].to_csv(os.path.join(a.outdir, 'register_profile_medians.csv'))
    summary['profile_means'] = prof.round(4).to_dict(orient='index')

    with open(os.path.join(a.outdir, 'summary.json'), 'w', encoding='utf8') as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)

    # Figures: confusion matrices (row-normalised), RF, protocols A and B
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for key, res, tag in [('protocol_A_document_cv', resA, 'A'), ('protocol_B_title_grouped_cv', resB, 'B')]:
        cm = np.array(res['Random forest']['confusion'], dtype=float)
        cmn = cm / cm.sum(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(6.2, 5.4))
        im = ax.imshow(cmn, cmap='Greys', vmin=0, vmax=1)
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha='right'); ax.set_yticklabels(labels)
        ax.set_xlabel('Predicted register'); ax.set_ylabel('True register')
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f'{cmn[i, j]:.2f}', ha='center', va='center', fontsize=8.5,
                        color='white' if cmn[i, j] > 0.55 else 'black')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        for ext in ['png', 'pdf', 'svg']:
            fig.savefig(os.path.join(a.outdir, f'confusion_rf_{tag}.{ext}'), dpi=300)
        plt.close(fig)

    # Markdown tables
    with open(os.path.join(a.outdir, 'tables.md'), 'w', encoding='utf8') as fh:
        for key, res, title in [('A', resA, 'Protocol A: document-level stratified 5-fold out-of-fold CV'),
                                ('B', resB, 'Protocol B: title-grouped stratified 5-fold out-of-fold CV')]:
            fh.write(f'## {title}\n\n| Model | Macro-F1 | Weighted-F1 | Accuracy |\n|---|---|---|---|\n')
            for name, r in res.items():
                fh.write(f"| {name} | {r['macro_f1']:.2f} | {r['weighted_f1']:.2f} | {r['accuracy']:.2f} |\n")
            mods = [n for n in res if 'baseline' not in n and 'control' not in n]
            fh.write('\n| Register | ' + ' | '.join(mods) + ' |\n|---|' + '---|' * len(mods) + '\n')
            for lab in labels:
                fh.write(f'| {lab} | ' + ' | '.join(f"{res[n]['per_class'][lab]['f1']:.2f}" for n in mods) + ' |\n')
            fh.write('\n')
        fh.write('## Protocol C: balanced subsample, title-grouped CV (mean over reps)\n\n| Model | Macro-F1 (sd) | Weighted-F1 | Accuracy |\n|---|---|---|---|\n')
        for name, r in agg.items():
            fh.write(f"| {name} | {r['macro_f1_mean']:.2f} ({r['macro_f1_sd']:.2f}) | {r['weighted_f1_mean']:.2f} | {r['accuracy_mean']:.2f} |\n")
        fh.write('\n## Feature importance\n\n| Feature | RF impurity | SVM |coef| | Permutation (macro-F1 drop) |\n|---|---|---|---|\n')
        for f in sorted(FEATURES, key=lambda k: -rf_imp[k]):
            fh.write(f'| {f} | {rf_imp[f]:.3f} | {svm_imp[f]:.3f} | {perm_imp[f]:.3f} |\n')
        fh.write('\n## Register profile (means)\n\n')
        fh.write(prof.round(3).to_markdown() + '\n')
    print(json.dumps({k: summary[k] for k in ['n_documents_after_min_tokens', 'class_counts']}, ensure_ascii=False))
    for key in ['protocol_A_document_cv', 'protocol_B_title_grouped_cv']:
        print(key, {n: round(r['macro_f1'], 3) for n, r in summary[key].items()})
    print('balanced', {n: round(r['macro_f1_mean'], 3) for n, r in agg.items()})


if __name__ == '__main__':
    main()
