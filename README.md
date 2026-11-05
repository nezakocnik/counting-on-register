# Counting on register: numeral distribution as a predictor of text register in Slovenian

Code and data for the article *Counting on register: Numeral distribution as a predictor of text register in Slovenian* (Kočnik, under review).

The study asks whether numerals alone can tell you what kind of text you are reading. It uses 24 numeral-based features (density, cardinal/ordinal type, digits vs. words, compounds, case, gender agreement, dual, sentence position, context) to classify texts into registers. There are two experiments:

- **Gigafida 2.1** (main experiment): every document of the reference corpus of written standard Slovene, six registers (fiction, non-fiction, legislative, magazine, newspaper, web), 35,683 documents.
- **SSJ** (replication): the manually annotated Universal Dependencies treebank of Slovenian (release 2.18), five registers, 389 windows of 600 tokens.

Headline result: a random forest reaches macro-F1 0.74 under document-level cross-validation and 0.66 when whole periodicals and books are held out (majority baseline 0.12).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
<!-- Replace XXXXXXX with the concept DOI Zenodo gives you after the first release. -->

## Contents

```
gigafida/
  kontext_extract.js            browser script: runs the 24 CQL queries against CLARIN.SI KonText
  analyze_gigafida.py           builds the features, runs protocols A/B/C, baselines, importances
  data/
    gigafida_numeral_counts.tsv per-document counts (38,310 documents × 24 counts + metadata)
  results/
    gigafida_features.csv       feature matrix (35,683 documents × 24 features + metadata)
    tables.md                   all result tables (protocols A, B, C; importances; profile)
    summary.json                everything in one file: corpus sizes, scores, confusion matrices
    stage_A/B/C/imp.json        raw results of each step
    register_profile_*.csv      per-register means and medians of the 24 features
    confusion_rf_A/B.png/.pdf   confusion matrices (Figure 1 of the article = confusion_rf_B)
ssj/
  extract_features.py           the 24 features per 600-token window, from the treebank's CoNLL-U
  analyze_real.py               classifiers, baselines, importances, profile, confusion matrix
  results/
    features.csv                feature matrix (404 windows; 389 in the five modelled registers)
    tables.md, summary.json     scores (Table 8 of the article), importances, profile (Table 9)
    register_profile_means.csv  per-register means of the 24 features
    confusion_rf.png/.pdf       confusion matrix of the random forest
  synthetic/
    make_synthetic.py           generator for the synthetic validation corpus (Section 4.7)
    results/                    scores on the synthetic corpus
```

## Requirements

Python 3.11 or newer (tested with 3.13). Install the exact package versions used for the article:

```
pip install -r requirements.txt
```

The versions matter for one model. scikit-learn 1.9 changed how class weights are handled in
random forests, so with 1.9 the class-weighted random forest scores differ slightly from those
reported (e.g. Protocol B macro-F1 0.64 instead of 0.66). All other models, including the plain
random forest behind the headline results, give identical numbers in 1.8 and 1.9.

## Reproducing the results

All runs use the fixed random seed 42.

### Gigafida 2.1

The count table is included, so you can go straight to the analysis:

```
cd gigafida
python analyze_gigafida.py --counts data/gigafida_numeral_counts.tsv --outdir results \
       --min-tokens 300 --folds 5 --seed 42 --balanced-reps 3 --stage all
```

The full run takes about 10 minutes on a two-core laptop. On a slow machine you can run the steps one
at a time (`--stage A`, then `B`, `C`, `imp`, and finally `report`, which writes the tables and figures).

To rebuild the count table from the corpus itself, open
<https://www.clarin.si/kontext/query?corpname=gfida21> in a browser, open the developer
console, paste the contents of `kontext_extract.js` and run it. It submits each query, waits for the
concordance, retrieves the frequency distribution over `text.id`, and downloads the table as
`gigafida_numeral_counts.tsv`. The console warns if any count column does not sum to the
concordance size of its query.

### SSJ treebank

The treebank is not included here. Get release 2.18 from GitHub and join its three files:

```
git clone --branch r2.18 --depth 1 https://github.com/UniversalDependencies/UD_Slovenian-SSJ.git
cat UD_Slovenian-SSJ/sl_ssj-ud-train.conllu UD_Slovenian-SSJ/sl_ssj-ud-dev.conllu \
    UD_Slovenian-SSJ/sl_ssj-ud-test.conllu > ssj_all.conllu
```

(Without git, download the three files from the `r2.18` tag on GitHub.) Then:

```
cd ssj
python extract_features.py --conllu ../ssj_all.conllu --register-field document_genre \
       --window 600 --min-window 300 --out results/features.csv
python analyze_real.py --features results/features.csv --outdir results --min-class 20 --folds 5 --seed 42
```

Each document is cut into consecutive 600-token windows; a final remainder is kept only if it has
at least 300 tokens. This gives 404 windows; the registers `internet` (1 window) and `other`
(14 windows) are too small and are dropped, leaving 389 windows from 265 documents.

Four features are read directly from the Universal Features and so differ slightly from their
Gigafida counterparts: cardinals are all numerals not marked `NumType=Ord`, and the
cardinal-to-ordinal ratio divides them by the number of ordinals (minimum one); the digit
proportion is taken over all numerals; the high-range rate counts digit-written numerals of 11 and
above (`2.000` = 2000); and the dual rate counts numerals that are themselves marked
`Number=Dual`. See the docstrings in `extract_features.py`.

With only 265 documents, the scores depend somewhat on how documents fall into the five folds:
other fold assignments move macro-F1 by up to 0.04 without changing the pattern. The reported
numbers use the seed 42 throughout.

Synthetic validation (Section 4.7 of the article):

```
cd ssj
python synthetic/make_synthetic.py --out synthetic.conllu
python extract_features.py --conllu synthetic.conllu --out synthetic_features.csv
python analyze_real.py --features synthetic_features.csv --outdir synthetic/results
```

## Data sources

- Gigafida 2.1: Krek et al. (2020), *Gigafida 2.0: The reference corpus of written standard Slovene*, LREC 2020; corpus at <http://hdl.handle.net/11356/2055>. Accessed through the CLARIN.SI KonText concordancer (corpus `gfida21`). The full text is not distributable; this repository contains only per-document counts and document metadata.
- SSJ: UD_Slovenian-SSJ, release 2.18, <https://github.com/UniversalDependencies/UD_Slovenian-SSJ> (CC BY-SA 4.0); Dobrovoljc and Ljubešić (2022), *Extending the SSJ Universal Dependencies treebank for Slovenian: Was it worth it?*, LAW-XVI. It is the Universal Dependencies edition of ssj500k (Krek et al. 2020), extended with Wikipedia texts from the ELEXIS corpus.

## Licence

- Code: MIT (see `LICENSE`).
- Data and results: see `LICENSE-DATA.md`.

## How to cite

Please cite the article (details will be added on publication) and this repository. GitHub's
**Cite this repository** button (from `CITATION.cff`) and the Zenodo record give the citation in
APA and BibTeX.
