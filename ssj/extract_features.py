#!/usr/bin/env python3
"""Numeral features per text window from a CoNLL-U file (SSJ UD treebank of Slovenian).

The 24 features are defined exactly as in ../gigafida/analyze_gigafida.py, so that the two
experiments are comparable. Instead of corpus queries, the counts are taken directly from the
manually annotated tokens:

  * A token is a numeral if its UPOS is NUM, if its features contain NumType, or if its
    MULTEXT-East tag (XPOS) starts with M. The MULTEXT-East numeral tags correspond one-to-one to
    the JOS tags used for Gigafida: Mdc = Kag (digit cardinal), Mdo = Kav (digit ordinal),
    Mlc = Kbg (word cardinal), Mlo = Kbv (word ordinal), Mlp = Kbz (pronominal: en, eden, drug),
    Mls = Kbd (other: dvojen, trojen), Mrc = Krg (Roman cardinal), Mro = Krv (Roman ordinal).
  * Case, gender and number are read from the Universal Features of word numerals (Ml*).
  * Documents (# newdoc id) are cut into consecutive windows of --window tokens; a trailing
    remainder is kept as its own window only if it has at least --min-window tokens. The register
    label is the document's # document_genre.

Output: one CSV row per window (document_id, window, register, ntok, the raw counts, 24 features).
"""
import argparse, csv, re

CTX = {
    'temporal': 'leto|mesec|dan|teden|ura|minuta|sekunda|stoletje|desetletje|tisočletje|januar|februar|marec|april|maj|junij|julij|avgust|september|oktober|november|december|ponedeljek|torek|sreda|četrtek|petek|sobota|nedelja|datum|obdobje|starost|star|letnik|rojstvo|smrt|dopoldne|popoldne|zjutraj|zvečer|noč|opoldne|polnoč',
    'measure': 'meter|kilometer|centimeter|milimeter|kilogram|gram|tona|liter|deciliter|mililiter|hektar|odstotek|procent|stopinja|kvadratni|kubični|km|m|cm|mm|kg|g|l|dl|ml|ha|%|°|milja|watt|kilovat|volt|amper|hitrost|dolžina|višina|širina|globina|teža|temperatura|mera|količina|delež|razmerje|površina|prostornina|masa|konjski|km/h|hektoliter|kalorija|celzij',
    'monetary': 'evro|euro|eur|tolar|sit|dolar|usd|cent|funt|frank|marka|krona|kuna|dinar|cena|plača|proračun|znesek|strošek|dobiček|prihodek|izguba|davek|kredit|posojilo|obresti|vrednost|€|$|denar|plačilo|prodaja|nakup|naložba|investicija|kapital|dolg|promet|prihranek|honorar|nagrada|štipendija|najemnina|kazen|globa',
    'enum': 'točka|člen|odstavek|alineja|paragraf|poglavje|stran|številka|št.|razred|mesto|krog|kolo|del|tabela|preglednica|slika|verz|kitica|vrstica|stolpec|lekcija|naloga|serija|izdaja|epizoda|sezona|nadstropje|soba|linija|proga|kategorija|skupina|stopnja|raven|nivo|faza|korak|poskus|različica|verzija|zvezek|knjiga|dejanje|prizor|tekma|liga|uvrstitev|finale',
}
CTX = {k: set(v.split('|')) for k, v in CTX.items()}   # same lemma lists as kontext_extract.js

LOW_CARD = set('dva trije štirje pet šest sedem osem devet deset'.split())
LOW_ORD = set('prvi tretji četrti peti šesti sedmi osmi deveti deseti'.split())
LOW_DIGIT = re.compile(r'^([0-9]|10)([,.][0-9]+)?\.?$')
LOW_ROMAN = re.compile(r'^(I|II|III|IV|V|VI|VII|VIII|IX|X)\.?$')
COMPOUND = re.compile(r'^.+in.+$')
TAGMAP = {'Mdc': 'Kag', 'Mdo': 'Kav', 'Mlc': 'Kbg', 'Mlo': 'Kbv', 'Mlp': 'Kbz', 'Mls': 'Kbd', 'Mrc': 'Krg', 'Mro': 'Krv'}
CASES = {'Nom': 'case_i', 'Gen': 'case_r', 'Dat': 'case_d', 'Acc': 'case_t', 'Loc': 'case_m', 'Ins': 'case_o'}
COUNTS = ['Kag', 'Kav', 'Kbg', 'Kbv', 'Kbz', 'Kbd', 'Krg', 'Krv', 'num_other', 'low', 'compound',
          'case_i', 'case_r', 'case_d', 'case_t', 'case_m', 'case_o', 'gender', 'dual', 'sinit', 'sfinal',
          'ctx_temporal', 'ctx_measure', 'ctx_monetary', 'ctx_enum',
          'ord_ud', 'digit_ud', 'high_digit', 'dual_self']
FEATURES = ['num_per_1k', 'card_per_1k', 'ord_per_1k', 'card_ord_ratio', 'low_per_1k', 'high_per_1k',
            'prop_digit', 'prop_compound',
            'prop_case_nom', 'prop_case_gen', 'prop_case_dat', 'prop_case_acc', 'prop_case_loc', 'prop_case_ins',
            'prop_gender_agr', 'dual_per_1k',
            'prop_pos_initial', 'prop_pos_medial', 'prop_pos_final',
            'prop_ctx_temporal', 'prop_ctx_measure', 'prop_ctx_monetary', 'prop_ctx_enum', 'prop_ctx_other']


def read_documents(path, register_field):
    """Yield (doc_id, register, tokens); each token is a dict with sentence-position flags."""
    doc_id, reg, toks, sent = None, None, [], []
    def flush_sent():
        if sent:
            sent[0]['s_first'] = True
            sent[-1]['s_last'] = True
            toks.extend(sent); sent.clear()
    for line in open(path, encoding='utf8'):
        line = line.rstrip('\n')
        if line.startswith('# newdoc id'):
            flush_sent()
            if doc_id is not None and toks: yield doc_id, reg, toks
            doc_id, reg, toks = line.split('=', 1)[1].strip(), None, []
        elif line.startswith('# ' + register_field):
            reg = line.split('=', 1)[1].strip()
        elif line.startswith('#'):
            continue
        elif not line.strip():
            flush_sent()
        else:
            f = line.split('\t')
            if '-' in f[0] or '.' in f[0]:
                continue
            feats = dict(x.split('=', 1) for x in f[5].split('|')) if f[5] != '_' else {}
            sent.append({'form': f[1], 'lemma': f[2], 'upos': f[3], 'xpos': f[4], 'feats': feats,
                         's_first': False, 's_last': False})
    flush_sent()
    if doc_id is not None and toks: yield doc_id, reg, toks


def digit_value(form):
    """Numeric value of a digit token, Slovene style: '2.000' = 2000, '1,5' = 1.5, '20.' = 20."""
    try:
        return float(form.rstrip('.').replace('.', '').replace(',', '.'))
    except ValueError:
        return None


def token_flags(toks):
    """Per-token counts, computed on the whole document so that ±3 windows and the
    'followed by' checks are not cut at window boundaries."""
    n = len(toks)
    is_num = [t['upos'] == 'NUM' or 'NumType' in t['feats'] or t['xpos'].startswith('M') for t in toks]
    is_punct = [t['upos'] == 'PUNCT' or t['xpos'] == 'Z' for t in toks]
    out = []
    for i, t in enumerate(toks):
        c = dict.fromkeys(COUNTS, 0)
        if is_num[i]:
            x, fe = t['xpos'], t['feats']
            k = TAGMAP.get(x[:3])
            c[k if k else 'num_other'] += 1
            if (k == 'Kbg' and t['lemma'] in LOW_CARD) or (k == 'Kbv' and t['lemma'] in LOW_ORD) \
               or (x[:2] == 'Md' and LOW_DIGIT.match(t['form'])) or (x[:2] == 'Mr' and LOW_ROMAN.match(t['form'])):
                c['low'] += 1
            if k == 'Kbg' and COMPOUND.match(t['lemma']):
                c['compound'] += 1
            if k == 'Kbg' and i + 1 < n and TAGMAP.get(toks[i + 1]['xpos'][:3]) == 'Kbg':
                c['compound'] += 1          # second branch of the CQL union: two adjacent word cardinals
            if x[:2] == 'Ml':
                if fe.get('Case') in CASES: c[CASES[fe['Case']]] += 1
                if 'Gender' in fe: c['gender'] += 1
            dual = x[:2] == 'Ml' and fe.get('Number') == 'Dual'
            for j in (i + 1, i + 2):
                if j < n and toks[j]['xpos'][:1] in ('N', 'A') and toks[j]['feats'].get('Number') == 'Dual':
                    dual = True
            c['dual'] += int(dual)
            # Definitions used for the ssj500k experiment reported in the article (Table 9):
            c['ord_ud'] += int(fe.get('NumType') == 'Ord')
            c['digit_ud'] += int(x[:2] == 'Md')
            c['high_digit'] += int(x[:2] == 'Md' and (digit_value(t['form']) or 0) >= 11)
            c['dual_self'] += int(fe.get('Number') == 'Dual')
            if t['s_first']: c['sinit'] += 1
            j = i + 1; k2 = 0             # numeral + up to two punctuation tokens + end of sentence
            if t['s_last']:
                c['sfinal'] += 1
            else:
                while j < n and is_punct[j] and k2 < 2:
                    if toks[j]['s_last']: c['sfinal'] += 1; break
                    j += 1; k2 += 1
            near = {toks[j]['lemma'] for j in range(max(0, i - 3), min(n, i + 4)) if j != i}
            for key, lex in CTX.items():
                if near & lex: c['ctx_' + key] += 1
        out.append(c)
    return out


def features(c, ntok):
    """The 24 features. Four of them follow the definitions with which the ssj500k results in the
    article were computed, which use the Universal Features directly and differ slightly from the
    Gigafida query-based versions (see README):
      cardinals      = all numerals that are not NumType=Ord; card_ord_ratio = cardinals / max(ordinals, 1)
      prop_digit     = numerals with NumForm=Digit / all numerals
      high_per_1k    = digit-written numerals with value >= 11, per 1,000 tokens
      dual_per_1k    = numerals that are themselves marked Number=Dual, per 1,000 tokens"""
    num = sum(c[k] for k in ['Kag', 'Kav', 'Kbg', 'Kbv', 'Kbz', 'Kbd', 'Krg', 'Krv', 'num_other'])
    ordn = c['ord_ud']
    card = num - ordn
    ncase = sum(c[k] for k in CASES.values())
    div = lambda a, b: a / b if b else 0.0
    f = {
        'num_per_1k': 1000 * num / ntok, 'card_per_1k': 1000 * card / ntok, 'ord_per_1k': 1000 * ordn / ntok,
        'card_ord_ratio': card / max(ordn, 1), 'low_per_1k': 1000 * c['low'] / ntok,
        'high_per_1k': 1000 * c['high_digit'] / ntok,
        'prop_digit': div(c['digit_ud'], num), 'prop_compound': div(c['compound'], c['Kbg']),
    }
    for col, lab in zip(CASES.values(), ['nom', 'gen', 'dat', 'acc', 'loc', 'ins']):
        f['prop_case_' + lab] = div(c[col], ncase)
    f['prop_gender_agr'] = div(c['gender'], num)
    f['dual_per_1k'] = 1000 * c['dual_self'] / ntok
    f['prop_pos_initial'] = div(c['sinit'], num)
    f['prop_pos_final'] = div(c['sfinal'], num)
    f['prop_pos_medial'] = max(1 - f['prop_pos_initial'] - f['prop_pos_final'], 0.0)
    for k in ['temporal', 'measure', 'monetary', 'enum']:
        f['prop_ctx_' + k] = min(div(c['ctx_' + k], num), 1.0)
    f['prop_ctx_other'] = max(1 - sum(f['prop_ctx_' + k] for k in ['temporal', 'measure', 'monetary', 'enum']), 0.0)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conllu', required=True)
    ap.add_argument('--register-field', default='document_genre')
    ap.add_argument('--window', type=int, default=600)
    ap.add_argument('--min-window', type=int, default=300)
    ap.add_argument('--out', default='features.csv')
    a = ap.parse_args()
    rows = []
    for doc_id, reg, toks in read_documents(a.conllu, a.register_field):
        flags = token_flags(toks)
        for w, start in enumerate(range(0, len(toks), a.window)):
            part = flags[start:start + a.window]
            if len(part) < a.window and len(part) < a.min_window:
                continue
            c = {k: sum(t[k] for t in part) for k in COUNTS}
            row = {'document_id': doc_id, 'window': w, 'register': reg or '', 'ntok': len(part), **c}
            row.update(features(c, len(part)))
            rows.append(row)
    with open(a.out, 'w', newline='', encoding='utf8') as fh:
        wr = csv.DictWriter(fh, fieldnames=['document_id', 'window', 'register', 'ntok'] + COUNTS + FEATURES)
        wr.writeheader(); wr.writerows(rows)
    print('windows:', len(rows), '->', a.out)


if __name__ == '__main__':
    main()
