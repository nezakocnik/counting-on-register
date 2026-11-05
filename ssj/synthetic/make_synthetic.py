#!/usr/bin/env python3
"""Synthetic CoNLL-U corpus for validating the ssj500k pipeline.

Four invented registers whose only differences lie in their numerals (density, digits vs. words,
ordinals, case, gender, dual, context). Everything else is identical filler. If extract_features.py
and analyze_real.py are implemented correctly, the classifiers must recover the four registers
almost perfectly; this checks the code, not the language.

Usage: python make_synthetic.py --out synthetic.conllu  [--docs-per-register 42 --tokens 600 --seed 42]
"""
import argparse, random

FILLER = [('hiša', 'hiša', 'NOUN', 'Ncfsn', 'Case=Nom|Gender=Fem|Number=Sing'),
          ('je', 'biti', 'AUX', 'Va-r3s-n', 'Mood=Ind|Number=Sing|Person=3|Polarity=Pos|Tense=Pres|VerbForm=Fin'),
          ('lepa', 'lep', 'ADJ', 'Agpfsn', 'Case=Nom|Degree=Pos|Gender=Fem|Number=Sing'),
          ('in', 'in', 'CCONJ', 'Cc', '_'),
          ('mesto', 'mesto', 'NOUN', 'Ncnsn', 'Case=Nom|Gender=Neut|Number=Sing'),
          ('gre', 'iti', 'VERB', 'Vmbr3s', 'Aspect=Imp|Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin'),
          ('tja', 'tja', 'ADV', 'Rgp', 'Degree=Pos')]
CONTEXT = {'temporal': ('leto', 'NOUN', 'Ncnsg', 'Case=Gen|Gender=Neut|Number=Sing'),
           'monetary': ('evro', 'NOUN', 'Ncmpg', 'Case=Gen|Gender=Masc|Number=Plur'),
           'enum': ('člen', 'NOUN', 'Ncmsg', 'Case=Gen|Gender=Masc|Number=Sing'),
           'measure': ('meter', 'NOUN', 'Ncmpg', 'Case=Gen|Gender=Masc|Number=Plur')}
CASES = ['Nom', 'Gen', 'Dat', 'Acc', 'Loc', 'Ins']
# per register: numerals per 1,000 tokens, P(digit), P(ordinal), P(dual 'dva'), case weights, context
PROFILES = {
    'narrative':  dict(density=8,  digit=0.05, ordinal=0.10, dual=0.30, cases=[4, 1, 0, 4, 1, 1], ctx='temporal'),
    'report':     dict(density=30, digit=0.60, ordinal=0.15, dual=0.10, cases=[3, 2, 0, 3, 2, 1], ctx='temporal'),
    'statute':    dict(density=55, digit=0.65, ordinal=0.60, dual=0.02, cases=[1, 5, 0, 1, 2, 0], ctx='enum'),
    'commercial': dict(density=40, digit=0.85, ordinal=0.05, dual=0.05, cases=[2, 3, 1, 2, 1, 1], ctx='monetary'),
}


def numeral(rng, p):
    """Return the token list for one numeral (and possibly its dual noun)."""
    case = rng.choices(CASES, weights=p['cases'])[0]
    if rng.random() < p['dual']:
        return [('dva', 'dva', 'NUM', 'Mlcmd' + case[0].lower(), f'Case={case}|Gender=Masc|NumForm=Word|NumType=Card|Number=Dual'),
                ('psa', 'pes', 'NOUN', 'Ncmd', f'Case={case}|Gender=Masc|Number=Dual')]
    ordinal = rng.random() < p['ordinal']
    if rng.random() < p['digit']:
        v = rng.randint(1, 2000)
        if ordinal:
            return [(f'{v}.', f'{v}.', 'ADJ', 'Mdo', 'NumForm=Digit|NumType=Ord')]
        return [(str(v), str(v), 'NUM', 'Mdc', 'NumForm=Digit|NumType=Card')]
    if ordinal:
        w = rng.choice(['prvi', 'tretji', 'peti', 'dvajseti'])
        return [(w, w, 'ADJ', 'Mlomsn', f'Case={case}|Gender=Masc|NumType=Ord|Number=Sing')]
    w = rng.choice(['trije', 'pet', 'sedem', 'dvajset', 'petindvajset'])
    return [(w, w, 'NUM', 'Mlc-p' + case[0].lower(), f'Case={case}|NumForm=Word|NumType=Card|Number=Plur')]


def document(rng, p, ntok):
    toks = []
    while len(toks) < ntok:
        if rng.random() < p['density'] / 1000 * 1.6:   # ~1.6 tokens per numeral event incl. context noun
            toks += numeral(rng, p)
            if rng.random() < 0.5:
                lem, upos, xpos, feats = CONTEXT[p['ctx']]
                toks.append((lem, lem, upos, xpos, feats))
        else:
            toks.append(rng.choice(FILLER))
    return toks[:ntok]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='synthetic.conllu')
    ap.add_argument('--docs-per-register', type=int, default=42)
    ap.add_argument('--tokens', type=int, default=600)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    with open(a.out, 'w', encoding='utf8') as fh:
        for reg, p in PROFILES.items():
            for d in range(a.docs_per_register):
                toks = document(rng, p, a.tokens)
                fh.write(f'# newdoc id = {reg}{d}\n')
                for s in range(0, len(toks), 12):            # 12-token sentences ending in a full stop
                    sent = toks[s:s + 12] + [('.', '.', 'PUNCT', 'Z', '_')]
                    fh.write(f'# sent_id = {reg}{d}.{s // 12}\n# document_genre = {reg}\n')
                    for i, (form, lemma, upos, xpos, feats) in enumerate(sent, 1):
                        fh.write(f'{i}\t{form}\t{lemma}\t{upos}\t{xpos}\t{feats}\t0\t{"root" if i == 1 else "dep"}\t_\t_\n')
                    fh.write('\n')
    print('written', a.out)


if __name__ == '__main__':
    main()
