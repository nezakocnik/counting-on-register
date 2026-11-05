// kontext_extract.js
// Per-document numeral counts from Gigafida 2.1 via the CLARIN.SI KonText JSON API.
//
// Run in the browser console on https://www.clarin.si/kontext/query?corpname=gfida21
// (same origin, so no CORS or authentication is needed for anonymous read access).
// For every CQL query below, the script (1) submits it asynchronously, (2) polls until the
// concordance is finished, and (3) requests the frequency distribution of the concordance over
// the structural attribute text.id (one row per Gigafida document; `norm` = document length in
// tokens). The result is a tab-separated table with one row per document (38,310 rows) and one
// column per query, which analyze_gigafida.py turns into the 24 features. When the script
// finishes, the browser downloads it as gigafida_numeral_counts.tsv. A full run takes several
// hours; keep the tab open until it ends.
//
// Tag positions (JOS/MULTEXT-East MSD for Slovene, category K = numeral):
//   K | form (a = Arabic digits, b = word, r = Roman) | type (g = cardinal, v = ordinal,
//   z = pronominal, d = other) | gender (m/z/s, '-' if unmarked) | number (e/d/m) | case (i/r/d/t/m/o)
//   S = noun (number at position 4), P = adjective (number at position 5), Z = punctuation.

const CORP = 'gfida21';

const QUERIES = [
  ['Kag', '[tag="Kag"]'],                       // digit cardinals
  ['Kav', '[tag="Kav"]'],                       // digit ordinals (token includes the dot: "25.")
  ['Kbg', '[tag="Kbg.*"]'],                     // spelled-out cardinals
  ['Kbv', '[tag="Kbv.*"]'],                     // spelled-out ordinals
  ['Kbz', '[tag="Kbz.*"]'],                     // pronominal numerals (en, drug, eden)
  ['Kbd', '[tag="Kbd.*"]'],                     // other numerals (dvojen, trojen, ...)
  ['Krg', '[tag="Krg"]'],                       // Roman cardinals
  ['Krv', '[tag="Krv"]'],                       // Roman ordinals
  ['low', '[tag="Kbg.*" & lemma="dva|trije|štirje|pet|šest|sedem|osem|devet|deset"] | [tag="Kbv.*" & lemma="prvi|tretji|četrti|peti|šesti|sedmi|osmi|deveti|deseti"] | [tag="Ka.*" & word="([0-9]|10)([,.][0-9]+)?\\.?"] | [tag="Kr.*" & word="(I|II|III|IV|V|VI|VII|VIII|IX|X)\\.?"]'],
  ['compound', '[tag="Kbg.*" & lemma=".+in.+"] | [tag="Kbg.*"] [tag="Kbg.*"]'],
  ['case_i', '[tag="Kb...i.*"]'], ['case_r', '[tag="Kb...r.*"]'], ['case_d', '[tag="Kb...d.*"]'],
  ['case_t', '[tag="Kb...t.*"]'], ['case_m', '[tag="Kb...m.*"]'], ['case_o', '[tag="Kb...o.*"]'],
  ['gender', '[tag="Kb.[mzs].*"]'],             // overt gender on the numeral
  ['dual', '(union [tag="Kb..d.*"] (meet [tag="K.*"] [tag="S..d.*|P...d.*"] 1 2))'],
  ['sinit', '<s> [tag="K.*"]'],
  ['sfinal', '[tag="K.*"] [tag="Z"]{0,2} </s>'],
  ['ctx_temporal', '(meet [tag="K.*"] [lemma="leto|mesec|dan|teden|ura|minuta|sekunda|stoletje|desetletje|tisočletje|januar|februar|marec|april|maj|junij|julij|avgust|september|oktober|november|december|ponedeljek|torek|sreda|četrtek|petek|sobota|nedelja|datum|obdobje|starost|star|letnik|rojstvo|smrt|dopoldne|popoldne|zjutraj|zvečer|noč|opoldne|polnoč"] -3 3)'],
  ['ctx_measure', '(meet [tag="K.*"] [lemma="meter|kilometer|centimeter|milimeter|kilogram|gram|tona|liter|deciliter|mililiter|hektar|odstotek|procent|stopinja|kvadratni|kubični|km|m|cm|mm|kg|g|l|dl|ml|ha|%|°|milja|watt|kilovat|volt|amper|hitrost|dolžina|višina|širina|globina|teža|temperatura|mera|količina|delež|razmerje|površina|prostornina|masa|konjski|km/h|hektoliter|kalorija|celzij"] -3 3)'],
  ['ctx_monetary', '(meet [tag="K.*"] [lemma="evro|euro|eur|tolar|sit|dolar|usd|cent|funt|frank|marka|krona|kuna|dinar|cena|plača|proračun|znesek|strošek|dobiček|prihodek|izguba|davek|kredit|posojilo|obresti|vrednost|€|\\$|denar|plačilo|prodaja|nakup|naložba|investicija|kapital|dolg|promet|prihranek|honorar|nagrada|štipendija|najemnina|kazen|globa"] -3 3)'],
  ['ctx_enum', '(meet [tag="K.*"] [lemma="točka|člen|odstavek|alineja|paragraf|poglavje|stran|številka|št.|razred|mesto|krog|kolo|del|tabela|preglednica|slika|verz|kitica|vrstica|stolpec|lekcija|naloga|serija|izdaja|epizoda|sezona|nadstropje|soba|linija|proga|kategorija|skupina|stopnja|raven|nivo|faza|korak|poskus|različica|verzija|zvezek|knjiga|dejanje|prizor|tekma|liga|uvrstitev|finale"] -3 3)'],
];

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function runQuery(cql, maxWait = 900000) {
  const body = { type: 'concQueryArgs', maincorp: CORP, usesubcorp: null, viewmode: 'kwic', pagesize: 40,
    attrs: ['word'], attr_vmode: 'visible-kwic', base_viewattr: 'word', ctxattrs: [], structs: [], refs: [],
    fromp: 1, shuffle: 0,
    queries: [{ qtype: 'advanced', corpname: CORP, query: cql, pcq_pos_neg: 'pos', include_empty: false, default_attr: null }],
    text_types: {}, context: { fc_lemword_wsize: [-5, 5], fc_lemword: '', fc_lemword_type: 'all', fc_pos_wsize: [-5, 5], fc_pos: [], fc_pos_type: 'all' },
    async: true, no_query_history: false };
  const r = await fetch('/kontext/query_submit?format=json', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json();
  if (!j.conc_persistence_op_id) throw new Error('submit failed: ' + JSON.stringify(j.messages));
  const id = j.conc_persistence_op_id, t0 = Date.now();
  let v;
  while (Date.now() - t0 < maxWait) {
    v = await (await fetch(`/kontext/view?maincorp=${CORP}&q=~${id}&format=json`)).json();
    if (v.finished) break;
    await sleep(2000);
  }
  return { id, n: v.concsize };
}

async function freqByDoc(concId, fcrit = 'text.id 0') {
  const r = await fetch(`/kontext/freqs?q=~${concId}&fcrit=${encodeURIComponent(fcrit)}&flimit=0&fmaxitems=100000&format=json`);
  const j = await r.json();
  if (!j.Blocks || !j.Blocks[0]) throw new Error('freqs failed: ' + JSON.stringify(j.messages));
  return j.Blocks[0].Items;
}

async function main() {
  // Document table: length (norm) and metadata from a <text/> query.
  const docs = {};
  const textConc = await runQuery('<text/>');
  for (const it of await freqByDoc(textConc.id)) docs[it.Word[0].n] = { ntok: it.norm, cls: '' };
  for (const it of await freqByDoc(textConc.id, 'text.id 0 text.class 0')) {   // multi-valued: keep the most specific value
    const d = docs[it.Word[0].n], c = it.Word[1].n; if (d && c.length > d.cls.length) d.cls = c;
  }
  for (const [key, attr] of [['title', 'text.title'], ['source', 'text.source'], ['year', 'text.year'], ['publisher', 'text.publisher']])
    for (const it of await freqByDoc(textConc.id, `text.id 0 ${attr} 0`)) { const d = docs[it.Word[0].n]; if (d) d[key] = it.Word[1].n; }

  // One frequency distribution over text.id per query.
  const counts = {};
  for (const [label, cql] of QUERIES) {
    const conc = await runQuery(cql);
    const m = {}; let s = 0;
    for (const it of await freqByDoc(conc.id)) { m[it.Word[0].n] = it.freq; s += it.freq; }
    if (s !== conc.n) console.warn(label, 'sum of per-document frequencies', s, '!= concordance size', conc.n);
    counts[label] = m;
    console.log(label, conc.n);
  }

  // Tab-separated output; tabs and line breaks inside metadata values are replaced by spaces.
  const san = s => String(s == null ? '' : s).replace(/[\t\r\n|]+/g, ' ').replace(/\s+/g, ' ').trim();
  const labels = QUERIES.map(q => q[0]);
  const lines = [['id', 'cls', 'title', 'source', 'year', 'publisher', 'ntok', ...labels].join('\t')];
  for (const id of Object.keys(docs).sort()) {
    const d = docs[id];
    lines.push([id, d.cls, san(d.title), san(d.source), san(d.year), san(d.publisher), d.ntok, ...labels.map(l => counts[l][id] || 0)].join('\t'));
  }
  window.__TSV = lines.join('\n');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([window.__TSV + '\n'], { type: 'text/tab-separated-values' }));
  a.download = 'gigafida_numeral_counts.tsv';
  document.body.appendChild(a); a.click(); a.remove();
  return lines.length - 1;
}

main().then(n => console.log('documents:', n));
