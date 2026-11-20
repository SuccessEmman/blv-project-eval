"""Failure taxonomy for the BLV caption study (draft, rule-based, for team verification).

Input : the four reviewers' final review sheets (CSV of the Review tab), the caption list
        with model/stratum, and the saved Gemini judge results (for its reasons only).
Output: taxonomy_captions.csv (one row per caption: labels + evidence),
        taxonomy_counts.csv (frequencies by stratum and model).

Which captions count as failures
  EH failure  : at least 3 of the 4 reviewers gave EH 0 (consensus confident error).
  EH contested: exactly 2 of 4 gave EH 0 (reported separately, not labelled).
  TR outcome  : for the 102 captions all four reviewers scored on TR.

How a failure is labelled
  Keyword rules over the caption text and the reviewers' written EH reasons (plus the judge's
  reason as extra evidence). A caption can match several rules; its PRIMARY label is the first
  match in the order below. Rules are deliberately simple so a human can audit them.
"""
import re, sys
import pandas as pd

SHEETS, CAPTIONS, JUDGE = sys.argv[1], sys.argv[2], sys.argv[3]
R = ['Success', 'Ebere', 'Yvonne', 'Samuel']

h = {r: pd.read_csv(f'{SHEETS}/{r}.csv', dtype=str, keep_default_na=False).set_index('caption_id') for r in R}
ids = h['Success'].index
cap = pd.read_csv(CAPTIONS, dtype=str)
norm = lambda s: ' '.join(str(s).split())
meta = {(r.file_name, norm(r.caption)): (r.model, r.stratum) for r in cap.itertuples()}
judge = pd.read_csv(JUDGE).set_index('caption_id')

df = pd.DataFrame(index=ids)
df['file_name'] = h['Success'].file_name
df['caption'] = h['Success'].caption
df['model'] = [('BLIP-2' if 'blip' in meta[(f, norm(c))][0] else 'LLaVA-1.5') for f, c in zip(df.file_name, df.caption)]
df['stratum'] = [meta[(f, norm(c))][1] for f, c in zip(df.file_name, df.caption)]
for r in R:
    df[f'eh_{r}'] = pd.to_numeric(h[r].eh_score)
    df[f'tr_{r}'] = pd.to_numeric(h[r].tr_score, errors='coerce')
df['eh_zeros'] = sum((df[f'eh_{r}'] == 0).astype(int) for r in R)
df['eh_ones'] = sum((df[f'eh_{r}'] == 1).astype(int) for r in R)
df['tr_n'] = sum(df[f'tr_{r}'].notna().astype(int) for r in R)
df['tr_mean'] = df[[f'tr_{r}' for r in R]].mean(axis=1)
df['zero_reasons'] = [' | '.join(h[r].loc[i, 'eh_reason'] for r in R if h[r].loc[i, 'eh_score'] == '0') for i in ids]
df['tr_reasons'] = [' | '.join(h[r].loc[i, 'tr_reason'] for r in R if h[r].loc[i, 'tr_score'] in ('0', '1')) for i in ids]
df['judge_reason'] = judge.reindex(ids).judge_eh_reason.fillna('')
df['words'] = df.caption.str.split().str.len()
df['cut_off'] = (df.model == 'LLaVA-1.5') & ~df.caption.str.strip().str.endswith(('.', '!', '?', '"'))

def has(pat, text): return re.search(pat, text.lower()) is not None
PEOPLE = r"\b(person|people|man|woman|men|women|hand|hands|finger|someone|boy|girl|lady|child|player)s?\b"
RULES = [  # (label, test) in priority order
 ('Degenerate output', lambda x: bool(re.search(r"([\w']+(?:\s+[\w']+){1,3})(?:\W+\1){2,}", x.caption.lower()))),
 ('Confident description of an unreadable photo', lambda x: x.stratum in ('low_light', 'blurry') and has(
     r"dark|pitch|black image|blur|can.?t see|cannot see|unclear|not (too )?clear|nothing (is )?visible|can.?t be deduced|cannot be deduced|can.?t tell|cannot tell|not visible|too blury|no visible", x.zero_reasons + ' ' + x.judge_reason)),
 ('Wrong text, brand or label', lambda x: has(r"brand|spell|label|wording|\breads\b|written|misread|message|button|\btext\b|website|title", x.zero_reasons)),
 ('Invented person or body part', lambda x: has(PEOPLE, x.caption) and has(PEOPLE + r"|no one|nobody|human|face", x.zero_reasons + ' ' + x.judge_reason)),
 ('Invented setting or furniture', lambda x: has(
     r"table|counter|\bbed\b|room|couch|sofa|floor|surface|desk|kitchen|wall|carpet|setting|living|sky|shelf|outside|outdoor|street|chair|rack|patio|window", x.zero_reasons)),
 ('Wrong colour, count or position', lambda x: has(
     r"colou?r|white|black|\bred\b|blue|green|pink|yellow|orange|brown|gr[ae]y|purple|silver|gold|\btwo\b|three|four|\bfive\b|number|\bleft\b|\bright\b|position", x.zero_reasons)),
 ('Wrong or invented main object', lambda x: has(
     r"misidentif|\bnot\b|instead|actually|incorrect|inaccurate|wrong|rather than|labell?ed as|\bno \w|there is no|there.?s no|not visible|hallucinat|isn.?t there|nowhere|unsupported|unsubstantiated|certain", x.zero_reasons + ' ' + x.judge_reason)),
]

def labels(x):
    return [name for name, test in RULES if test(x)]

df['eh_class'] = 'Not a failure'
df.loc[df.eh_zeros == 2, 'eh_class'] = 'Contested (2 of 4 gave 0)'
df.loc[df.eh_zeros >= 3, 'eh_class'] = 'Consensus failure (3+ of 4 gave 0)'
fail = df.eh_zeros >= 3
df['all_labels'] = ''
df['primary_label'] = ''
for i in df.index[fail]:
    L = labels(df.loc[i])
    df.loc[i, 'all_labels'] = '; '.join(L)
    df.loc[i, 'primary_label'] = L[0] if L else 'Other unsupported claim'
df.loc[(df.eh_ones >= 3), 'primary_label'] = 'Vague or over-hedged (3+ of 4 gave 1)'

def tr_class(x):
    if x.tr_n < 4: return ''
    if x.tr_mean == 2: return 'Text read correctly'
    if x.tr_mean < 0.5: return 'Text ignored'
    if has(r"misread|incorrect|wrong|not \"|instead|misspel|inaccurate|not the", x.tr_reasons): return 'Text misread'
    return 'Text partly read (key words missed)'
df['tr_class'] = df.apply(tr_class, axis=1)

out_cols = ['file_name', 'model', 'stratum', 'caption', 'words', 'cut_off'] + [f'eh_{r}' for r in R] + ['eh_zeros', 'eh_class', 'primary_label', 'all_labels', 'tr_mean', 'tr_class', 'zero_reasons', 'tr_reasons', 'judge_reason']
df[out_cols].to_csv('taxonomy_captions.csv')

rows = []
for col, name in [('primary_label', 'EH failure type'), ('tr_class', 'TR outcome')]:
    sub = df[df[col] != '']
    t = pd.crosstab(sub[col], [sub.stratum], margins=True, margins_name='All photos')
    m = pd.crosstab(sub[col], sub.model)
    t = t.join(m)
    t.insert(0, 'dimension', name)
    rows.append(t)
counts = pd.concat(rows)
counts.to_csv('taxonomy_counts.csv')
pd.set_option('display.width', 220)
print(df.eh_class.value_counts().to_string()); print()
print(counts.to_string())
print('\nLLaVA captions cut off at the 64-token limit:', int(df.cut_off.sum()))
