"""Reproduce the main BLV human-score analysis from bundled inputs.

Offline: no API calls, image downloads, or model generation.
"""
from pathlib import Path
import hashlib
import itertools
import json
import platform
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr, kendalltau
from agreement_methods import pair_result, matrix, kappa

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'inputs'
OUT = ROOT / 'results'
RATERS = ['Ebere', 'Samuel', 'Success', 'Yvonne']
ITERATIONS = 1000
WORKBOOK_URL = 'https://docs.google.com/spreadsheets/d/1C49HiFCjuVMTiACyT-5jtqgAtTHDLiv8/edit'

def read(name):
    return json.loads((INPUT / name).read_text())

def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))

def norm(x):
    return ' '.join(str(x).split())

def text(x):
    return '' if pd.isna(x) else str(x)

def extract(canonical):
    path = ROOT / 'BLV_Combined_Human_Ratings.xlsx'
    scores = pd.read_excel(path, sheet_name='Combined scores')
    reasons = pd.read_excel(path, sheet_name='Reasons').set_index('caption_id', verify_integrity=True)
    assert len(scores) == 500 and scores.caption_id.is_unique
    assert set(scores.caption_id) == set(canonical) == set(reasons.index)
    human = []
    for _, row in scores.iterrows():
        cid = row.caption_id
        c = canonical[cid]
        assert row.file_name == c['file_name']
        assert norm(row.caption) == norm(c['caption']) == norm(reasons.loc[cid, 'caption'])
        assert reasons.loc[cid, 'file_name'] == row.file_name
        for rater in RATERS:
            eh = row['EH ' + rater]
            app = str(row['TR applicable ' + rater]).strip().lower()
            tr = row['TR ' + rater]
            assert eh in (0, 1, 2), (cid, rater, 'EH', eh)
            assert app in ('yes', 'no'), (cid, rater, app)
            assert (app == 'yes' and tr in (0, 1, 2)) or (app == 'no' and pd.isna(tr)), (cid, rater, tr)
            human.append(dict(caption_id=cid, file_name=c['file_name'], caption=c['caption'],
                rater=rater, eh_score=int(eh), tr_applicable=app.title(),
                tr_score=int(tr) if app == 'yes' else 'N/A',
                eh_reason=text(reasons.loc[cid, 'EH reason ' + rater]),
                tr_reason=text(reasons.loc[cid, 'TR reason ' + rater]),
                review_flag=text(reasons.loc[cid, 'Review flag ' + rater]),
                rubric_version=text(reasons.loc[cid, 'Rubric version ' + rater]),
                source_url=WORKBOOK_URL))
    # Compare workbook aggregates with independently recomputed means.
    h = pd.DataFrame(human)
    for cid, group in h.groupby('caption_id'):
        row = scores.set_index('caption_id').loc[cid]
        assert np.isclose(group.eh_score.mean(), row['EH mean'])
        numeric = pd.to_numeric(group.tr_score, errors='coerce').dropna()
        assert len(numeric) == row['TR raters scored']
        assert (not len(numeric) and pd.isna(row['TR mean'])) or np.isclose(numeric.mean(), row['TR mean'])
    return human

def merge(human, inputs, judge, lookup):
    h = pd.DataFrame(human)
    assert len(h) == 2000 and not h.duplicated(['caption_id', 'rater']).any()
    h['tr_numeric'] = pd.to_numeric(h.tr_score, errors='coerce')
    means = h.groupby('caption_id').agg(human_eh=('eh_score', 'mean'), eh_n=('eh_score', 'count'),
                                        human_tr=('tr_numeric', 'mean'), tr_n=('tr_numeric', 'count'))
    assert means.eh_n.eq(4).all()
    j = pd.DataFrame([dict(caption_id=r['caption_id'], judge_eh=r['scores']['eh_score'],
            judge_tr=r['scores']['tr_score'], judge_text_present=r['scores']['text_present']) for r in judge])
    d = pd.DataFrame(inputs).merge(means, on='caption_id', validate='one_to_one').merge(j, on='caption_id', validate='one_to_one')
    d['caption_model'] = d.caption_id.map(lookup)
    d['generation_flag'] = d.caption_id.eq('C0104')
    return d

def statistic(x, y, method):
    if len(set(x)) < 2 or len(set(y)) < 2:
        return np.nan
    return float(spearmanr(x, y).statistic if method == 'spearman' else kendalltau(x, y, variant='b').statistic)

def correlation(d, x, y, method):
    seed = 20261116 if method == 'spearman' else 20261117
    # Stable sorting does not change the statistic; image blocks retain pairs.
    d = d.sort_values(['file_name', 'caption_id']).reset_index(drop=True)
    blocks = [g.index.to_numpy() for _, g in d.groupby('file_name', sort=True)]
    a, b = d[x].to_numpy(), d[y].to_numpy()
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(ITERATIONS):
        ix = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        v = statistic(a[ix], b[ix], method)
        if np.isfinite(v):
            boot.append(v)
    assert len(boot) >= ITERATIONS * .8
    ci = np.quantile(boot, [.025, .975])
    return dict(caption_pairs=len(d), unique_images=int(d.file_name.nunique()),
                estimate=statistic(a, b, method), ci_95_lower=float(ci[0]), ci_95_upper=float(ci[1]),
                valid_bootstrap_samples=len(boot))

def spearman_results(data, version):
    result = []
    for model in ['pooled'] + sorted(data.caption_model.unique()):
        d = data if model == 'pooled' else data.loc[data.caption_model.eq(model)]
        for metric in ['eh', 'tr']:
            eligible = d if metric == 'eh' else d.loc[d.tr_n.eq(4) & d.judge_tr.notna()]
            result.append(dict(version=version, caption_model=model, metric=metric, exclusion='none',
                **correlation(eligible, 'human_' + metric, 'judge_' + metric, 'spearman')))
    result.append(dict(version=version, caption_model='pooled', metric='eh', exclusion='C0104',
        **correlation(data.loc[~data.generation_flag], 'human_eh', 'judge_eh', 'spearman')))
    return result

def coverage(d):
    return dict(scored_captions=len(d), human_reviews=2000,
                human_all4_TR_applicable=int(d.tr_n.eq(4).sum()),
                human_TR_applicability_disagreements=int(d.tr_n.between(1, 3).sum()),
                TR_correlation_pairs=int((d.tr_n.eq(4) & d.judge_tr.notna()).sum()),
                human_all4_TR_yes_but_judge_NA=int((d.tr_n.eq(4) & d.judge_tr.isna()).sum()),
                human_all4_TR_no_but_judge_yes=int((d.tr_n.eq(0) & d.judge_tr.notna()).sum()),
                generation_flagged_caption_ids=['C0104'])

def agreement(human):
    idx = {(r['rater'], r['caption_id']): r for r in human}
    ids = sorted({r['caption_id'] for r in human})
    results = []
    for a, b in itertools.combinations(RATERS, 2):
        for metric in ['EH', 'TR_applicability', 'TR_score_both_applicable', 'EH_excluding_C0104']:
            items = []
            for cid in ids:
                x, y = idx[a, cid], idx[b, cid]
                ax, ay = x['tr_applicable'] == 'Yes', y['tr_applicable'] == 'Yes'
                if metric == 'TR_score_both_applicable' and not (ax and ay):
                    continue
                if metric == 'EH_excluding_C0104' and cid == 'C0104':
                    continue
                sx, sy = (int(ax), int(ay)) if metric == 'TR_applicability' else ((x['tr_score'], y['tr_score']) if metric.startswith('TR_score') else (x['eh_score'], y['eh_score']))
                items.append((x['file_name'], sx, sy))
            results.append(dict(reviewer_1=a, reviewer_2=b, metric=metric,
                                **pair_result(items, 2 if metric == 'TR_applicability' else 3)))
    return results

def main():
    OUT.mkdir(exist_ok=True)
    inputs = read('judge_inputs.json')
    canonical = {r['caption_id']: r for r in inputs}
    lookup = read('model_lookup.json')
    judge = [json.loads(x) for x in (INPUT / 'judge_results.jsonl').read_text().splitlines() if x.strip()]
    config = read('experiment_config.json')
    assert len(judge) == len({r['caption_id'] for r in judge}) == 500
    assert set(canonical) == {r['caption_id'] for r in judge} == set(lookup)
    assert {r['fingerprint'] for r in judge} == {config['fingerprint']}
    assert hashlib.sha256((INPUT / 'judge_inputs.json').read_bytes()).hexdigest() == config['config']['input_hash']
    for r in judge:
        assert r['file_name'] == canonical[r['caption_id']]['file_name']
        assert r['scores']['eh_score'] in (0, 1, 2)
        assert r['scores']['tr_score'] in (0, 1, 2, None)
        assert (r['scores']['tr_score'] is not None) == r['scores']['text_present']
    human = extract(canonical)
    save('human_reviews_main.json', human)
    frames = {'Main analysis': merge(human, inputs, judge, lookup)}
    sp = []
    for version, d in frames.items():
        key = 'main'
        d.to_csv(OUT / f'caption_level_{key}.csv', index=False)
        sp += spearman_results(d, version)
        print('Calculated Spearman:', version, flush=True)
    pd.DataFrame(sp).to_csv(OUT / 'spearman_results.csv', index=False)
    agr = agreement(human)
    save('human_agreement_results.json', {'source': 'human-rating workbook; main analysis', 'results': agr})
    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, list)} for r in agr]).to_csv(OUT / 'pairwise_kappa.csv', index=False)
    print('Calculated six reviewer-pair agreement results.', flush=True)
    cider = pd.read_csv(INPUT / 'cider_scores.csv')[['caption_id', 'file_name', 'model', 'cider']]
    assert len(cider) == 434 and cider.caption_id.is_unique and cider.file_name.nunique() == 217
    assert cider.groupby('file_name').size().eq(2).all() and np.isfinite(cider.cider).all()
    kr = []
    for version, d in frames.items():
        merged = cider.merge(d, on=['caption_id', 'file_name'], validate='one_to_one')
        assert len(merged) == 434
        for short, full in [('blip2', 'Salesforce/blip2-flan-t5-xl'), ('llava', 'llava-hf/llava-1.5-7b-hf')]:
            assert merged.loc[merged.model.eq(short), 'caption_model'].eq(full).all()
        merged.to_csv(OUT / 'cider_matched_main.csv', index=False)
        for model in ['pooled', 'blip2', 'llava']:
            part = merged if model == 'pooled' else merged.loc[merged.model.eq(model)]
            for metric in ['eh', 'tr']:
                eligible = part if metric == 'eh' else part.loc[part.tr_n.eq(4)]
                kr.append(dict(version=version, caption_model=model, metric=metric,
                    **correlation(eligible, 'cider', 'human_' + metric, 'kendall')))
        print('Calculated Kendall:', version, flush=True)
    pd.DataFrame(kr).to_csv(OUT / 'kendall_tau_results.csv', index=False)
    # Descriptive model scores use the same eligibility rule as correlations.
    desc = []
    for version, d in frames.items():
        for model, g in d.groupby('caption_model'):
            for group in ['all'] + sorted(g.stratum.unique()):
                part = g if group == 'all' else g.loc[g.stratum.eq(group)]
                for metric in ['eh', 'tr']:
                    eligible = part if metric == 'eh' else part.loc[part.tr_n.eq(4)]
                    desc.append(dict(version=version, model=model, image_group=group, metric=metric,
                                     captions=len(eligible), mean=float(eligible['human_' + metric].mean()) if len(eligible) else None))
    pd.DataFrame(desc).to_csv(OUT / 'human_model_summary.csv', index=False)
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(INPUT.iterdir()) if p.is_file()}
    hashes['BLV_Combined_Human_Ratings.xlsx'] = hashlib.sha256((ROOT / 'BLV_Combined_Human_Ratings.xlsx').read_bytes()).hexdigest()
    report = dict(source_workbook=WORKBOOK_URL, source_sha256=hashes['BLV_Combined_Human_Ratings.xlsx'],
                  input_sha256=hashes, bootstrap_iterations=ITERATIONS,
                  seeds={'spearman': 20261116, 'kendall_and_kappa': 20261117},
                  python_version=platform.python_version(), scipy_version=scipy.__version__, pandas_version=pd.__version__,
                  coverage={v: coverage(d) for v, d in frames.items()}, spearman=sp, kendall=kr,
                  methods=['Human mean over four reviewers; TR requires all four applicable, plus numeric judge for Spearman.',
                           'Spearman uses average ranks for ties; Kendall uses tau-b. 1000 image-cluster percentile bootstrap samples retain paired captions.',
                           'Six pairwise unweighted and linear-weighted Cohen kappas from the human-rating workbook. Mean pairwise kappa is not a four-rater kappa.',
                           'TR score kappa uses captions both reviewers mark applicable. Applicability kappa is separate. N/A is never zero.',
                           'CIDEr unchanged: 434 captions / 217 images. 33 images without valid references remain excluded.',
                           'No caption generation, judge API calls, CIDEr rescoring, or failure-taxonomy labelling performed.'])
    save('analysis_report.json', report)
    print(json.dumps({'coverage': report['coverage'],
        'pooled_spearman': [r for r in sp if r['caption_model'] == 'pooled' and r['exclusion'] == 'none'],
        'pooled_kendall': [r for r in kr if r['caption_model'] == 'pooled']}, indent=2))

if __name__ == '__main__':
    main()
