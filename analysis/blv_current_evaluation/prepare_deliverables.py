"""Validate main results independently and write a concise numerical summary.

Run after refresh_analysis.py. Reads only files inside this package.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'


def main():
    h = pd.DataFrame(json.loads((OUT / 'human_reviews_main.json').read_text()))
    d = pd.read_csv(OUT / 'caption_level_main.csv')
    c = pd.read_csv(OUT / 'cider_matched_main.csv')
    sp = pd.read_csv(OUT / 'spearman_results.csv', keep_default_na=False)
    kt = pd.read_csv(OUT / 'kendall_tau_results.csv')
    agr = json.loads((OUT / 'human_agreement_results.json').read_text())['results']
    assert len(h) == 2000 and len(d) == 500 and d.caption_id.is_unique
    assert not h.duplicated(['rater', 'caption_id']).any()
    assert h.groupby('caption_id').rater.nunique().eq(4).all()
    assert d.file_name.nunique() == 250 and d.groupby('file_name').size().eq(2).all()
    assert d.groupby('file_name').caption_model.nunique().eq(2).all()
    assert d.drop_duplicates('file_name').stratum.value_counts().to_dict() == {
        'clear': 70, 'blurry': 70, 'low_light': 60, 'text': 50}
    assert h.eh_score.isin([0, 1, 2]).all()
    assert h.tr_applicable.isin(['Yes', 'No']).all()
    assert h.loc[h.tr_applicable.eq('Yes'), 'tr_score'].isin([0, 1, 2]).all()
    assert h.loc[h.tr_applicable.eq('No'), 'tr_score'].eq('N/A').all()
    means = h.groupby('caption_id').eh_score.mean()
    assert np.allclose(d.set_index('caption_id').human_eh, means.reindex(d.caption_id))
    numeric = h.assign(tr=pd.to_numeric(h.tr_score, errors='coerce')).groupby('caption_id').tr.agg(['mean', 'count'])
    assert np.allclose(d.human_tr, numeric.reindex(d.caption_id)['mean'], equal_nan=True)
    assert (d.tr_n.to_numpy() == numeric.reindex(d.caption_id)['count'].to_numpy()).all()
    assert len(c) == 434 and c.file_name.nunique() == 217
    assert d.tr_n.eq(4).sum() == 102 and c.tr_n.eq(4).sum() == 100
    assert set(pd.read_csv(ROOT / 'inputs/cider_scores.csv').columns) == {'caption_id','file_name','model','cider'}
    checked = 0
    for _, r in sp.iterrows():
        p = d if r.caption_model == 'pooled' else d.loc[d.caption_model.eq(r.caption_model)]
        if r.exclusion == 'C0104':
            p = p.loc[p.caption_id.ne('C0104')]
        if r.metric == 'tr':
            p = p.loc[p.tr_n.eq(4) & p.judge_tr.notna()]
        v = spearmanr(p['human_' + r.metric], p['judge_' + r.metric]).statistic
        assert len(p) == r.caption_pairs and np.isclose(v, r.estimate, atol=1e-12, rtol=0)
        checked += 1
    for _, r in kt.iterrows():
        p = c if r.caption_model == 'pooled' else c.loc[c.model.eq(r.caption_model)]
        if r.metric == 'tr':
            p = p.loc[p.tr_n.eq(4)]
        v = kendalltau(p.cider, p['human_' + r.metric], variant='b').statistic
        assert len(p) == r.caption_pairs and np.isclose(v, r.estimate, atol=1e-12, rtol=0)
        checked += 1
    for r in agr:
        pair = h.loc[h.rater.eq(r['reviewer_1'])].merge(
            h.loc[h.rater.eq(r['reviewer_2'])], on='caption_id', suffixes=('_a','_b'), validate='one_to_one')
        metric = r['metric']
        if metric == 'TR_score_both_applicable':
            pair = pair.loc[pair.tr_applicable_a.eq('Yes') & pair.tr_applicable_b.eq('Yes')]
        if metric == 'EH_excluding_C0104':
            pair = pair.loc[pair.caption_id.ne('C0104')]
        if metric == 'TR_applicability':
            x, y = pair.tr_applicable_a.eq('Yes').astype(int), pair.tr_applicable_b.eq('Yes').astype(int)
            labels = [0, 1]
        else:
            field = 'tr_score' if metric.startswith('TR_score') else 'eh_score'
            x, y = pair[field + '_a'].astype(int), pair[field + '_b'].astype(int)
            labels = [0, 1, 2]
        assert len(pair) == r['caption_pairs']
        for weight, field in [(None, 'unweighted_kappa'), ('linear', 'linear_weighted_kappa')]:
            actual = cohen_kappa_score(x, y, labels=labels, weights=weight)
            assert np.isclose(actual, r[field], atol=1e-12, rtol=0)
            checked += 1
    summary = []
    for _, r in sp.loc[sp.caption_model.eq('pooled') & sp.exclusion.eq('none')].iterrows():
        summary.append(dict(statistic='Spearman rho', metric=r.metric.upper(), estimate=r.estimate,
            caption_pairs=r.caption_pairs, ci_95_lower=r.ci_95_lower, ci_95_upper=r.ci_95_upper))
    for _, r in kt.loc[kt.caption_model.eq('pooled')].iterrows():
        summary.append(dict(statistic='Kendall tau-b: CIDEr versus human score', metric=r.metric.upper(), estimate=r.estimate,
            caption_pairs=r.caption_pairs, ci_95_lower=r.ci_95_lower, ci_95_upper=r.ci_95_upper))
    for metric in ['EH','TR_score_both_applicable','TR_applicability']:
        for field in ['unweighted_kappa','linear_weighted_kappa']:
            rows = [r for r in agr if r['metric'] == metric]
            summary.append(dict(statistic='Mean of six pairwise ' + field, metric=metric,
                                estimate=float(np.mean([r[field] for r in rows]))))
    pd.DataFrame(summary).to_csv(OUT / 'summary.csv', index=False)
    validation = dict(passed=True, numerical_crosschecks=checked, human_reviews=len(h),
        captions=len(d), images=int(d.file_name.nunique()), all_four_TR_applicable=int(d.tr_n.eq(4).sum()),
        TR_applicability_disagreements=int(d.tr_n.between(1,3).sum()),
        cider_captions=len(c), cider_images=int(c.file_name.nunique()),
        checks=['Unique caption IDs and reviewer records; canonical mapping checked during extraction.',
                'All four reviewers per caption; valid EH/TR values; N/A preserved.',
                'Human means and group counts independently verified.',
                'Spearman and Kendall estimates and subset sizes recalculated.',
                'All 48 kappa estimates independently verified with scikit-learn.',
                'CIDEr input contains no human-rating columns.'])
    (OUT / 'validation.json').write_text(json.dumps(validation, indent=2))
    print('Validation passed:', checked, 'numerical cross-checks.')
    print(pd.DataFrame(summary).to_string(index=False))

if __name__ == '__main__':
    main()
