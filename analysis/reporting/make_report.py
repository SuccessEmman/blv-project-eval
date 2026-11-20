"""Create report tables and publication figures from the final BLV analysis.

No API calls or model runs. Inputs and existing analysis results are read-only.
Run from any directory: python make_report.py --project-root /path/to/blv-project-team
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import tempfile

# Keep the plotting cache outside the project and support computers without a display.
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'blv-matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import cohen_kappa_score

GROUPS = ['clear', 'blurry', 'low_light', 'text']
LABELS = {'clear': 'Clear', 'blurry': 'Blurry', 'low_light': 'Low-light', 'text': 'Text'}
RATERS = ['Ebere', 'Samuel', 'Success', 'Yvonne']
MODELS = ['BLIP-2', 'LLaVA-1.5']
COLORS = ['#176B9A', '#D97925']
SCORE_COLORS = ['#A63C40', '#E9B44C', '#24816A']


def model_name(value):
    value = str(value).lower()
    if value == 'pooled': return 'Pooled'
    if 'blip' in value: return MODELS[0]
    if 'llava' in value: return MODELS[1]
    raise ValueError(f'Unknown captioning model: {value}')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def require(ok, message):
    if not bool(ok): raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find_project(explicit=None):
    if explicit:
        candidates = [Path(explicit).expanduser().resolve()]
    else:
        candidates = list(Path(__file__).resolve().parents) + [Path.cwd()] + list(Path.cwd().parents)
    for p in candidates:
        if (p/'analysis/blv_current_evaluation/BLV_Combined_Human_Ratings.xlsx').is_file():
            return p
    raise FileNotFoundError('Set --project-root to your blv-project-team folder.')


class Report:
    def __init__(self, root, output, iterations=2000):
        self.root, self.output, self.iterations = root, output, iterations
        self.package = root/'analysis/blv_current_evaluation'
        self.results = self.package/'results'
        self.inputs = {}
        self.tables, self.figures = [], []
        for sub in ['tables', 'figures']:
            (output/sub).mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
            'axes.spines.top':False, 'axes.spines.right':False,
            'axes.titleweight':'bold', 'axes.titlesize':12, 'figure.dpi':110,
            'savefig.dpi':300, 'pdf.fonttype':42, 'svg.fonttype':'none'})

    def track(self, path):
        self.inputs[str(Path(path).relative_to(self.root))] = sha256(path)
        return path

    def csv(self, path, **kw):
        return pd.read_csv(self.track(path), **kw)

    def load(self):
        p = self.package
        report = read_json(self.track(self.results/'analysis_report.json'))
        for relative, expected in report['input_sha256'].items():
            path = p/relative
            require(sha256(self.track(path)) == expected,
                f'{relative} changed after analysis. Rerun refresh_analysis.py and prepare_deliverables.py first.')
        self.d = self.csv(self.results/'caption_level_main.csv')
        self.cider = self.csv(self.results/'cider_matched_main.csv')
        self.sp = self.csv(self.results/'spearman_results.csv', keep_default_na=False)
        self.kt = self.csv(self.results/'kendall_tau_results.csv')
        self.kappa = self.csv(self.results/'pairwise_kappa.csv')
        self.h = pd.DataFrame(read_json(self.track(self.results/'human_reviews_main.json')))
        self.h['tr_numeric'] = pd.to_numeric(self.h.tr_score, errors='coerce')
        self.d['Model'] = self.d.caption_model.map(model_name)
        self.cider['Model'] = self.cider.caption_model.map(model_name)
        self.h = self.h.merge(self.d[['caption_id','Model','stratum','tr_n']], on='caption_id', validate='many_to_one')
        # Check the human data against the authoritative workbook, not an earlier snapshot.
        scores = pd.read_excel(self.track(p/'BLV_Combined_Human_Ratings.xlsx'), sheet_name='Combined scores').set_index('caption_id')
        require(scores.index.is_unique and self.d.caption_id.is_unique, 'Duplicate caption IDs.')
        require(set(scores.index)==set(self.d.caption_id), 'Workbook and results caption IDs differ.')
        require(len(self.h)==2000 and len(self.d)==500, 'Expected 2,000 reviews and 500 captions.')
        require(not self.h.duplicated(['rater','caption_id']).any(), 'Duplicate human-review record.')
        require(self.h.groupby('caption_id').rater.nunique().eq(4).all(), 'Missing reviewer.')
        require(self.d.groupby('file_name').Model.nunique().eq(2).all(), 'Expected two models per image.')
        for rater in RATERS:
            h = self.h.loc[self.h.rater.eq(rater)].set_index('caption_id').reindex(scores.index)
            require(np.array_equal(h.eh_score, scores['EH '+rater]), f'{rater}: EH does not match workbook.')
            require(np.allclose(h.tr_numeric, pd.to_numeric(scores['TR '+rater], errors='coerce'), equal_nan=True), f'{rater}: TR mismatch.')
            require(h.tr_applicable.str.lower().eq(scores['TR applicable '+rater].str.lower()).all(), f'{rater}: applicability mismatch.')
            require(h.file_name.eq(scores.file_name).all(), f'{rater}: image mapping mismatch.')
        means = self.h.groupby('caption_id').agg(eh=('eh_score','mean'), tr=('tr_numeric','mean'), n=('tr_numeric','count'))
        means = means.reindex(self.d.caption_id)
        require(np.allclose(means.eh, self.d.human_eh), 'Human EH means are stale.')
        require(np.allclose(means.tr, self.d.human_tr, equal_nan=True), 'Human TR means are stale.')
        require(np.array_equal(means.n, self.d.tr_n), 'TR eligibility differs.')
        judge = pd.DataFrame([{'caption_id':x['caption_id'], 'eh':x['scores']['eh_score'],
            'tr':x['scores']['tr_score']} for x in [json.loads(line) for line in
                (p/'inputs/judge_results.jsonl').read_text().splitlines() if line.strip()]]).set_index('caption_id').reindex(self.d.caption_id)
        require(np.allclose(judge.eh, self.d.judge_eh), 'Judge EH mismatch.')
        require(np.allclose(judge.tr, self.d.judge_tr, equal_nan=True), 'Judge TR mismatch.')
        source_cider = pd.read_csv(p/'inputs/cider_scores.csv').set_index('caption_id')
        require(set(source_cider.index)==set(self.cider.caption_id), 'CIDEr ID mismatch.')
        require(np.allclose(source_cider.reindex(self.cider.caption_id).cider, self.cider.cider), 'CIDEr values mismatch.')
        # Correlations and all six reviewer pairs are independently checked before plotting.
        self.checks = 0
        for method, table in [('Spearman',self.sp),('Kendall',self.kt)]:
            for row in table.itertuples():
                data = self.d if method=='Spearman' else self.cider
                if row.caption_model!='pooled': data=data.loc[data.Model.eq(model_name(row.caption_model))]
                if getattr(row,'exclusion','none')=='C0104': data=data.loc[data.caption_id.ne('C0104')]
                if row.metric=='tr': data=data.loc[data.tr_n.eq(4)]
                if method=='Spearman':
                    data=data.loc[data['judge_'+row.metric].notna()]
                    val=spearmanr(data['human_'+row.metric],data['judge_'+row.metric]).statistic
                else: val=kendalltau(data.cider,data['human_'+row.metric],variant='b').statistic
                require(len(data)==row.caption_pairs and np.isclose(val,row.estimate,atol=1e-12), f'{method} result mismatch.')
                require(-1<=row.ci_95_lower<=row.ci_95_upper<=1, 'Invalid saved correlation interval.')
                self.checks+=1
        for row in self.kappa.itertuples():
            a=self.h.loc[self.h.rater.eq(row.reviewer_1)].set_index('caption_id')
            b=self.h.loc[self.h.rater.eq(row.reviewer_2)].set_index('caption_id').reindex(a.index)
            if row.metric=='TR_score_both_applicable':
                mask=a.tr_applicable.eq('Yes') & b.tr_applicable.eq('Yes');a=a.loc[mask];b=b.loc[mask]
            if row.metric=='EH_excluding_C0104': a=a.drop(index='C0104');b=b.drop(index='C0104')
            if row.metric=='TR_applicability':
                x=a.tr_applicable.eq('Yes').astype(int);y=b.tr_applicable.eq('Yes').astype(int);labels=[0,1]
            else:
                field='tr_numeric' if row.metric.startswith('TR_score') else 'eh_score'
                x=a[field].astype(int);y=b[field].astype(int);labels=[0,1,2]
            for weight,field in [(None,'unweighted_kappa'),('linear','linear_weighted_kappa')]:
                require(np.isclose(cohen_kappa_score(x,y,labels=labels,weights=weight),getattr(row,field),atol=1e-12), 'Kappa mismatch.')
                self.checks+=1
        self.final = self.csv(self.root/'sampling/outputs/final_sample/sample_manifest_with_blur.csv')
        bp = self.root/'sampling/outputs/blur_calibration'
        self.pilot = self.csv(bp/'pilot_with_blur_scores.csv')
        self.search = self.csv(bp/'threshold_search.csv')
        self.threshold_config = read_json(self.track(bp/'selected_threshold.json'))
        self.threshold = self.threshold_config['threshold']
        require(len(self.final)==250 and self.final.file_name.is_unique, 'Invalid final sample.')
        require(set(self.final.file_name)==set(self.d.file_name), 'Sampling and analysis image sets differ.')
        require(len(self.pilot)==200 and self.pilot.file_name.is_unique, 'Invalid pilot.')
        require(not set(self.pilot.file_name)&set(self.final.file_name), 'Pilot overlaps final sample.')
        best=self.search.loc[np.isclose(self.search.balanced_accuracy,self.search.balanced_accuracy.max())].sort_values('threshold')
        require(np.isclose(best.iloc[len(best)//2].threshold,self.threshold), 'Saved threshold differs from calibration selection.')
        for purpose in ['choose_threshold','check_threshold']:
            require(self.pilot.loc[self.pilot.purpose.eq(purpose)].reference_group.value_counts().to_dict()=={'blurry':50,'clear':50}, 'Pilot split mismatch.')
        self.pilot['prediction']=np.where(self.pilot.laplacian_variance<self.threshold,'blurry','clear')
        self.pilot['correct']=self.pilot.prediction.eq(self.pilot.reference_group)
        print('Validated final workbook, mappings, blur pilot, and',self.checks,'statistical estimates.',flush=True)

    def table(self, name, title, df, note):
        path=self.output/'tables'/name
        df.to_csv(path.with_suffix('.csv'),index=False,na_rep='N/A')
        # Tables are static report exports; the underlying script is the source of calculations.
        df.to_latex(path.with_suffix('.tex'), index=False, escape=True, na_rep='N/A', float_format=lambda x:f'{x:.4f}')
        self.tables.append({'name':name,'title':title,'data':df,'note':note})

    def figure(self, name, title, fig, note, section='Main'):
        for fmt in ['png','pdf','svg']:
            fig.savefig(self.output/'figures'/f'{name}.{fmt}',bbox_inches='tight',facecolor='white')
        plt.close(fig)
        self.figures.append({'name':name,'title':title,'note':note,'section':section})

    def mean_ci(self, data, value, seed):
        # Cluster by image. Both model captions remain together when pooled.
        clusters=data.groupby('file_name')[value].agg(['sum','count'])
        mean=float(data[value].mean()) if len(data) else np.nan
        if len(clusters)<2: return mean,np.nan,np.nan
        rng=np.random.default_rng(seed)
        ix=rng.integers(0,len(clusters),size=(self.iterations,len(clusters)))
        draws=clusters['sum'].to_numpy()[ix].sum(axis=1)/clusters['count'].to_numpy()[ix].sum(axis=1)
        low,high=np.quantile(draws,[.025,.975])
        return mean,float(low),float(high)

    def dataset_and_blur(self):
        f=self.final
        rows=[]
        for group in GROUPS:
            s=f.loc[f.stratum.eq(group)]
            rows.append({'Group':LABELS[group],'Images':len(s),'Train':int(s.split.eq('train').sum()),
                'Validation':int(s.split.eq('val').sum()),'Captions':2*len(s),'EH ratings':8*len(s)})
        self.table('01_dataset','Dataset composition',pd.DataFrame(rows),
            'Train/validation describe the source VizWiz splits. These images evaluate pretrained models; they did not train the captioning models.')
        bs=f.groupby('stratum').laplacian_variance.agg(['count','median','mean','min','max']).reindex(GROUPS).reset_index()
        bs.stratum=bs.stratum.map(LABELS)
        self.table('02_blur_scores','Laplacian-variance summaries',bs,
            'Descriptive measurements only. Original annotation-based image groups were retained.')
        performance=[]
        for purpose,title in [('choose_threshold','Calibration'),('check_threshold','Reserved evaluation')]:
            s=self.pilot.loc[self.pilot.purpose.eq(purpose)];recall=s.groupby('reference_group').correct.mean()
            performance.append({'Subset':title,'Images':len(s),'Threshold':self.threshold,
                'Blurry recall':recall['blurry'],'Clear recall':recall['clear'],'Balanced accuracy':recall.mean()})
        self.table('03_blur_performance','Blur heuristic performance',pd.DataFrame(performance),
            'Recall and balanced accuracy are proportions. Only calibration data selected the threshold; the reserved set evaluated that fixed rule.')
        check=self.pilot.loc[self.pilot.purpose.eq('check_threshold')]
        cm=pd.crosstab(check.reference_group,check.prediction).reindex(index=['blurry','clear'],columns=['blurry','clear'],fill_value=0)
        self.table('04_blur_confusion','Reserved-set blur confusion matrix',cm.rename_axis('Existing label').reset_index(),
            'Columns are predicted labels. Annotation labels are the reference; they are not assumed to be infallible ground truth.')
        fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
        vals=[f.loc[f.stratum.eq(g)].laplacian_variance.to_numpy() for g in GROUPS]
        require(all((v>0).all() for v in vals),'Nonpositive blur score cannot be shown on a log scale.')
        bp=axes[0].boxplot(vals,tick_labels=[LABELS[g] for g in GROUPS],patch_artist=True,showfliers=True)
        for patch,color in zip(bp['boxes'],['#80B5D5','#B6D5E7','#EBC48B','#A5D0C5']):patch.set_facecolor(color)
        axes[0].set(yscale='log',ylabel='Laplacian variance (log scale)',title='A. Final sample: 250 images')
        axes[0].tick_params(axis='x',rotation=20)
        s=self.search
        axes[1].plot(s.threshold,s.balanced_accuracy*100,color=COLORS[0])
        axes[1].axvline(self.threshold,color=COLORS[1],ls='--')
        axes[1].scatter([self.threshold],[s.balanced_accuracy.max()*100],color=COLORS[1],zorder=3)
        axes[1].set(xscale='symlog',xlabel='Candidate threshold (symlog scale)',ylabel='Calibration balanced accuracy (%)',ylim=(40,100),title='B. Calibration: 100 images')
        axes[1].text(.04,.95,f'Selected {self.threshold:.2f}\nBest calibration: {s.balanced_accuracy.max():.0%}',transform=axes[1].transAxes,va='top')
        axes[2].imshow(cm.to_numpy(),cmap='Blues',vmin=0,vmax=50)
        for i in range(2):
            for j in range(2):axes[2].text(j,i,str(cm.iloc[i,j]),ha='center',va='center',fontsize=20,color='white' if cm.iloc[i,j]>30 else '#172536')
        axes[2].set(xticks=[0,1],xticklabels=['Blurry','Clear'],yticks=[0,1],yticklabels=['Blurry','Clear'],xlabel='Predicted label',ylabel='Existing label',title='C. Reserved evaluation: 100 images')
        self.figure('02_blur_analysis','Blur scores, calibration and reserved evaluation',fig,
            'Whiskers extend to 1.5 IQR; dots are outliers. The score axis is logarithmic and the threshold axis is symmetric-logarithmic. The fixed threshold achieved 77% calibration and 71% reserved-set balanced accuracy. It was not validated for low-light or text classification.')

    def coverage(self):
        d,h=self.d,self.h
        all_yes=int(d.tr_n.eq(4).sum());mixed=int(d.tr_n.between(1,3).sum());all_no=int(d.tr_n.eq(0).sum())
        rows=[('Final images',d.file_name.nunique(),'Images'),('Generated captions',len(d),'Captions'),
            ('Human EH ratings',len(h),'Ratings'),('All numerical human TR ratings',h.tr_numeric.notna().sum(),'Ratings'),
            ('Common TR subset',all_yes,'Captions'),('Common TR images',d.loc[d.tr_n.eq(4)].file_name.nunique(),'Images'),
            ('TR ratings in common subset',4*all_yes,'Ratings'),('Human applicability disagreements',mixed,'Captions'),
            ('All humans marked TR N/A',all_no,'Captions'),('CIDEr available',len(self.cider),'Captions'),
            ('CIDEr images',self.cider.file_name.nunique(),'Images'),('Images without valid CIDEr references',d.file_name.nunique()-self.cider.file_name.nunique(),'Images'),
            ('CIDEr and common TR overlap',self.cider.tr_n.eq(4).sum(),'Captions')]
        self.table('05_coverage','Review and analysis coverage',pd.DataFrame(rows,columns=['Measure','Count','Unit']),
            'N/A is not zero. Applicability disagreements remain in the source workbook but are excluded from the common numerical TR subset. EH and TR subsets overlap; do not add them.')
        fig,ax=plt.subplots(figsize=(11,7));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
        def box(x,y,w,ht,text,color='#E9F2F8'):
            ax.add_patch(FancyBboxPatch((x,y),w,ht,boxstyle='round,pad=0.012',facecolor=color,edgecolor='#7298AF'))
            ax.text(x+w/2,y+ht/2,text,ha='center',va='center',fontsize=11)
        def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#527086','lw':1.5})
        box(.05,.81,.54,.14,'Annotation-based selection + manual text review\n250 final images: 70 clear / 70 blurry / 60 low-light / 50 text')
        box(.68,.81,.27,.14,'Separate blur pilot\n100 calibration + 100 reserved\nNo overlap with final sample','#F9EEDB')
        box(.22,.59,.56,.13,'BLIP-2 + LLaVA: 500 generated captions\nFour human reviewers + one Gemini judge')
        arrow((.32,.80),(.50,.73))
        box(.02,.32,.28,.16,f'Error Honesty\n{len(d)} captions / {d.file_name.nunique()} images\n{len(h):,} human ratings')
        box(.36,.32,.28,.16,f'Common Text Recognition\n{all_yes} captions / 51 images\n{all_yes*4} human ratings')
        box(.70,.32,.28,.16,f'CIDEr comparison\n{len(self.cider)} captions / 217 images\n33 images lack valid references')
        for x in [.16,.50,.84]:arrow((.50,.58),(x,.49))
        box(.32,.07,.36,.13,f'TR eligibility exclusions\n{all_no} captions: all humans N/A\n{mixed} captions: applicability disagreement','#F3F3F3')
        arrow((.50,.31),(.50,.21))
        fig.suptitle('Study workflow and evaluation subsets',fontsize=16,fontweight='bold')
        self.figure('01_workflow','Study workflow and evaluation subsets',fig,
            'The three analysis branches overlap. TR uses the common all-four-applicable subset, while CIDEr excludes images without valid reference captions. The separate blur pilot contains 200 different images.')
        app=pd.DataFrame({'Human applicability':np.select([d.tr_n.eq(4),d.tr_n.eq(0)],['All four: applicable','All four: N/A'],default='Reviewers disagree'),
            'Gemini applicability':np.where(d.judge_tr.notna(),'Applicable','N/A')})
        cm=pd.crosstab(app['Human applicability'],app['Gemini applicability']).reindex(index=['All four: applicable','Reviewers disagree','All four: N/A'],columns=['Applicable','N/A'],fill_value=0)
        self.table('06_applicability','Human and judge TR applicability',cm.reset_index(),
            'Counts are captions. Disagreement alone does not establish whether the humans or Gemini were correct.')
        fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained');ax.imshow(cm.to_numpy(),cmap='Blues')
        for i in range(3):
            for j in range(2):ax.text(j,i,str(cm.iloc[i,j]),ha='center',va='center',fontsize=18,color='white' if cm.iloc[i,j]>cm.to_numpy().max()/2 else '#172536')
        ax.set(xticks=[0,1],xticklabels=cm.columns,yticks=[0,1,2],yticklabels=cm.index,xlabel='Gemini text applicability',title='TR applicability: 500 captions')
        self.figure('08_tr_applicability','Human and judge text applicability',fig,
            'All-four agreement determines the common human TR subset. This is a comparison of applicability judgments, not a correctness confusion matrix. N/A does not enter score correlations as zero.','Appendix')

    def model_scores(self):
        rows=[]
        for metric in ['eh','tr']:
            eligible=self.d if metric=='eh' else self.d.loc[self.d.tr_n.eq(4)]
            for model in MODELS:
                for group in ['all']+GROUPS:
                    data=eligible.loc[eligible.Model.eq(model)]
                    if group!='all':data=data.loc[data.stratum.eq(group)]
                    mean,lo,hi=self.mean_ci(data,'human_'+metric,20261119+len(rows))
                    rows.append({'Metric':metric.upper(),'Model':model,'Group':'Overall' if group=='all' else LABELS[group],
                        'Captions':len(data),'Images':data.file_name.nunique(),'Mean':mean,'CI lower':lo,'CI upper':hi})
        stats=pd.DataFrame(rows)
        self.table('07_human_model_scores','Human scores by model and image condition',stats,
            f'Mean of per-caption four-reviewer means, on the 0–2 scale. Descriptive 95% percentile intervals use {self.iterations} image-bootstrap replicates (seeds 20261119 + row index). No CI for fewer than two images; N/A means no eligible observations. These new mean-score intervals are distinct from the saved correlation intervals.')
        fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        for ax,metric,groups in [(axes[0],'EH',[LABELS[g] for g in GROUPS]),(axes[1],'TR',['Overall','Text','Blurry'])]:
            for mi,model in enumerate(MODELS):
                sub=stats.loc[stats.Metric.eq(metric)&stats.Model.eq(model)].set_index('Group').reindex(groups)
                x=np.arange(len(groups))+(mi-.5)*.18
                ax.plot(x,sub.Mean,'o',color=COLORS[mi],label=model,ms=7)
                for k,(_,row) in enumerate(sub.iterrows()):
                    if pd.notna(row['CI lower']) and pd.notna(row['CI upper']):
                        ax.vlines(x[k],row['CI lower'],row['CI upper'],color=COLORS[mi],lw=2)
            ax.set(xticks=np.arange(len(groups)),xticklabels=[f'{g}\nn={int(stats.loc[stats.Metric.eq(metric)&stats.Model.eq(MODELS[0])&stats.Group.eq(g),"Images"].iloc[0])}/model' for g in groups],
                ylim=(-.08,2.08),ylabel='Mean human score (0–2)',title=metric+' by image condition')
            ax.grid(axis='y',alpha=.2);ax.legend(loc='upper right')
        self.figure('03_human_scores','Human scores across image conditions',fig,
            'Dots are means of four-reviewer caption means; whiskers are image-bootstrap 95% intervals. TR contains 50 text-group images and one blurry image per model. There are no common-applicable TR observations in the clear or low-light groups. No confidence interval is shown for the single blurry image; its value is descriptive only.')
        dist=[]
        fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
        for ax,metric,field in [(axes[0],'EH','eh_score'),(axes[1],'TR','tr_numeric')]:
            h=self.h if metric=='EH' else self.h.loc[self.h.tr_n.eq(4)]
            bottoms=np.zeros(2)
            for score in [0,1,2]:
                perc=[]
                for model in MODELS:
                    sub=h.loc[h.Model.eq(model)];n=int(sub[field].eq(score).sum());pct=100*n/len(sub)
                    dist.append({'Metric':metric,'Model':model,'Score':score,'Ratings':n,'Eligible ratings':len(sub),'Percent':pct});perc.append(pct)
                ax.bar(MODELS,perc,bottom=bottoms,color=SCORE_COLORS[score],label=f'Score {score}')
                for i,v in enumerate(perc):
                    if v>=7:ax.text(i,bottoms[i]+v/2,f'{v:.1f}%',ha='center',va='center',color='white' if score!=1 else '#172536')
                bottoms+=perc
            ax.set(ylim=(0,100),ylabel='Individual human ratings (%)',title=f'{metric}: '+('1,000 ratings/model' if metric=='EH' else '204 ratings/model'))
        axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=3)
        self.table('08_score_distribution','Human rating distributions',pd.DataFrame(dist),'Individual ratings are descriptive; ratings for the same caption/image are not independent observations. TR uses only the common subset.')
        self.figure('06_score_distributions','Human rating distributions',fig,
            'Each bar pools individual ratings for one model: 1,000 EH ratings or 204 common-subset TR ratings. N/A is excluded from TR, never zero. These are individual-rating proportions, not rounded caption-mean scores.','Appendix')

    def agreement(self):
        k=self.kappa
        self.table('09_reviewer_agreement','Pairwise reviewer agreement',k.loc[k.metric.isin(['EH','TR_score_both_applicable','TR_applicability'])].copy(),
            'Six pairs, not a single four-rater Cohen kappa. TR score agreement requires both members of each pair to mark the caption applicable. Exact agreement is a proportion and differs from chance-corrected kappa.')
        rows=[]
        for metric in ['EH','TR_score_both_applicable','TR_applicability']:
            s=k.loc[k.metric.eq(metric)]
            rows.append({'Metric':metric,'Pairs':len(s),'Mean unweighted kappa':s.unweighted_kappa.mean(),
                'Mean linear-weighted kappa':s.linear_weighted_kappa.mean(),'Mean exact agreement':s.exact_agreement.mean()})
        self.table('10_team_agreement','Mean of six reviewer-pair agreement values',pd.DataFrame(rows),
            'Arithmetic means of the six pairwise statistics. Final human-score snapshot; not a reconstruction of first-pass independent agreement.')
        for weight,label,num,section in [('unweighted_kappa','Unweighted', '04','Main'),('linear_weighted_kappa','Linear-weighted','05','Appendix')]:
            fig,axes=plt.subplots(1,2,figsize=(10,4.6),layout='constrained')
            for ax,metric,title in [(axes[0],'EH','Error Honesty'),(axes[1],'TR_score_both_applicable','Text Recognition')]:
                values=np.full((4,4),np.nan);s=k.loc[k.metric.eq(metric)]
                for row in s.itertuples():
                    i,j=RATERS.index(row.reviewer_1),RATERS.index(row.reviewer_2)
                    values[i,j]=values[j,i]=getattr(row,weight)
                ax.imshow(np.ma.masked_invalid(values),cmap='Blues',vmin=0,vmax=1)
                for i in range(4):
                    for j in range(4):
                        value=values[i,j];ax.text(j,i,'—' if np.isnan(value) else f'{value:.3f}',ha='center',va='center',color='white' if value>.6 else '#172536')
                ax.set(xticks=range(4),xticklabels=RATERS,yticks=range(4),yticklabels=RATERS,
                    title=f'{title}\nMean of six pairs: {s[weight].mean():.3f}')
                ax.tick_params(axis='x',rotation=25)
            fig.suptitle(label+" Cohen's κ",fontsize=15,fontweight='bold')
            self.figure(num+'_reviewer_agreement_'+label.lower().replace('-','_'),label+' reviewer agreement',fig,
                label+' Cohen kappa for each reviewer pair. Mirrored cells repeat the same six pairs; diagonal cells are omitted. Both panels use a fixed 0–1 colour scale. TR is pairwise-applicable score agreement. The underlying table preserves full precision.',section)

    def correlations(self):
        rows=[]
        for name,t in [('Spearman rho',self.sp),('Kendall tau-b',self.kt)]:
            for row in t.itertuples():
                rows.append({'Statistic':name,'Model':model_name(row.caption_model),'Metric':row.metric.upper(),
                    'Exclusion':getattr(row,'exclusion','none'),'Caption pairs':row.caption_pairs,'Images':row.unique_images,
                    'Estimate':row.estimate,'CI lower':row.ci_95_lower,'CI upper':row.ci_95_upper})
        table=pd.DataFrame(rows)
        self.table('11_correlations','Human-score correlation results',table,
            'Saved image-cluster bootstrap 95% intervals (1,000 replicates). Spearman compares judge with human means; Kendall tau-b compares CIDEr with human means. They measure ranking association, not accuracy or exact agreement.')
        fig,axes=plt.subplots(1,2,figsize=(13,5.6),layout='constrained')
        for ax,name,title in [(axes[0],'Spearman rho','Gemini versus human'),(axes[1],'Kendall tau-b','CIDEr versus human')]:
            sub=table.loc[table.Statistic.eq(name)&table.Exclusion.eq('none')].copy()
            sub['sort']=sub.Model.map({'Pooled':0,MODELS[0]:1,MODELS[1]:2})*2+sub.Metric.map({'EH':0,'TR':1})
            sub=sub.sort_values('sort').reset_index(drop=True)
            for i,row in sub.iterrows():
                color=COLORS[0] if row.Metric=='EH' else COLORS[1]
                ax.hlines(i,row['CI lower'],row['CI upper'],color=color,lw=2)
                ax.plot(row.Estimate,i,'o',color=color)
            ax.axvline(0,color='#888',ls=':',lw=1)
            ax.set(yticks=range(len(sub)),yticklabels=[f'{r.Model} · {r.Metric} (n={int(r["Caption pairs"])})' for _,r in sub.iterrows()],xlim=(-1,1),xlabel=name+' with 95% CI',title=title)
            ax.invert_yaxis();ax.grid(axis='x',alpha=.18)
        self.figure('07_correlations','Spearman and Kendall correlations',fig,
            'Both panels show ranking associations on −1 to 1 scales, with saved image-cluster 95% intervals. Each n is the number of paired caption scores. Model-specific values are descriptive; differences between correlations are not a formal significance test.')
        sensitivity=table.loc[table.Statistic.eq('Spearman rho')&table.Model.eq('Pooled')&table.Metric.eq('EH')]
        self.table('12_eh_sensitivity','Generation-flag sensitivity analysis',sensitivity,
            'Primary analysis includes C0104. Sensitivity excludes that generation-flagged caption. This does not include the separate text-review sensitivity analysis, which is outside the main package.')
        fig,ax=plt.subplots(figsize=(8,2.8),layout='constrained')
        for i,(_,row) in enumerate(sensitivity.iterrows()):
            ax.hlines(i,row['CI lower'],row['CI upper'],color=COLORS[i],lw=2);ax.plot(row.Estimate,i,'o',color=COLORS[i])
        ax.set(yticks=range(len(sensitivity)),yticklabels=['All captions (n=500)','Exclude C0104 (n=499)'],xlim=(0,1),xlabel='EH Spearman rho with 95% CI',title='Sensitivity to one generation-flagged caption');ax.invert_yaxis()
        self.figure('09_eh_sensitivity','Generation-flag sensitivity',fig,
            'The primary and sensitivity estimates use the same fixed judge responses and final human scores. Overlapping intervals are not a formal test of the difference.','Appendix')
        fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
        bubble=[]
        for ax,metric in zip(axes,['eh','tr']):
            d=self.d if metric=='eh' else self.d.loc[self.d.tr_n.eq(4)&self.d.judge_tr.notna()]
            counts=d.groupby(['human_'+metric,'judge_'+metric]).size().reset_index(name='Captions')
            ax.scatter(counts.iloc[:,0],counts.iloc[:,1],s=counts.Captions*13,alpha=.5,color=COLORS[0],edgecolor='#17405B')
            for row in counts.itertuples(index=False):
                if row[2]>=5:ax.text(row[0],row[1],str(row[2]),ha='center',va='center',fontsize=8)
            ax.plot([0,2],[0,2],ls=':',color='#777',label='Equal score')
            ax.set(xlim=(-.18,2.18),ylim=(-.35,2.35),xticks=[0,.5,1,1.5,2],yticks=[0,1,2],xlabel='Mean of four human scores',ylabel='Gemini score',title=f'{metric.upper()}: {len(d)} captions')
            for _,row in counts.iterrows():bubble.append({'Metric':metric.upper(),'Human mean':row.iloc[0],'Judge score':row.iloc[1],'Captions':row.Captions})
        self.table('13_score_pairs','Human/judge score-pair frequencies',pd.DataFrame(bubble),'Each caption contributes once. TR uses common applicability. Bubble area is proportional to caption count; numeric labels show counts of five or more.')
        self.figure('10_judge_score_pairs','Human and judge score pairs',fig,
            'Each circle represents a score combination; area is proportional to the number of captions. Labels show counts of at least five. The diagonal indicates equal numeric scores and is not a fitted regression line.','Appendix')

    def cider_and_taxonomy(self):
        rows=[]
        fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
        for ax,metric in zip(axes,['eh','tr']):
            eligible=self.cider if metric=='eh' else self.cider.loc[self.cider.tr_n.eq(4)]
            for mi,model in enumerate(MODELS):
                sub=eligible.loc[eligible.Model.eq(model)]
                rows.append({'Metric':metric.upper(),'Model':model,'Matched captions':len(sub),
                    'Images':sub.file_name.nunique(),'Mean CIDEr (raw scale)':sub.cider.mean(),'Mean human score (0–2)':sub['human_'+metric].mean()})
                ax.scatter(sub.cider,sub['human_'+metric],s=19,alpha=.45,color=COLORS[mi],label=model)
            ax.set(xlabel='CIDEr (raw score)',ylabel='Mean human score (0–2)',ylim=(-.1,2.1),title=f'{metric.upper()}: {len(eligible)} matched captions')
            ax.legend(fontsize=9);ax.grid(alpha=.15)
        self.table('14_cider_model_comparison','Models compared on matched CIDEr subsets',pd.DataFrame(rows),
            'Both metrics in each row use exactly the same captions. EH uses 217 images/model; TR uses 50. CIDEr remains on its raw scale, not multiplied by 100. These are descriptive means, not a significance test or a combined score.')
        self.figure('11_cider_human_scores','CIDEr against human scores',fig,
            'Only captions with valid reference-based CIDEr scores are shown. TR additionally requires all four humans to consider text applicable. Scores keep their original separate axes; no composite score or fitted line is imposed.','Appendix')
        path=self.root/'analysis/failure_taxonomy_exploratory/taxonomy_captions.csv'
        if not path.exists():
            self.taxonomy_note='Exploratory taxonomy omitted: source file is absent.';return
        tax=self.csv(path,keep_default_na=False).rename(columns={'Unnamed: 0':'caption_id'})
        require('caption_id' in tax and tax.caption_id.is_unique,'Taxonomy caption ID column missing/duplicated.')
        tax=tax.set_index('caption_id');require(set(tax.index)==set(self.d.caption_id),'Taxonomy IDs differ.')
        for rater in RATERS:
            h=self.h.loc[self.h.rater.eq(rater)].set_index('caption_id').reindex(tax.index)
            require(np.allclose(pd.to_numeric(tax['eh_'+rater]),h.eh_score),'Taxonomy EH scores differ from final human scores.')
        d=self.d.set_index('caption_id').reindex(tax.index)
        require(np.allclose(pd.to_numeric(tax.tr_mean,errors='coerce'),d.human_tr,equal_nan=True),'Taxonomy TR means differ.')
        require(tax.file_name.eq(d.file_name).all(),'Taxonomy image mismatch.')
        counts=tax.loc[tax.primary_label.ne('')].primary_label.value_counts().rename_axis('Recorded rule label').reset_index(name='Captions')
        counts['Percent of all 500 captions']=counts.Captions/len(tax)*100
        self.table('15_exploratory_taxonomy','Exploratory automated failure categories',counts,
            'Recorded labels are keyword-rule outputs, not validated human coding. The rule labelled vague/over-hedged includes all majority-EH1 cases and can include colour-only errors. Counts are captions, not images. Judge reasons influence some categories.')
        display=counts['Recorded rule label'].replace({'Vague or over-hedged (3+ of 4 gave 1)':'Majority EH1 (rule label; mixed causes)'})
        fig,ax=plt.subplots(figsize=(11,5.8),layout='constrained')
        ax.barh(np.arange(len(counts)),counts.Captions,color=COLORS[0])
        ax.set(yticks=range(len(counts)),yticklabels=display,xlabel='Captions',title='Exploratory automated failure categories')
        for i,value in enumerate(counts.Captions):ax.text(value+.5,i,str(value),va='center')
        ax.set_xlim(0,counts.Captions.max()*1.15);ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
        self.figure('12_exploratory_taxonomy','Exploratory automated failure categories',fig,
            'Provisional keyword-based categories; not fully validated human coding and not independent validation of Gemini. One primary label per included caption. The majority-EH1 label is displayed without asserting that all such cases are over-hedging. Unlabelled captions are not shown.','Appendix')
        self.taxonomy_note='Exploratory taxonomy included; scores and IDs match the final human-score analysis.'

    def finish(self):
        for relative, expected in self.inputs.items():
            require(sha256(self.root/relative)==expected,
                f'Input changed during report generation: {relative}. Rerun when editing is finished.')
        self.figures.sort(key=lambda item:item['name'])
        self.tables.sort(key=lambda item:item['name'])
        style='''body{font:16px/1.55 system-ui,sans-serif;color:#172536;background:#f6f8fb;margin:0}main{max-width:1150px;margin:auto;padding:36px}h1{font-size:32px}h2{font-size:23px}section{background:white;padding:24px;margin:24px 0;border-radius:12px;border:1px solid #dbe3ec}img{width:100%;height:auto}.note{color:#46566b;font-size:14px}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px;white-space:nowrap}td,th{padding:8px 12px;border-bottom:1px solid #dbe3ec;text-align:right}th{background:#eaf2f8}td:first-child,th:first-child{text-align:left}a{color:#126694}code{background:#edf1f5;padding:2px 4px}@media print{body{background:white}main{padding:0}section{break-inside:avoid;border:0}img{max-height:650px;object-fit:contain}}'''
        parts=['<!doctype html><html lang="en"><meta charset="utf-8"><title>BLV report tables and figures</title><style>'+style+'</style><main>',
            '<h1>BLV report tables and figures</h1><p>Human scores from the final combined workbook. Existing model and judge responses are reused; no new model requests.</p>',
            '<p>PNG files are for Word and slides; vector PDF files are for Overleaf; SVG files are editable graphics. Tables are available as CSV and LaTeX (requires <code>booktabs</code>). Full-precision numeric values are retained in CSV.</p>']
        captions=['# Figure captions and table notes\n','Human scores refer to the final combined workbook.\n']
        for f in self.figures:
            name=f['name'];title=html.escape(f['title']);note=html.escape(f['note'])
            parts.append(f'<section><p class="note">{f["section"]}</p><h2>{title}</h2><img src="figures/{name}.png" alt="{title}"><p class="note">{note}</p><p><a href="figures/{name}.png">PNG</a> · <a href="figures/{name}.pdf">PDF</a> · <a href="figures/{name}.svg">SVG</a></p></section>')
            captions.extend([f'## {name}: {f["title"]}\n',f['note']+'\n'])
        for t in self.tables:
            name=t['name'];parts.append(f'<section><h2>{html.escape(t["title"])}</h2><div class="scroll">'+t['data'].to_html(index=False,border=0,na_rep='N/A',float_format=lambda v:f'{v:.4f}')+f'</div><p class="note">{html.escape(t["note"])}</p><p><a href="tables/{name}.csv">CSV</a> · <a href="tables/{name}.tex">LaTeX</a></p></section>')
            captions.extend([f'## Table {name}: {t["title"]}\n',t['note']+'\n'])
        parts.append('</main></html>')
        (self.output/'index.html').write_text('\n'.join(parts),encoding='utf8')
        (self.output/'CAPTIONS.md').write_text('\n'.join(captions),encoding='utf8')
        manifest={'source_workbook':'analysis/blv_current_evaluation/BLV_Combined_Human_Ratings.xlsx',
            'generator_sha256':sha256(Path(__file__)),
            'input_sha256':self.inputs,'statistical_crosschecks':self.checks,
            'mean_score_bootstrap_replicates':self.iterations,'mean_score_seed_start':20261119,
            'existing_correlation_intervals':'Reused from final analysis (1,000 image-cluster replicates).',
            'figures':self.figures,'tables':[{'name':t['name'],'title':t['title'],'note':t['note'],'rows':len(t['data'])} for t in self.tables],
            'taxonomy':self.taxonomy_note,'separate_text_review_sensitivity':'Not included: not part of the main human-score package.',
            'no_api_calls':True,'inputs_modified':False}
        (self.output/'report_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
        print(f'Saved {len(self.tables)} tables and {len(self.figures)} figures in {self.output}',flush=True)
        print('Open index.html to browse the complete report pack.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path)
    parser.add_argument('--output',type=Path,help='Default: analysis/reporting/report_outputs')
    parser.add_argument('--mean-bootstrap',type=int,default=2000)
    args=parser.parse_args()
    require(args.mean_bootstrap>=1000,'Use at least 1,000 bootstrap replicates for reported mean intervals.')
    root=find_project(args.project_root)
    output=(args.output or root/'analysis/reporting/report_outputs').expanduser().resolve()
    # Keep report output away from the frozen analysis inputs/results.
    protected=root/'analysis/blv_current_evaluation'
    require(output!=protected and protected not in output.parents,'Choose an output folder outside the final analysis package.')
    report=Report(root,output,args.mean_bootstrap)
    report.load();report.dataset_and_blur();report.coverage();report.model_scores()
    report.agreement();report.correlations();report.cider_and_taxonomy();report.finish()


if __name__=='__main__':main()
