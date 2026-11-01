"""Recreate the report's four figures from verified saved tables and training data."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter,MaxNLocator
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
TAB=ROOT/'reports/tables'; OUT=ROOT/'reports/figures/apa';OUT.mkdir(exist_ok=True)
BLUE='#1F567D'; ORANGE='#B65D20'; GREY='#697680'; INK='#20272B'
plt.rcParams.update({'font.family':'Arial','font.size':10,'axes.labelsize':10,
 'xtick.labelsize':9,'ytick.labelsize':10,'legend.fontsize':9,'axes.edgecolor':'#68747B',
 'axes.spines.top':False,'axes.spines.right':False,'text.color':INK,'axes.labelcolor':INK,
 'xtick.color':INK,'ytick.color':INK,'savefig.facecolor':'white'})
def save(fig,name):
 fig.savefig(OUT/name,dpi=350,facecolor='white')
 plt.close(fig)

missing=pd.read_csv(TAB/'01_missingness_by_variable.csv').head(8).iloc[::-1]
labels={'rbc':'Red blood cell appearance','rbcc':'Red blood cell count','wbcc':'White blood cell count',
 'pot':'Potassium','sod':'Sodium','pcv':'Packed cell volume','pc':'Pus cell appearance','hemo':'Hemoglobin'}
fig,ax=plt.subplots(figsize=(6.5,3.15));fig.subplots_adjust(left=.32,right=.985,bottom=.19,top=.98)
y=np.arange(len(missing))
ax.barh(y,missing.missing_pct,color=BLUE,height=.67)
ax.set_yticks(y,[labels[x] for x in missing.variable]);ax.tick_params(axis='y',length=0)
ax.set_xlim(0,49);ax.set_xticks([0,10,20,30,40]);ax.xaxis.set_major_formatter(PercentFormatter(100,decimals=0))
ax.set_xlabel('Records with a missing value (N = 400)');ax.set_axisbelow(True);ax.grid(axis='x',color='#E4E8EA',linewidth=.7)
for i,row in enumerate(missing.itertuples()):ax.text(row.missing_pct+.7,i,f'{row.missing_n} ({row.missing_pct:.1f}%)',va='center',fontsize=9)
save(fig,'figure1_missingness.png')

findings=json.loads((TAB/'01_findings.json').read_text())
counts=[findings['class_counts']['ckd'],findings['class_counts']['notckd']]
assert sum(counts)==400
fig,ax=plt.subplots(figsize=(6.5,2.25));fig.subplots_adjust(left=.14,right=.97,bottom=.28,top=.95)
ax.barh(['notckd','ckd'],counts[::-1],color=[GREY,BLUE],height=.55)
ax.set_xlim(0,310);ax.set_xlabel('Clinical records (N = 400)');ax.tick_params(axis='y',length=0)
for y,value in enumerate(counts[::-1]):ax.text(value+5,y,f'{value} ({value/4:.1f}%)',va='center')
ax.xaxis.set_major_locator(MaxNLocator(integer=True));ax.set_axisbelow(True);ax.grid(axis='x',color='#E4E8EA',linewidth=.7)
save(fig,'figure2_class_balance.png')

cv=pd.read_csv(TAB/'03_model_comparison_cv.csv')
fig,ax=plt.subplots(figsize=(6.5,2.45));fig.subplots_adjust(left=.29,right=.985,bottom=.24,top=.96)
names=['Majority baseline','Logistic regression','Decision tree','Random forest']
for i,row in enumerate(cv.itertuples()):
 color=GREY if i==0 else BLUE
 ax.errorbar(row.macro_f1_mean,3-i,xerr=row.macro_f1_sd,fmt='o',markersize=7,capsize=4,color=color,elinewidth=1.3)
 ax.text(.66,3-i,f'{row.macro_f1_mean:.3f}',va='center',fontsize=10,color=color,
         bbox={'facecolor':'white','edgecolor':'none','pad':1})
ax.set_yticks([3,2,1,0],names);ax.set_ylim(-.55,3.55);ax.set_xlim(0,1.04)
ax.set_xticks(np.arange(0,1.01,.2));ax.set_xlabel('Macro F1 (mean across 15 validation folds)')
ax.set_axisbelow(True);ax.grid(axis='x',color='#E4E8EA',linewidth=.7);ax.tick_params(axis='y',length=0)
save(fig,'figure3_model_comparison.png')

result=json.loads((TAB/'03_results.json').read_text())['test']
pred=pd.read_csv(TAB/'03_test_predictions.csv')
cm=np.array([[(int(((pred.true_ckd==a)&(pred.predicted_ckd==b)).sum())) for b in [0,1]] for a in [0,1]])
assert cm.tolist()==[[result['tn'],result['fp']],[result['fn'],result['tp']]]
fig,ax=plt.subplots(figsize=(3.5,2.45));fig.subplots_adjust(left=.3,right=.98,bottom=.25,top=.96)
ax.imshow(cm,cmap='Blues',vmin=0,vmax=50,aspect='auto')
ax.set_xticks([0,1],['Not CKD','CKD']);ax.set_yticks([0,1],['Not CKD','CKD'])
ax.set_xlabel('Predicted label');ax.set_ylabel('Supplied label');ax.tick_params(length=0)
for (row,col),value in np.ndenumerate(cm):ax.text(col,row,str(value),ha='center',va='center',fontsize=17,
 color='white' if value>=25 else INK,fontweight='bold')
for spine in ax.spines.values():spine.set_visible(False)
save(fig,'figure4_confusion_matrix.png')
manifest={'source_tables':['01_missingness_by_variable.csv','03_model_comparison_cv.csv','03_results.json','03_test_predictions.csv'],
 'sample_sizes':{'all':400,'training':320,'test':80},'test_matrix':cm.tolist(),
 'displayed_missingness_rows':missing.variable.iloc[::-1].tolist(),
 'class_counts':findings['class_counts'],'cv_whiskers':'Mean +/- 1 descriptive fold SD, not confidence intervals',
 'figure_notes':'Standalone figure images omit number/title so APA numbering and titles are provided by the document.'}
(OUT/'figure_manifest.json').write_text(json.dumps(manifest,indent=2))
print('Four APA report figures rebuilt and tied to verified source tables.')
