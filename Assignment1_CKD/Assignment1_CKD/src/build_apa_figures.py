from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
ROOT=Path(__file__).resolve().parents[1]
TAB=ROOT/'reports/tables';OUT=ROOT/'reports/figures/apa';OUT.mkdir(parents=True,exist_ok=True)
BLUE='#285B78';INK='#172D3A';GREY='#E7ECEF';MUTED='#52636E'
from matplotlib import font_manager
REPORT_FONT = 'Arial' if any(f.name == 'Arial' for f in font_manager.fontManager.ttflist) else 'DejaVu Sans'
plt.rcParams.update({'font.family':REPORT_FONT,'font.size':11,'axes.labelsize':11,'xtick.labelsize':10.5,'ytick.labelsize':11,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK})
def save(fig,n):fig.savefig(OUT/['figure1_missingness.png','figure2_class_balance.png','figure3_confusion_matrix.png'][n-1],dpi=400,facecolor='white');plt.close(fig)
m=pd.read_csv(TAB/'01_missingness_by_variable.csv').head(5).iloc[::-1]
clean=pd.read_csv(ROOT/'data/processed/ckd_clean.csv');assert len(clean)==400
# Confirm record completeness using the archived audit result and predictor table.
assert clean.drop(columns=[c for c in ['class','record_id','id','target','ckd'] if c in clean],errors='ignore').isna().any(axis=1).sum()==242
labels={'rbc':'Red blood cell appearance','rbcc':'Red blood cell count','wbcc':'White blood cell count','pot':'Potassium','sod':'Sodium'}
fig=plt.figure(figsize=(6.5,2.35))
fig.text(.035,.76,'242',fontsize=29,fontweight='bold',color=BLUE)
fig.text(.035,.55,'records had\nmissing information',fontsize=11,linespacing=1.4)
fig.text(.035,.29,'158',fontsize=24,fontweight='bold',color=MUTED)
fig.text(.035,.10,'records were\ncomplete',fontsize=11,color=MUTED,linespacing=1.4)
fig.add_artist(plt.Line2D([.245,.245],[.09,.93],transform=fig.transFigure,color='#CFD9DF',linewidth=1))
ax=fig.add_axes([.59,.24,.38,.68]);ax.barh(range(5),m.missing_pct,color=BLUE,height=.56)
ax.set_yticks(range(5),[labels[v] for v in m.variable],fontsize=10.5);ax.set_xlim(0,72)
ax.set_xticks([0,20,40],['0%','20%','40%']);ax.set_xlabel('Records missing this measurement',fontsize=10,labelpad=6)
for i,r in enumerate(m.itertuples()):ax.text(r.missing_pct+1.5,i,f'{r.missing_pct:.1f}% ({r.missing_n})',va='center',fontsize=10.5)
ax.tick_params(axis='y',length=0,pad=6)
for s in ['top','right','left']:ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color('#A9B7BF');save(fig,1)

counts=json.loads((TAB/'01_findings.json').read_text())['class_counts'];assert counts=={'ckd':250,'notckd':150}
fig=plt.figure(figsize=(6.5,1.85))
fig.text(.035,.65,'62.5%',fontsize=28,fontweight='bold',color=BLUE)
fig.text(.035,.41,'of records were\nlabeled CKD',fontsize=11,linespacing=1.4)
fig.text(.035,.10,'400 records total',fontsize=10.5,color=MUTED)
fig.add_artist(plt.Line2D([.27,.27],[.09,.93],transform=fig.transFigure,color='#CFD9DF',linewidth=1))
ax=fig.add_axes([.44,.29,.53,.60]);ax.barh([0,1],[150,250],height=.50,color=['#8B9DA8',BLUE])
ax.set_yticks([0,1],['Non-CKD','CKD'],fontsize=11);ax.set_xlim(0,345);ax.set_xticks([0,100,200,300])
ax.set_xlabel('Number of records',fontsize=10.5,labelpad=6)
for y,n in enumerate([150,250]):ax.text(n+6,y,f'{n} ({n/4:.1f}%)',va='center',fontsize=11)
ax.tick_params(axis='y',length=0,pad=7)
for s in ['top','right','left']:ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color('#A9B7BF');save(fig,2)

pred=pd.read_csv(TAB/'03_test_predictions.csv');cm=np.array([[int(((pred.true_ckd==a)&(pred.predicted_ckd==b)).sum()) for b in [0,1]] for a in [0,1]])
assert cm.tolist()==[[30,0],[0,50]]
fig=plt.figure(figsize=(6.5,2.35))
fig.text(.035,.76,'80/80',fontsize=27,fontweight='bold',color=BLUE)
fig.text(.035,.55,'test records matched\nthe existing labels',fontsize=11,linespacing=1.4)
fig.text(.035,.29,'0',fontsize=24,fontweight='bold',color=MUTED)
fig.text(.035,.10,'errors in this\ntest group',fontsize=11,color=MUTED,linespacing=1.4)
fig.add_artist(plt.Line2D([.29,.29],[.09,.93],transform=fig.transFigure,color='#CFD9DF',linewidth=1))
ax=fig.add_axes([.51,.26,.46,.66])
ax.imshow([[1,0],[0,1]],cmap=ListedColormap([GREY,BLUE]),vmin=0,vmax=1,aspect='auto')
ax.set_xticks([0,1],['Non-CKD','CKD'],fontsize=11);ax.set_yticks([0,1],['Non-CKD','CKD'],fontsize=11)
ax.set_xlabel('Model prediction',labelpad=7,fontsize=11);ax.set_ylabel('Existing label',labelpad=9,fontsize=11);ax.tick_params(length=0,pad=7)
desc=[['Matched','Incorrect CKD'],['Missed CKD','Matched']]
for (r,c),n in np.ndenumerate(cm):
 color='white' if r==c else INK
 ax.text(c,r-.1,str(n),ha='center',va='center',fontsize=25,fontweight='bold',color=color)
 ax.text(c,r+.25,desc[r][c],ha='center',va='center',fontsize=10.5,color=color)
ax.set_xticks([.5],minor=True);ax.set_yticks([.5],minor=True);ax.grid(which='minor',color='white',linewidth=2);ax.tick_params(which='minor',length=0)
for s in ax.spines.values():s.set_visible(False)
save(fig,3)
print('Three decision-focused figures generated from source tables; record counts and predictions checked.')



(OUT/'figure_manifest.json').write_text(json.dumps({'figure_files':['figure1_missingness.png','figure2_class_balance.png','figure3_confusion_matrix.png'],'sample_sizes':{'all':400,'training':320,'test':80},'test_matrix':cm.tolist(),'source_tables':['01_missingness_by_variable.csv','01_findings.json','03_test_predictions.csv'],'font':REPORT_FONT},indent=2),encoding='utf-8')

