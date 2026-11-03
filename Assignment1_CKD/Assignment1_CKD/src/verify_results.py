"""Independent raw parsing, split reconstruction and selected-model check.

python src/verify_results.py [--original-zip PATH]
Does not import the project parser or model-building helpers.
"""
import argparse,hashlib,json,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.metrics import confusion_matrix
root=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--original-zip');args=ap.parse_args()
rawfile=root/'data/raw/ckd/chronic_kidney_disease_full.arff'
lines=rawfile.read_text(encoding='latin-1').splitlines()
names=[v.split()[1].strip("'\"") for v in lines if v.lower().startswith('@attribute')]
records=[]; repairs=[]; started=False
for line in lines:
    if line.strip().lower()=='@data': started=True; continue
    if not started or not line.strip(): continue
    row=line.split(',')
    if len(row)==26:
        blanks=[i for i,v in enumerate(row) if not v.strip()]
        assert len(blanks)==1
        repairs.append(len(records)+1); row.pop(blanks[0])
    assert len(row)==25
    records.append(row)
raw=pd.DataFrame(records,columns=names)
ws=int(sum(v!=v.strip() for row in records for v in row))
frame=raw.map(lambda v: np.nan if v.strip()=='?' else v.strip())
num=['age','bp','bgr','bu','sc','sod','pot','hemo','pcv','wbcc','rbcc','sg','al','su']
cat=['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane']
for c in num: frame[c]=pd.to_numeric(frame[c],errors='raise')
missing=int(frame[num+cat].isna().sum().sum())
assert len(frame)==400 and missing==1012 and ws==16 and repairs==[70,73,370]
assert frame['class'].value_counts().to_dict()=={'ckd':250,'notckd':150}
assert int(frame[num+cat].notna().all(axis=1).sum())==158
assert int(frame.duplicated().sum())==0
train_idx,test_idx=train_test_split(np.arange(400),test_size=.2,random_state=634,stratify=frame['class'])
train_idx=np.sort(train_idx);test_idx=np.sort(test_idx)
saved=pd.read_csv(root/'data/processed/split_assignment.csv')
assert np.array_equal(saved.loc[saved.split=='test','record_id'].values,test_idx+1)
assert set(train_idx).isdisjoint(test_idx)
prep=ColumnTransformer([
 ('num',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),num),
 ('cat',make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore')),cat)])
model=make_pipeline(prep,LogisticRegression(max_iter=3000,random_state=634))
y=frame['class'].eq('ckd').astype(int)
model.fit(frame.iloc[train_idx][num+cat],y.iloc[train_idx])
pred=model.predict(frame.iloc[test_idx][num+cat]);cm=confusion_matrix(y.iloc[test_idx],pred,labels=[0,1])
assert cm.tolist()==[[30,0],[0,50]]
pred_saved=pd.read_csv(root/'reports/tables/03_test_predictions.csv')
assert np.array_equal(pred_saved.predicted_ckd,pred)
assert np.array_equal(pred_saved.record_id,test_idx+1)
assert np.allclose(pred_saved.probability_ckd,model.predict_proba(frame.iloc[test_idx][num+cat])[:,1])
importer=pd.read_csv(root/'data/processed/uci_importer_copy.csv').rename(columns={'wc':'wbcc','rc':'rbcc'})
for c in importer:
    importer[c]=importer[c].map(lambda v: np.nan if pd.isna(v) or str(v).strip()=='?' else str(v).strip())
for c in num: importer[c]=pd.to_numeric(importer[c],errors='raise')
pd.testing.assert_frame_equal(frame[names],importer[names],check_dtype=False)
checks={'raw_records':400,'missing_predictor_cells':missing,'complete_records':158,'whitespace_cells':ws,
 'malformed_rows':repairs,'independent_test_confusion_matrix':cm.tolist(),
 'independent_predictions_match_saved':True,'cached_importer_all_cleaned_values_match':True,
 'cached_importer_is_not_fresh_network_verification':True,'zip_comparisons':{}}
if args.original_zip:
    import io
    with zipfile.ZipFile(args.original_zip) as z:
        for entry in z.namelist():
            if '/data/raw/' in entry and not entry.endswith('/'):
                local=root/entry.split('/',1)[1]
                assert hashlib.sha256(local.read_bytes()).digest()==hashlib.sha256(z.read(entry)).digest()
        for name in ['03_model_comparison_cv.csv','03_missing_data_strategies_cv.csv','03_sensitivity_checks_cv.csv']:
            entry=next(n for n in z.namelist() if n.endswith('/reports/tables/'+name))
            old=pd.read_csv(io.BytesIO(z.read(entry)));new=pd.read_csv(root/'reports/tables'/name)
            cols=list(old.select_dtypes('number').columns)
            diff=float(np.nanmax(np.abs(old[cols].values-new[cols].values)))
            checks['zip_comparisons'][name]={'max_absolute_numeric_difference':diff,'within_1e_10':diff<1e-10}
        checks['raw_files_match_original_zip']=True
checks['all_assertions_passed']=True
(root/'reports/independent_verification.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
