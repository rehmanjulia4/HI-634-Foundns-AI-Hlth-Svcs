"""Recreate the approved report while preserving its reviewed wording/layout.

Changed evidence requires editorial review rather than silently reusing claims.
"""
from __future__ import annotations
import hashlib
import json
import posixpath
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'
FIG = ROOT / 'reports/figures/apa'
SOURCE_FILES = [
 'data/processed/ckd_clean.csv', 'data/raw/ckd/chronic_kidney_disease.info.txt',
 'reports/tables/01_findings.json', 'reports/tables/01_missingness_by_variable.csv',
 'reports/tables/01_numeric_ranges.csv', 'reports/tables/02_prepare_log.json',
 'reports/tables/02_split_class_counts.csv', 'reports/tables/02_imputation_before_after_train.csv',
 'reports/tables/03_model_comparison_cv.csv', 'reports/tables/03_missing_data_strategies_cv.csv',
 'reports/tables/03_sensitivity_checks_cv.csv', 'reports/tables/03_test_predictions.csv',
 'reports/tables/03_training_oof_errors.csv', 'reports/tables/03_results.json']

def rounded(value):
 if isinstance(value, dict): return {k: rounded(v) for k,v in value.items()}
 if isinstance(value, list): return [rounded(v) for v in value]
 return round(value, 3) if isinstance(value, float) else value

def evidence(path):
 if path.suffix == '.json': return rounded(json.loads(path.read_text(encoding='utf-8-sig')))
 if path.suffix != '.csv': return {'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 frame = pd.read_csv(path)
 if path.name == '03_test_predictions.csv':
  frame = frame[['record_id','true_ckd','predicted_ckd']]
 canonical = frame.round(3).to_json(orient='records')
 if path.name == 'ckd_clean.csv': return {'sha256':hashlib.sha256(canonical.encode('utf-8')).hexdigest()}
 return json.loads(canonical)

def validate_results():
 approved = json.loads((DOCS/'report_sources.json').read_text(encoding='utf-8'))
 template = DOCS/'report_template.docx'
 if hashlib.sha256(template.read_bytes()).hexdigest() != approved['template_sha256']:
  raise ValueError('The approved template changed. Review and update its evidence record before rebuilding.')
 changed = [name for name in SOURCE_FILES if evidence(ROOT/name) != approved['sources'][name]]
 if changed:
  raise ValueError('Results differ from the approved paper. Review its claims before rebuilding: '+', '.join(changed))
 return template

def build():
 template = validate_results()
 ns = {'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
       'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
 chart_names = ['figure1_missingness.png','figure2_class_balance.png','figure3_confusion_matrix.png']
 output = ROOT/'reports/HI634_CKD_Report.docx'
 output.parent.mkdir(exist_ok=True)
 with zipfile.ZipFile(template) as source:
  document = ET.fromstring(source.read('word/document.xml'))
  relationships = ET.fromstring(source.read('word/_rels/document.xml.rels'))
  targets = {item.attrib['Id']:item.attrib['Target'] for item in relationships}
  images = document.findall('.//a:blip',ns)
  if len(images) != 3: raise ValueError('Expected exactly three approved report figures.')
  replacements = {}
  for blip,filename in zip(images,chart_names):
   rel = blip.attrib['{'+ns['r']+'}embed']
   entry = posixpath.normpath(posixpath.join('word',targets[rel]))
   replacements[entry] = (FIG/filename).read_bytes()
  # Preserve XML verbatim: prose, tables, citations, layout, page numbering.
  with tempfile.NamedTemporaryFile(dir=output.parent,suffix='.docx',delete=False) as handle:
   temporary = Path(handle.name)
  try:
   with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as result:
    for item in source.infolist():
     result.writestr(item,replacements.get(item.filename,source.read(item.filename)))
   with zipfile.ZipFile(temporary) as check:
    if check.testzip() is not None: raise ValueError('Report file-integrity check failed.')
   temporary.replace(output)
  finally: temporary.unlink(missing_ok=True)
 print('Approved prose, tables, and page formatting preserved; evidence matches.')
 print(output)

if __name__ == '__main__': build()
