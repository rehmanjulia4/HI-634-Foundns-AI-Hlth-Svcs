"""APA student paper. Body pages 2-7; supporting material follows references."""
import json
from pathlib import Path
import pandas as pd
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

ROOT=Path(__file__).resolve().parents[1]; TAB=ROOT/'reports/tables';FIG=ROOT/'reports/figures/apa'
TITLE='Classifying Chronic Kidney Disease Records'
URL='https://github.com/rehmanjulia4/HI-634-Foundns-AI-Hlth-Svcs'

def build():
 f=json.loads((TAB/'01_findings.json').read_text());prep=json.loads((TAB/'02_prepare_log.json').read_text())
 r=json.loads((TAB/'03_results.json').read_text());t=r['test']
 cv=pd.read_csv(TAB/'03_model_comparison_cv.csv');sens=pd.read_csv(TAB/'03_sensitivity_checks_cv.csv')
 strategies=pd.read_csv(TAB/'03_missing_data_strategies_cv.csv');missing=pd.read_csv(TAB/'01_missingness_by_variable.csv')
 d=Document();sec=d.sections[0]
 sec.page_width=Inches(8.5);sec.page_height=Inches(11)
 sec.left_margin=sec.right_margin=sec.top_margin=sec.bottom_margin=Inches(1)
 sec.header_distance=Inches(.5);sec.footer_distance=Inches(.5)
 for sty in d.styles:
  if sty.type==1:
   sty.font.name='Times New Roman';sty.font.size=Pt(12);sty.font.color.rgb=RGBColor(0,0,0)
   pf=sty.paragraph_format;pf.line_spacing=2;pf.space_before=Pt(0);pf.space_after=Pt(0)
  for border in list(sty.element.iter(qn('w:pBdr'))):border.getparent().remove(border)
 normal=d.styles['Normal'];normal.paragraph_format.first_line_indent=Inches(.5)
 for styname in ['Title','Heading 1','Heading 2']:
  st=d.styles[styname];st.font.size=Pt(12);st.font.bold=True;st.font.italic=False
  st.paragraph_format.first_line_indent=Inches(0);st.paragraph_format.keep_with_next=True
  st.paragraph_format.alignment=WD_ALIGN_PARAGRAPH.CENTER if styname!='Heading 2' else WD_ALIGN_PARAGRAPH.LEFT
 def format_run(run):
  run.font.name='Times New Roman';run.font.size=Pt(12);run.font.color.rgb=RGBColor(0,0,0)
  rpr=run._element.get_or_add_rPr();fonts=rpr.find(qn('w:rFonts'))
  if fonts is None:fonts=OxmlElement('w:rFonts');rpr.insert(0,fonts)
  for name in ['ascii','hAnsi','eastAsia','cs']:fonts.set(qn('w:'+name),'Times New Roman')
 def para(s='',indent=True,align=None,bold=False,italic=False,keep=False):
  p=d.add_paragraph();p.paragraph_format.first_line_indent=Inches(.5 if indent else 0)
  p.paragraph_format.keep_with_next=keep
  if align is not None:p.alignment=align
  run=p.add_run(s);format_run(run);run.bold=bold;run.italic=italic
  return p
 def head(s):return d.add_heading(s,1)
 def page(s=None):
  if s:
   p=head(s);p.paragraph_format.page_break_before=True;return p
  d.add_page_break()
 def hyperlink(p,label,url):
  rid=p.part.relate_to(url,'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink',True)
  h=OxmlElement('w:hyperlink');h.set(qn('r:id'),rid);rr=OxmlElement('w:r');rpr=OxmlElement('w:rPr')
  fonts=OxmlElement('w:rFonts');fonts.set(qn('w:ascii'),'Times New Roman');fonts.set(qn('w:hAnsi'),'Times New Roman');rpr.append(fonts)
  size=OxmlElement('w:sz');size.set(qn('w:val'),'24');rpr.append(size);rr.append(rpr);tt=OxmlElement('w:t');tt.text=label;rr.append(tt);h.append(rr);p._p.append(h)
 def note(s):
  p=para(indent=False);p.add_run('Note. ').italic=True;p.add_run(s)
  for run in p.runs:format_run(run)
 def figure(n,title,file,width,note_text):
  para(f'Figure {n}',False,bold=True,keep=True)
  para(title,False,italic=True,keep=True)
  p=para(indent=False,keep=True);p.paragraph_format.line_spacing=1;p.alignment=1
  pic=p.add_run().add_picture(str(FIG/file),width=Inches(width))
  pic._inline.docPr.set('descr',note_text)
  note(note_text)
 def table(n,title,headers,rows,widths):
  para(f'Table {n}',False,bold=True,keep=True);para(title,False,italic=True,keep=True)
  tb=d.add_table(rows=1,cols=len(headers));tb.autofit=False;tb.alignment=WD_TABLE_ALIGNMENT.CENTER
  for i,w in enumerate(widths):tb.columns[i].width=Inches(w)
  for i,v in enumerate(headers):tb.rows[0].cells[i].text=v
  for row in rows:
   for cell,v in zip(tb.add_row().cells,row):cell.text=str(v)
  tb.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
  for ri,row in enumerate(tb.rows):
   row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
   for ci,cell in enumerate(row.cells):
    cell.width=Inches(widths[ci]);tcpr=cell._tc.get_or_add_tcPr();b=OxmlElement('w:tcBorders')
    for side in ['top','left','bottom','right','insideH','insideV']:
     el=OxmlElement('w:'+side);el.set(qn('w:val'),'single' if (side=='top' and ri==0) or (side=='bottom' and ri in [0,len(tb.rows)-1]) else 'nil');el.set(qn('w:sz'),'6');el.set(qn('w:color'),'000000');b.append(el)
    tcpr.append(b)
    for p in cell.paragraphs:
     p.paragraph_format.first_line_indent=Inches(0);p.paragraph_format.line_spacing=2
     p.paragraph_format.space_after=p.paragraph_format.space_before=Pt(0)
     for run in p.runs:format_run(run);run.bold=ri==0
  return tb
 # Student page number, without a running head.
 hp=sec.header.paragraphs[0];hp.alignment=2;hp.paragraph_format.first_line_indent=Inches(0)
 rr=hp.add_run();format_run(rr);field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');rr._r.addnext(field)
 for _ in range(3):para('',False)
 para(TITLE,False,align=1,bold=True)
 para('',False)
 for line in ['Julia Rehman','University of Alabama at Birmingham','HI 634-QJ: Foundations of AI for Health Services','Dr. Rahul Sharma','November 3, 2026']:
  para(line,False,align=1)

 # MAIN PAGE 1
 page();d.add_paragraph(TITLE,style='Title')
 para('For this project, I examined how well the measurements in the UCI Chronic Kidney Disease dataset distinguish its two supplied labels, ckd and notckd. My question was: How well do the available measurements reproduce these labels compared with a simple baseline, and how does missing data affect preparation? Each row represents one clinical record. CKD, or chronic kidney disease, is the positive class. This question concerns an existing label; it does not predict future disease.')
 head('Question and Data Source')
 para('The dataset contains 400 records, 24 predictors, and one outcome label (Rubini et al., 2015). Its codebook identifies a hospital source in Tamil Nadu, India, and is dated July 2015. The fields include laboratory measurements and health history. I used the original ARFF data file and its codebook, with a saved UCI importer copy as a cross-check. ARFF means Attribute-Relation File Format. Appendix C lists the variables, units, and missingness.')
 para('I focused on data quality before model performance because a high score can be misleading when measurements are incomplete or closely related to how a diagnosis was assigned. The analysis retained all 400 records. There were 250 ckd records and 150 notckd records, so a model that always predicts ckd would already be correct 62.5% of the time.')
 para('The original downloads were preserved. The project includes a saved run and an independent verification script. The README explains how to repeat the analysis. The course repository link is:')
 pp=para(indent=False);hyperlink(pp,URL,URL)

 # MAIN PAGE 2
 page('Initial Exploration')
 para(f"The audit found {f['missing_cells']:,} missing predictor cells out of {f['total_cells']:,} (10.5%). Only {f['complete_cases']} records were complete. Figure 1 shows the most incomplete fields. CKD records averaged 3.63 missing predictors, compared with 0.69 for notckd. This difference does not tell me why the values are missing.")
 para('Three rows had extra empty fields, 16 cells contained whitespace, and two ckd labels had trailing tabs. Trimming exposed three additional missing markers. There were no duplicates. Ten values fell outside broad screening ranges. I kept them because an unusual measurement is not automatically an error.')
 figure(1,'Predictors With the Most Missing Values','figure1_missingness.png',6.5,
        'N = 400 clinical records. Labels show missing counts and percentages. Red blood cell appearance was missing in 38.0% of records. The eight fields shown have the highest missingness. Data are from Rubini et al. (2015).')

 # MAIN PAGE 3
 page('Preparation and Reproducibility')
 para('I removed extra empty fields, trimmed whitespace, converted documented missing markers, and corrected numerical types. All 400 rows remained. No outcome label was filled, and unusual observed values were retained. Appendix B explains these decisions.')
 para('Complete-case analysis would retain just 122 training records: 32 ckd and 90 notckd. I compared three imputation treatments and retained numerical medians and the most frequent categories. Median filling reduces spread; red cell count standard deviation falls 17.8%. Figure 2 shows the class balance.')
 para('A stratified 80/20 split with seed 634 gives 320 training and 80 test records. Imputation, scaling, and encoding learn within training folds. Training has 200 ckd and 120 notckd records; testing has 50 and 30. Numeric fields are scaled and categories one-hot encoded. The label and ID are excluded. Specific gravity, albumin, and sugar are modeled as ordered numbers.')
 figure(2,'Class Balance in the Original Dataset','figure2_class_balance.png',6.5,
        'N = 400 clinical records: 250 ckd (62.5%) and 150 notckd (37.5%). Bars show supplied class counts. Predicting ckd for every record would give 62.5% accuracy while missing every notckd record.')

 # MAIN PAGE 4
 page('Model Comparison')
 para('I compared a majority baseline with logistic regression, a decision tree, and a random forest using scikit-learn (Pedregosa et al., 2011). Logistic regression models class probability. The depth-limited tree uses branching rules, while the forest averages 300 trees. Settings were fixed. Appendix A gives the comparison table.')
 para('Repeated stratified five-fold cross-validation, with three repeats, used training data only. Macro F1 averages class F1 scores equally. F1 balances precision and recall. CKD recall had to reach .95, a project rule rather than a validated clinical target. Recall alone rewards the baseline for calling everyone ckd, despite identifying no notckd records.')
 para(f"Within .01 of the best macro F1, the rule prefers simplicity. It selected logistic regression ({cv.iloc[1].macro_f1_mean:.3f}), close to the forest ({cv.iloc[3].macro_f1_mean:.3f}). The scores are close in Figure 3, but that does not prove the methods perform equally well.")
 figure(3,'Cross-Validated Macro F1 by Method','figure3_model_comparison.png',6.5,
        'n = 320 training records across 15 validation folds. Dots show means; whiskers show one fold standard deviation in either direction, not confidence intervals. Repeated folds reuse records. The fitted alternatives substantially outperform the majority baseline.')

 # MAIN PAGE 5
 page('Test Results and Errors')
 para(f"Logistic regression correctly classified {t['tp']} ckd and {t['tn']} notckd test records (Figure 4). Precision, recall, F1, macro F1, accuracy, and specificity were 1.000. Precision measures positive prediction correctness; recall measures CKD records found. In a hypothetical screening setting, I would prioritize avoiding false negatives because a missed CKD label could falsely reassure. False positives could prompt unnecessary follow-up. Neither occurred here.")
 figure(4,'Confusion Matrix for the Selected Logistic Regression','figure4_confusion_matrix.png',3.5,
        'n = 80 test records: 50 ckd and 30 notckd. Rows show supplied labels and columns show predictions. The two off-diagonal cells count errors. No errors occurred in this fixed sample; this does not imply error-free performance elsewhere.')
 para(f"Separate five-fold training predictions produced {t['oof_train_errors']} error: record 103 was labeled ckd but predicted notckd. It had a missing red cell count and hemoglobin of 13.9. This was a training validation error, with a different fold arrangement.")
 para(f"Approximate 95% Wilson intervals are [{t['wilson_95ci_recall'][0]:.3f}, {t['wilson_95ci_recall'][1]:.3f}] for recall and [{t['wilson_95ci_specificity'][0]:.3f}, {t['wilson_95ci_specificity'][1]:.3f}] for specificity. They assume independent sampling and do not capture site or label bias. Resampling perfect predictions does not establish certainty.")

 # MAIN PAGE 6
 page('Interpretation and Limitations')
 para('The measurements reproduce these labels very well, but I would be cautious about interpreting that as clinical usefulness. Hemoglobin, creatinine, albumin, and specific gravity may overlap with information used to assign the labels. Measurement timing and label construction remain uncertain.')
 para(f"Missingness flags alone achieved cross-validated accuracy of {sens.iloc[0].accuracy_mean:.3f}, suggesting collection patterns carry label information. Removing five highly informative predictors lowered macro F1 to {sens.iloc[1].macro_f1_mean:.3f}. This is descriptive because those fields were selected from the full training sample, rather than within each fold.")
 para('Other limits include the small historical sample, uncertain repeated-patient status, unverified extreme values, and imputation effects. Original exploration used all 400 labels, although test records were excluded from fitting and selection. Repeating that split verifies the analysis without creating new external validation. Calibration and subgroup performance were not assessed.')
 para('For me, the value of this project is understanding the data and decisions behind the score. My next step would be to clarify label assignment and timing, then validate independently before considering practical use.')
 d.add_heading('Use of Coding Agents',2)
 para('Coding agents helped develop the scripts, figures, and report. I ran the analysis in my Anaconda environment and checked that it finished successfully. The independent verifier checks raw parsing, split membership, predictions, and the saved importer copy. I also reviewed the model comparison and confusion matrix. I remain responsible for explaining the choices and limitations.')

 page('References')
 refs=[
 ('Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. ', 'Journal of Machine Learning Research, 12', ', 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html'),
 ('Rubini, L., Soundarapandian, P., & Eswaran, P. (2015). ', 'Chronic kidney disease', ' [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5G020')]
 for lead,ital,tail in refs:
  p=para(indent=False);p.paragraph_format.left_indent=Inches(.5);p.paragraph_format.first_line_indent=Inches(-.5)
  for text,italic in [(lead,False),(ital,True),(tail,False)]:
   run=p.add_run(text);format_run(run);run.italic=italic

 page('Appendix A')
 para('Supporting Model Results',False,align=1,bold=True)
 table(1,'Cross-Validated Performance on the Training Sample',['Method','Recall','Precision','Macro F1','Accuracy'],
       [[name,f'{x.recall_mean:.3f}',f'{x.precision_mean:.3f}',f'{x.macro_f1_mean:.3f}',f'{x.accuracy_mean:.3f}'] for name,x in zip(['Majority baseline','Logistic regression','Decision tree','Random forest'],cv.itertuples())],[2.3,1.05,1.05,1.05,1.05])
 note('n = 320 training records. Entries are means over 15 validation folds. Recall and precision refer to ckd. Macro F1 averages the two class F1 scores equally. Tree: maximum depth 4, minimum leaf size 5. Forest: 300 trees, minimum leaf size 2. Logistic regression uses default regularization strength and 3,000 maximum iterations.')
 para('The three missing-data treatments were compared using training data only. Logistic regression used numerical median filling with categorical most-frequent filling, with and without numerical missingness indicators, and an explicit Unknown category for missing categorical values. The simple option was retained without treating small score differences as proof of equivalence.')
 page('Appendix A Continued')
 table(2,'Missing-Data Treatments Compared During Training',['Treatment','Method','Recall','Macro F1'],
       [[('Median/mode + numerical flags' if '+ ' in x.strategy else 'Median/Unknown' if 'Unknown' in x.strategy else 'Median/mode'),('Logistic' if x.model=='Logistic regression' else 'Forest'),f'{x.recall_mean:.3f}',f'{x.macro_f1_mean:.3f}'] for x in strategies.itertuples()],[2.7,1.5,1.15,1.15])
 note('n = 320 training records. Entries are means over repeated five-fold cross-validation. Mode means the most frequent category. Unknown is an explicit category for missing categorical values. The saved CSV files retain unrounded results and all other measures.')
 para('Precision is the share of predicted ckd records labeled ckd. Recall is the share of labeled ckd records found. F1 is the harmonic mean of precision and recall. Macro F1 averages class F1 scores equally. Specificity is the share of notckd records correctly classified.')

 page('Appendix B')
 para('Preparation Decision Log',False,align=1,bold=True)
 table(3,'How Raw Findings Informed Preparation',['Finding','Decision and Reason'],[
  ['Three extra empty fields','Remove only the extra empty field and log its position. Retain all 400 records.'],
  ['Whitespace and question marks','Trim text and convert the documented missing marker. Missing predictor cells remain 1,012.'],
  ['Two ckd labels with a tab','Merge spellings. No outcome label is missing or filled.'],
  ['Few complete training records','Keep incomplete records and fit imputation inside training folds.'],
  ['Ten flagged extreme values','Retain and flag because an unusual value is not automatically an error.'],
  ['Unequal class sizes','Use a stratified split and class-sensitive evaluation measures.']],[2.2,4.3])
 note('The raw ARFF file remains unchanged. Data rows 70 and 73 contain trailing empty fields; row 370 contains an extra interior empty field. Sample size is 400 before and after cleaning.')
 para('Observed ranges included age 2–90 years, blood pressure 50–180 mm Hg, serum creatinine 0.4–76 mg/dL, and hemoglobin 3.1–17.8 in the source hemoglobin units.')

 page('Appendix C')
 para('Predictor Definitions and Missingness',False,align=1,bold=True)
 from common import PREDICTORS,FULL_NAMES,UNITS,CATEGORICAL,ORDINAL
 clean=pd.read_csv(ROOT/'data/processed/ckd_clean.csv');m=missing.set_index('variable')
 # Two deliberate pages keep definitions at 12 point rather than shrinking the table.
 for group,columns in enumerate([PREDICTORS[:14],PREDICTORS[14:]]):
  if group:page('Appendix C Continued')
  rows=[]
  for c in columns:
   unit=(', '.join(sorted(clean[c].dropna().astype(str).unique())) if c in CATEGORICAL else UNITS[c])
   rows.append([c,FULL_NAMES[c],unit,f"{m.loc[c,'missing_pct']:.1f}"])
  table(4+group,'Predictor Meanings and Missing Values' if not group else 'Categorical Predictors and Missing Values',
        ['Field','Meaning','Units or Values','Missing %'],rows,[.6,2.1,2.75,1.05])
  note('N = 400 clinical records. Definitions come from the codebook and ARFF header distributed with Rubini et al. (2015). The label and record ID are excluded from predictors. Specific gravity, albumin, and sugar are treated as ordered codes. The source abbreviates hemoglobin units as gms; assay details are incompletely documented.')
 # Direct font formatting also overrides inherited theme fonts in Word headings.
 for paragraph in d.paragraphs:
  for run in paragraph.runs:format_run(run)
 d.core_properties.author='Julia Rehman';d.core_properties.title=TITLE
 d.core_properties.subject='HI 634-QJ: Foundations of AI for Health Services — Assignment 1'
 out=ROOT/'reports/HI634_CKD_Report.docx';d.save(out)
 print(out)

if __name__=='__main__':build()

