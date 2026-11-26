"""Build the DDP report in the format of the senior's report (Aadit Mahajan, same guide):
Times New Roman, justified, title page with IITM logo, Abstract, numbered sections 1-6, numbered figures/tables,
academic numbered references, 'Page X of Y' footer. No future-work section (per the user).
Detail and numbers are drawn from the project's result CSVs and from the student's own slide deck (fact-checked).
Output: DDP_report_Mahima.docx
"""
import pandas as pd
from docx import Document
from docx.shared import Pt, Cm, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

import os; FIG = os.path.dirname(os.path.abspath(__file__))
LOGO = FIG + '/report_assets/iitm_logo_0.png'

doc = Document()
st = doc.styles['Normal']; st.font.name = 'Times New Roman'; st.font.size = Pt(12)
st.element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')
pf = st.paragraph_format; pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY; pf.space_after = Pt(10); pf.line_spacing = 1.15
for s in doc.sections:
    s.left_margin = s.right_margin = Cm(2.5); s.top_margin = s.bottom_margin = Cm(2.4)

def _run(p, text, bold=False, italic=False, size=12, sup=False):
    r = p.add_run(text); r.bold = bold; r.italic = italic; r.font.size = Pt(size)
    r.font.name = 'Times New Roman'; r.font.superscript = sup; return r

def para(text='', bold=False, italic=False, align='justify', size=12, after=10, before=0):
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(after); p.paragraph_format.space_before = Pt(before)
    p.alignment = {'justify': WD_ALIGN_PARAGRAPH.JUSTIFY, 'center': WD_ALIGN_PARAGRAPH.CENTER, 'left': WD_ALIGN_PARAGRAPH.LEFT}[align]
    if text: _run(p, text, bold, italic, size)
    return p

def heading(num_text, level=1):
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(16 if level == 1 else 10); p.paragraph_format.space_after = Pt(8)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT; _run(p, num_text, bold=True, size=14 if level == 1 else 12); return p

def cite(p, text, refs, tail='.'):
    _run(p, text); _run(p, '[' + refs + ']', sup=True); _run(p, tail)

def bullet(t, lead=None):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY; p.paragraph_format.space_after = Pt(6)
    _run(p, '• ')
    if lead: _run(p, lead, bold=True)
    _run(p, t); return p

def numbered(i, t, lead=None):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY; p.paragraph_format.space_after = Pt(6)
    _run(p, f'{i}. ')
    if lead: _run(p, lead, bold=True)
    _run(p, t); return p

def figure(path, caption, width_in=5.6):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(path, width=Inches(width_in))
    c = doc.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER; c.paragraph_format.space_after = Pt(12)
    _run(c, caption, size=11)

def table_caption(text):
    c = doc.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER; c.paragraph_format.space_before = Pt(8)
    c.paragraph_format.keep_with_next = True
    _run(c, text, size=11)

def simple_table(headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers)); t.style = 'Table Grid'; t.alignment = 1
    if widths: t.autofit = False
    for j, h in enumerate(headers):
        r = t.rows[0].cells[j].paragraphs[0].add_run(h); r.bold = True; r.font.name = 'Times New Roman'; r.font.size = Pt(11)
    for row in rows:
        cells = t.add_row().cells
        for j, v in enumerate(row):
            r = cells[j].paragraphs[0].add_run(str(v)); r.font.name = 'Times New Roman'; r.font.size = Pt(11)
    for row in t.rows[:-1]:
        for c in row.cells:
            for pp in c.paragraphs: pp.paragraph_format.keep_with_next = True
    for row in t.rows:
        trPr = row._tr.get_or_add_trPr(); cs = OxmlElement('w:cantSplit'); trPr.append(cs)
    if widths:
        for j, w in enumerate(widths): t.columns[j].width = Cm(w)
        for row in t.rows:
            for j, w in enumerate(widths): row.cells[j].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

def page_field(p, which):
    fld = OxmlElement('w:fldSimple'); fld.set(qn('w:instr'), which); r = OxmlElement('w:r'); t = OxmlElement('w:t'); t.text = '1'; r.append(t); fld.append(r); p._p.append(fld)

footer = doc.sections[0].footer; fp = footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
_run(fp, 'Page '); page_field(fp, 'PAGE'); _run(fp, ' of '); page_field(fp, 'NUMPAGES')

# ================= TITLE PAGE =================
for _ in range(2): doc.add_paragraph()
para('Recommending Counter-ions for Active Pharmaceutical Ingredients to form API Ionic Liquids', bold=True, align='center', size=18, after=18)
para("Dual Degree Project Report", italic=True, align='center', size=13, after=28)
para('Jilla Sai Mahima (BE22B047)', bold=True, align='center', size=13, after=2)
para('Guide: Prof. Sanjib Senapati', bold=True, align='center', size=13, after=18)
para('Dual Degree (B.Tech. and M.Tech.), Biotechnology', align='center', size=12, after=2)
para('Department of Biotechnology', bold=True, align='center', size=12, after=16)
lp = doc.add_paragraph(); lp.alignment = WD_ALIGN_PARAGRAPH.CENTER; lp.add_run().add_picture(LOGO, width=Inches(2.3))
doc.add_paragraph()
para('Indian Institute of Technology Madras', bold=True, align='center', size=12, after=4)
para('November 2026', align='center', size=12)
doc.add_page_break()

# ================= ABSTRACT =================
para('Abstract', bold=True, align='center', size=13, after=10)
para('Many approved drugs are limited by low aqueous solubility and poor bioavailability. One way to address this is '
     'to pair the drug ion with a counter-ion to form a salt or an ionic liquid, which can change the solubility and '
     'uptake of the drug. The difficult step is choosing the counter-ion, because one drug can pair with hundreds of '
     'ions and testing each in the laboratory is slow. This project builds a recommender for that choice. Given a drug, '
     'it ranks the counter-ions that could pair with it and returns a score for each. The pairs were collected by '
     'searching about 4,600 articles, giving 652 reported API-counter-ion pairs over 131 drugs and 291 counter-ions. '
     'The counter-ion pKₐ values needed for the score were taken from the literature where available, treated as '
     'not required for permanently charged ions, and otherwise predicted with pKasso, an open-source protonation-state '
     'tool. pKasso was chosen over density-functional theory (DFT), which is slower and computationally expensive, '
     'because it needs a full quantum-chemistry calculation for every ion, and which was also less accurate for bases. '
     'The score itself is one additive equation built from four feature groups: two-dimensional structural similarity, '
     'three-dimensional shape similarity, the pKₐ relationship, and electrostatic-surface complementarity. Tested '
     'leave-drug-out with two data-leakage controls, the model places a reported counter-ion in the top ten for about '
     '72% of unseen drugs, against 22% for a random baseline, and is as accurate as the best of eight machine-learning '
     'models while staying interpretable. Most of the signal comes from structural similarity to drugs that already have '
     'a reported counter-ion. Solubility, toxicity and bioavailability of the API-ILs are deliberately not predicted. '
     'These are properties of the ion pair, not of either ion alone: they depend on how the two ions bond to each other '
     '(proton transfer, hydrogen bonding and ion pairing in water), so the same drug ion can behave very differently with '
     'different partners. Values can be found for many single ions, but measured values for the bonded pairs are few '
     'and reported in different units, media and assays, which is far too little to train a model. The report records '
     'both what worked and what did not, so the limits of the method are clear.')
doc.add_page_break()

# ================= 1. INTRODUCTION =================
heading('1. Introduction')
p = para(); cite(p, 'Active pharmaceutical ingredients (APIs) are the molecules that give a drug its therapeutic effect. '
         'A large share of modern drug candidates are solids that dissolve poorly in water, and poor aqueous solubility '
         'is one of the main reasons a candidate fails, because the solubility, the dissolution rate and the intestinal '
         'permeability limit how much drug reaches the target. About 40% of marketed drugs, and as many as 70 to 90% of '
         'candidates in the development pipeline, are poorly water-soluble', '24', '. A poorly soluble drug often needs a higher dose, which raises the '
         'chance of side effects, and solid drugs also bring problems such as polymorphism, where the same molecule '
         'crystallises in more than one form with different properties.')
p = para(); cite(p, 'One practical route around these problems is to convert the solid drug into a salt or a liquid salt '
         'form, an ionic liquid (IL). An ionic liquid is a salt that is liquid below about 100 degrees Celsius, and in an '
         'API-IL the active molecule itself is one of the two ions', '2', ' (Figure 1). Pairing a drug with a well-chosen counter-ion '
         'can raise its solubility by a large factor. Ibuprofen is a clear example: pure ibuprofen is practically '
         'insoluble in water (about 0.02 grams per litre), but formulating it with cholinium-based ionic liquids raises '
         'its aqueous solubility by several hundred- to several thousand-fold')
_run(p, '[3]', sup=True); _run(p, ' (Figure 2). Because an approved drug has already passed safety testing, reformulating '
     'it as an API-IL also lowers the barrier to use.')
figure(FIG + '/fig_api_il_example.png', 'Figure 1. An API ionic liquid. The drug ibuprofen, as its anion, is paired with the '
       'cholinium cation; together they form the ionic liquid cholinium ibuprofenate.', 5.4)
figure(FIG + '/fig_solubility_boost.png', 'Figure 2. Pairing a poorly soluble drug with a counter-ion can raise its '
       'aqueous solubility by orders of magnitude. The example is ibuprofen with a cholinium-based ionic liquid; the bars '
       'are on a logarithmic scale.', 5.2)
para('The practical question this project addresses is narrow: for a given drug, which counter-ion should be paired with '
     'it? The space of candidate counter-ions is large, the same few ions tend to be reused, and each '
     'candidate pair has to be made and tested in the laboratory. A tool that ranks the counter-ions most likely to form '
     'a real pair with a given drug would narrow that search to a short list worth testing first.')
p = para(); cite(p, 'The idea the recommender is built on is simple: chemically similar drugs tend to pair with similar '
         'counter-ions. It follows from the similar-property principle of medicinal chemistry, that molecules with similar '
         'structures tend to have similar properties', '17, 18'); _run(p, '. For salt formation this is expected, because the choice of counter-ion depends on '
     'the drug’s acidic or basic group, its pKₐ, its size and its lipophilicity, and similar drugs share these. API-IL '
     'studies already work this way: a counter-ion that works for one drug is carried across a family of related drugs, '
     'for example cholinium with several anti-inflammatory acids'); _run(p, '[3, 34, 36]', sup=True)
_run(p, ' and lidocainium with several anti-inflammatory anions'); _run(p, '[23]', sup=True); _run(p, '.')
para('The principle also holds in this project’s own data, which contains only salt-type pairs (proton transfer or a '
     'permanently charged ion). Two drugs of the same charge that are structurally dissimilar (2D similarity below 0.2) '
     'share at least one counter-ion 24% of the time; at a similarity of 0.3 to 0.4 this rises to 58%, and at 0.4 to 0.5 '
     'to 71%. For each drug, its most similar other drug shares a counter-ion with it for 53% of drugs, against 27% for '
     'an average other drug. Similar partner-recommendation methods have been published for co-crystals, but co-crystals '
     'form without proton transfer, so they are not used as evidence here. This project applies the principle to '
     'API-counter-ion pairs, scores every candidate counter-ion for a drug, and explains each suggestion.')

# ================= 2. LITERATURE =================
heading('2. Critical Review of Literature')
para('This section reviews the areas the project draws on: ionic liquids as a form of APIs, the rules that decide '
     'whether a drug and a counter-ion form a salt, the methods for predicting pKₐ, and structure-based '
     'recommendation.')

heading('2.1 Ionic liquids as a form of APIs', 2)
p = para(); cite(p, 'Ionic liquids have been studied for two decades as tunable solvents and materials', '1',
         ', and a large body of work applies them to pharmaceuticals. Hough and co-workers described API-ILs as a '
         '"third evolution" of ionic liquids, in which the active molecule is one of the ions'); _run(p, '[2]', sup=True)
_run(p, ', and reviews by Shamshina and Rogers survey the range of API-IL chemistry, including prodrug conversion, '
     'amino-acid-ester counter-ions, and double-salt forms'); _run(p, '[4]', sup=True); _run(p, '. Individual studies show '
     'that a well-chosen counter-ion improves a drug: lidocaine has been made as a room-temperature-liquid salt with '
     'anti-inflammatory anions such as salicylate'); _run(p, '[23]', sup=True); _run(p, ', and cholinium-based ionic '
     'liquids raise the solubility of poorly soluble drugs such as ibuprofen'); _run(p, '[3]', sup=True); _run(p, '.')

heading('2.2 Counter-ion selection and the pKₐ rule', 2)
p = para(); cite(p, 'Whether an acid and a base form a salt (full proton transfer) or a co-crystal (no transfer) is '
         'governed mainly by the difference in their pKₐ values, ΔpKₐ = pKₐ(protonated base) − '
         'pKₐ(acid). From a survey of 6,465 acid-base crystal structures, Cruz-Cabeza showed that a salt forms '
         'almost always when ΔpKₐ is above about 4 and a co-crystal almost always when it is below about −1, '
         'with the salt fraction rising smoothly in between', '5')
p = para(); cite(p, 'Brittain turned the same idea into a number: from the acid and base pKₐ values alone, the '
         'fraction of salt expected at equimolar mixing can be estimated', '6',
         '. This project uses that estimate as one of its pKₐ features. For the ionic-liquid case specifically, the '
         'degree of proton transfer and the low melting point also depend on ion size, flexibility and charge spread')
_run(p, '[7, 8, 9]', sup=True); _run(p, '.')

heading('2.3 Predicting pKₐ', 2)
p = para(); cite(p, 'Because many candidate counter-ions have no measured pKₐ, this value often has to be predicted. '
         'Machine-learning predictors exist: a random forest trained on a literature compilation reaches a test error of '
         'about 0.5 pKₐ units', '10', ', and a graph neural network with multi-instance learning predicts a pKₐ '
         'for each ionizable atom and combines them'); _run(p, '[11]', sup=True); _run(p, '. Both, however, are trained on '
         'general, mostly single-site drug-like molecules, which limits how well they transfer to multi-charged '
         'counter-ions. A physics-based alternative is calibrated DFT, which computes the deprotonation energy and fits a '
         'short linear correction to measured values; this needs no training chemistry and matches the acid-base problem '
         'directly'); _run(p, '[12]', sup=True); _run(p, '. A third route predicts a per-site (micro) pKₐ for each '
     'ionizable atom and combines these into one overall value, which naturally handles molecules with several charge '
     'sites'); _run(p, '[25]', sup=True); _run(p, '. Section 4.2 reports how these methods performed on this '
     'project’s ions and which one was chosen.')

heading('2.4 Structure-based recommendation', 2)
p = para(); cite(p, 'Structure-based methods compare molecules through fingerprints and descriptors; extended-connectivity '
         '(Morgan) fingerprints with the Tanimoto coefficient are a standard choice', '13',
         ', implemented in the RDKit and scikit-learn libraries'); _run(p, '[14, 15]', sup=True); _run(p, ', while electronic '
     'descriptors such as partial charges come from the semi-empirical GFN2-xTB method'); _run(p, '[16]', sup=True)
_run(p, '. Comparing a new molecule with known ones rests on the similar-property principle'); _run(p, '[17, 18]', sup=True)
_run(p, '. Recommender systems are already used for ionic liquids, but to predict properties of cation–anion combinations '
     'rather than to choose a partner for a drug'); _run(p, '[19]', sup=True); _run(p, '. Property-prediction models for ionic-liquid solubility');
_run(p, '[22]', sup=True); _run(p, ' and for drug absorption exist, but they predict a property of one molecule rather '
     'than rank which counter-ion suits which drug, which is the task here.')

# ================= 3. OBJECTIVES =================
heading('3. Project Objectives')
para('The aim is a recommender that, given a drug, ranks the counter-ions most likely to form a real pair with it, and '
     'explains each suggestion. This is pursued through the following tasks:')
objs = [('Data curation', 'Collect reported API-counter-ion pairs from the literature and patents, and prepare the drug '
         'and ion structures, pKₐ values and electronic descriptors for modelling.'),
        ('pKₐ assignment', 'Obtain a pKₐ for every counter-ion, from measured values, from the permanent-ion '
         'rule, or from a calculation method chosen after comparing machine-learning and physics-based options.'),
        ('Feature design', 'Build pair features that capture structural similarity, shape, the pKₐ relationship and '
         'electrostatic complementarity.'),
        ('Scoring model', 'Combine the features into a single, interpretable score per drug-ion pair, and calibrate it so '
         'the score can be read as a probability.'),
        ('Rigorous evaluation', 'Test the model leave-drug-out with controls against data leakage, compare it against a '
         'random baseline and standard machine-learning models, and measure what each feature group contributes.'),
        ('Delivery', 'Make the recommender usable for the known drugs and for a new drug through a web application and a '
         'notebook, with a safety label on each suggested counter-ion.')]
for i, (t, d) in enumerate(objs, 1):
    numbered(i, d, lead=t + ': ')

# ================= 4. METHODOLOGY =================
heading('4. Methodology')
para('Figure 3 gives an overview of how the recommender works. For a drug, every counter-ion of the opposite charge is a '
     'candidate. Each drug–ion pair is described by four groups of features, computed against the training drugs and '
     'the ions they are reported with. One additive equation turns the features into a probability, which is calibrated '
     'and shown as a score from 0 to 100, and the candidates are ranked by it. The sections below describe each step.')
figure(FIG + '/fig_model_workflow.png', 'Figure 3. Overview of the recommender, shown for ibuprofen held out as an unseen drug. '
       '(a) The query drug and its 158 candidate counter-ions. (b) Each drug–ion pair is described by four feature '
       'groups (shading shows the value relative to all candidates). (c) The learned spline curve for each group; the '
       'curves are added and calibrated into a score. (d) Candidates ranked by score; red bars are pairs reported in '
       'the literature, grey bars are new suggestions.', 6.3)

heading('4.1 Dataset', 2)
para('The reported pairs were collected by a literature search: about 1,935 PubMed abstracts and about 2,707 Europe PMC '
     'full-text articles were screened, roughly 4,600 articles in total. After reading, curation and de-duplication, '
     'this gave 652 reported API-counter-ion pairs. Of these, 384 are Level-1 pairs, where a sentence names the pair '
     'directly, and 268 are Level-2 pairs, where the pair is implied (for example, a paper states that an ionic liquid '
     'was made "by acid-base reaction between parent primaquine and cinnamic acids").')
para('The 652 pairs cover 131 unique drugs (about 80 that form cations and 51 that form anions) and 291 counter-ions '
     'used by the recommender. Some molecules appear twice (for example a drug that can be drawn as both a cation and an '
     'anion), so the 131 unique drugs give 138 drug entries, and the 291 ion entries are about 284 distinct molecules '
     'once different charge forms are merged. The model works on the entries, and all forms of one molecule are kept '
     'together when it is tested (Section 4.6). A drug is a base, which forms a cation, or an acid, which forms an anion, and an ion can '
     'pair only with a drug of the opposite charge. The data are skewed towards a few frequently studied drugs and ions: salicylic acid '
     'and ibuprofen alone account for 121 of the 652 pairs (19%), and the most-used counter-ions are choline (27 drugs) '
     'and docusate (26 drugs). A reported pair means the pair was made, not that it is the best pair for that drug, '
     'because laboratories tend to study convenient ions; this bias is carried into the labels and is discussed with the '
     'results.')
para('A separate search of patents was made afterwards to see how many pairs the literature search missed. Seven patents '
     'on API ionic liquids and drug salts were read (Section 5.7). They name 102 distinct drug–counter-ion pairs, of '
     'which 27 are already among the 652. The patent pairs were kept separate from the training data and used only as '
     'a check.')
figure(FIG + '/fig_data_pipeline.png', 'Figure 4. (a) How the dataset was built, from the literature search to the 131 drugs '
       '(80 that form cations, 51 that form anions) and 291 counter-ions (182 cations, 109 anions). (b) Number of reported pairs for each counter-ion, sorted by use: a few ions are '
       'used many times, and 173 of the 291 ions appear in only one pair.', 6.3)

heading('4.2 Assigning a pKₐ to each counter-ion', 2)
para('The pKₐ relationship between the drug and the ion is one of the score inputs, so every drug and every '
     'counter-ion needs a pKₐ.')
p = para(); cite(p, 'Drug pKₐ. The drug values were looked up, not predicted in this project. Table 1 lists where they '
     'come from. Of the 138 drug entries, 104 use a measured value: from the IUPAC digitized pKₐ dataset', '37',
     ', from the ChEMBL, DataWarrior and AvLiLuMoVe measured sets compiled by Baltruschat and Czodrowski');
_run(p, '[10]', sup=True); _run(p, ', from the PKKB drug database, from reference works and drug labels, or from the '
     'original papers. Seven more are measured values whose source does not say which ionisable group they belong to. '
     'For the remaining 27 no measured value could be found, and the value listed in a public database (HMDB, '
     'ContaminantDB or ChemicalBook) was used; these database values are themselves computed, not measured. The full '
     'list, with the source of every value, is in the project workbook.')
table_caption('Table 1. Where the drug pKₐ values come from (138 drug entries).')
simple_table(['Source', 'Entries', 'Type'],
             [['IUPAC digitized pKₐ dataset', '16', 'measured'],
              ['ChEMBL / DataWarrior / AvLiLuMoVe (Baltruschat and Czodrowski)', '18', 'measured'],
              ['PKKB drug database', '27', '25 measured, 2 site not stated'],
              ['Original papers (journal articles, theses, reports)', '25', '22 measured, 2 site not stated, 1 project sheet'],
              ['Reference works and drug labels (Merck-style monographs, FDA labels, PubChem)', '25', '22 measured, 3 site not stated'],
              ['Public databases listing a computed value (HMDB, ContaminantDB, ChemicalBook)', '27', 'computed']],
             widths=[9.5, 2.0, 5.0])
para('Counter-ion pKₐ. The counter-ion values come from three routes: measured values from the literature where they '
     'exist; no value for permanent ions; and, for the rest, a prediction with the open-source tool pKasso.')
para('A permanent ion is an ion whose charge does not come from gaining or losing a proton. Examples are quaternary '
     'ammonium ions, where the nitrogen carries four carbon groups (cholinium, tetrabutylammonium, benzalkonium), '
     'quaternary phosphonium ions, and N-alkylated aromatic cations such as 1-alkyl-3-methylimidazolium and '
     'alkylpyridinium. Such an ion has no acidic hydrogen on its charged atom, so it stays charged at every pH and has '
     'no pKₐ for that charge. Its salt with a drug is made by ion exchange (for example from choline hydroxide or '
     'bicarbonate), not by proton transfer, so the ΔpKₐ rule does not apply. In the model it gets a salt probability of 1 '
     'and a flag that marks it as permanent. 85 of the 291 counter-ions are permanent cations.')
p = para(); cite(p, 'Three other methods were tried first and set aside. A random forest (the "Machine learning meets '
         'pKₐ" model), which reaches a test R-squared of about 0.89 on general molecules, gave an R-squared of only '
         'about 0.50 on our ions', '10', '. A graph neural network with multi-instance learning (Graph-pKₐ), '
         're-trained on about 5,900 molecules because its weights are not public, gave an R-squared of about 0.27')
_run(p, '[11]', sup=True); _run(p, '. The machine-learning models were built for general, mostly single-site drug-like '
     'molecules, and too few of our ions have a measured pKₐ to re-train or fine-tune on, so their patterns do not carry over.')
p = para(); cite(p, 'Calibrated DFT (B3LYP/def2-SVPD with SMD implicit water, followed by a linear fit to measured values)',
     '12', ' was accurate for acids (80 ions, mean absolute error 0.79, R-squared 0.88) but weaker for bases (64 ions, '
     'mean absolute error 1.29, R-squared 0.20). A breakdown by chemical family shows why. For the plain aliphatic '
     'ammonium ions the error was under one unit, but a few families were far off: an amino-acid-type ion with a carboxylate '
     'and an amine (ampicillinate, off by 5.7 units), an aniline-type base (benzocainium, 5.6), a guanidinium '
     '(tetramethylguanidinium, 3.1) and an imidazolium (1.8). One calibration line cannot fit all of these at once. This '
     'is a known weakness of implicit-solvent models for charged nitrogen groups: a protonated N–H⁺ group forms strong '
     'hydrogen bonds with the surrounding water, which a continuum model does not describe, and the SMD model itself has '
     'errors of about 4 kcal/mol for ions against 0.6 to 1.0 kcal/mol for neutral molecules');
_run(p, '[30]', sup=True); _run(p, '. Since 1.36 kcal/mol corresponds to one pKₐ unit at 25 °C, an error of that size is '
     'about three pKₐ units. A recent study of amino acids found the same pattern, with errors for amine groups about '
     'twice those for carboxylic acids and about four units for the lysine, arginine and histidine side chains');
_run(p, '[32]', sup=True); _run(p, ', and reviews note that the direct DFT route is generally not reliable without '
     'explicit water or a reference reaction'); _run(p, '[31]', sup=True)
_run(p, '. Most of our acids, in contrast, are carboxylic or sulfonic acids, whose error is similar from one molecule to '
     'the next and is absorbed by the calibration line. The R-squared for bases is also low because most measured base '
     'pKₐ values lie in a narrow band (about 7 to 11), so even a one-unit error leaves little of the spread explained. On '
     'top of this, DFT is slow and computationally expensive, because every ion needs a full quantum-chemistry '
     'calculation over several conformers.')
p = para(); cite(p, 'The chosen method is pKasso, an open-source tool that predicts the overall (macro) pKₐ of a '
         'molecule by building its possible protonation states and combining a per-site (micro) pKₐ for each', '25',
         '. It was chosen for three reasons that match this problem. First, it handles ions with several ionizable sites, '
         'which is exactly where the random forest (single-site only) and the graph model struggled; many of our '
         'counter-ions are multi-site. Second, it predicts acids and bases equally well, avoiding the weakness of DFT for '
         'bases. Third, it is a fast Python tool (MIT licence) that runs from a SMILES string, so it '
         'needs no quantum-chemistry calculation per ion.')
para('On the counter-ions that have a measured pKₐ (138 ions), pKasso gave a mean absolute error of 0.50, a mean '
     'squared error of 0.92 (a root-mean-square error of about 0.96) and an R-squared of 0.93 — better than DFT '
     'overall and across both acids and bases, and far faster. The mean absolute error of 0.50 means half the ions are '
     'predicted within about half a pKₐ unit; the larger squared error of 0.92 comes from a few ions with bigger '
     'misses rather than from a broad spread, which is expected for a dataset that mixes simple and awkward multi-site '
     'ions. This level of accuracy is more than enough here, because pKₐ enters the score only as a soft supporting '
     'input and ΔpKₐ is shown as a label rather than used as a filter: an occasional pKₐ error shifts a '
     'score slightly but never removes a candidate or drives the ranking, which is carried mostly by structural '
     'similarity.')

heading('4.3 Features', 2)
para('Each drug-ion pair is described by four groups of numbers:')
bullet('how similar the drug is to known drugs that already use this ion, computed with Morgan fingerprints and the '
       'Tanimoto coefficient. This is the "ions used by similar drugs" idea.', lead='2D similarity — ')
bullet('the same idea using thirteen RDKit three-dimensional shape descriptors, averaged over five conformers.', lead='3D similarity — ')
bullet('the ΔpKₐ between the drug and the ion, the salt probability from the Brittain and Cruz-Cabeza rules, '
       'and two flags for a missing pKₐ and for a permanent cation.', lead='pKₐ — ')
bullet('how well the electrostatic surfaces of the two ions fit, using GFN2-xTB charges.', lead='ESP complementarity — ')
para('Both a two-dimensional and a three-dimensional similarity are used because they answer different questions. The '
     '2D fingerprint compares functional groups and connectivity, which drive hydrogen bonding and the acid-base '
     'chemistry of salt formation; the 3D descriptors compare overall size and shape. A benchmark of seven fingerprints '
     'and four similarity measures against 3D shape descriptors (USR and USRCAT) showed that 2D fingerprints are clearly '
     'the stronger signal for this task (top-5 accuracy about 70% for MACCS with Tanimoto, against about 54 to 60% for '
     'the 3D measures), but 3D shape adds a small and independent contribution, so both are kept.')
para('The pKₐ group is used only as supporting information, never as a filter. The ΔpKₐ and the salt '
     'probability are shown next to every suggestion as a salt, ambiguous, or co-crystal label, and they enter the score '
     'as one gentle input among the four. They never remove, veto or reject a candidate. This is deliberate, because the '
     'dataset contains reported pairs across the whole ΔpKₐ range, including the ambiguous and co-crystal zones, '
     'so treating pKₐ as a hard cut-off would wrongly discard real pairs. The similarity features only ever use the '
     'training drugs, so a drug is never compared with itself.')
para('The 3D group. Five conformers of each drug are generated with RDKit (ETKDG), and the thirteen shape descriptors in '
     'Table 2 are averaged over them. Skewed descriptors are log-transformed and all are z-scored; the similarity of two '
     'drugs is 1/(1 + d/√13), where d is the Euclidean distance between their descriptor vectors.')
table_caption('Table 2. The thirteen 3D shape descriptors (all computed with RDKit).')
simple_table(['#', 'Descriptor', 'What it measures'],
             [['1–3', 'PMI1, PMI2, PMI3', 'principal moments of inertia (mass spread along the three main axes)'],
              ['4–5', 'NPR1, NPR2', 'normalised PMI ratios: rod-like, disc-like or sphere-like shape'],
              ['6', 'Radius of gyration', 'overall size around the centre of mass'],
              ['7', 'Asphericity', 'how far the shape is from a sphere'],
              ['8', 'Eccentricity', 'how elongated the shape is'],
              ['9', 'Inertial shape factor', 'flatness, from the moments of inertia'],
              ['10', 'Spherocity index', 'how round the molecule is'],
              ['11', 'PBF (plane of best fit)', 'how far the atoms are from one flat plane'],
              ['12', 'Maximum extent', 'largest distance between two atoms'],
              ['13', 'van der Waals volume proxy', 'molecular volume']],
             widths=[1.2, 4.6, 10.0])
para('The electrostatic group. Ion pairs are held together mainly by electrostatic attraction, so an electrostatic '
     'feature was expected to help, and several were tried in turn. A dipole moment (Gasteiger charges) was tried first, '
     'as a filter in an earlier version of the recommender: it removed real partners and lowered top-5 accuracy from '
     '70.2% to 61.1%, and on its own it ranked partners little better than random (Hit@10 0.26 against 0.23). The computed interaction energy of the drug–ion pair '
     '(GFN2-xTB) was tried next and did not separate real pairs from other pairs (area under the curve 0.47, interval '
     '0.41 to 0.53). Six electronic descriptors from GFN2-xTB were then each tested as the electrostatic group in the '
     'final equation, with the same folds (Table 3). Only ESP complementarity gave a gain whose interval excludes zero: '
     'Hit@1 rose from 0.268 to 0.326 and MRR from 0.419 to 0.449, while Hit@10 changed little. It was therefore chosen. '
     'ESP complementarity is the most positive electrostatic potential on the surface of the cation minus the most '
     'negative on the anion, a measure of how strongly the most charged sites of the two ions can attract each other.')
table_caption('Table 3. Choosing the electrostatic group. Base: 2D + 3D + pKₐ. Each row adds one descriptor (GFN2-xTB); gains are '
              'against the base, with 95% bootstrap intervals over drugs.')
simple_table(['Descriptor added', 'Hit@1', 'Hit@10', 'MRR', 'Hit@1 gain (95% CI)'],
             [['None (base)', '0.268', '0.710', '0.419', '–'],
              ['ESP complementarity (chosen)', '0.326', '0.717', '0.449', '+0.058 (0.014 to 0.105)'],
              ['Electronegativity gap', '0.293', '0.721', '0.432', '+0.025 (−0.014 to 0.065)'],
              ['Charge spread (sum)', '0.297', '0.717', '0.431', '+0.029 (−0.018 to 0.076)'],
              ['Chemical hardness (sum)', '0.283', '0.699', '0.416', '+0.014 (−0.029 to 0.054)'],
              ['Polarisability gap', '0.283', '0.699', '0.419', '+0.014 (−0.014 to 0.047)'],
              ['Dipole moment (ion + gap)', '0.275', '0.681', '0.413', '+0.007 (−0.040 to 0.051)']],
             widths=[5.6, 1.5, 1.9, 1.5, 4.8])

heading('4.4 Score equation and calibration', 2)
para('The four groups are combined in one additive logistic equation, where each f is a smooth spline curve (five '
     'knots), so a feature can help more in some ranges than in others:')
eq = doc.add_paragraph(); eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
_run(eq, 'logit(P) = b₀ + f', italic=True); _run(eq, '2D'); _run(eq, ' + f', italic=True); _run(eq, '3D')
_run(eq, ' + f', italic=True); _run(eq, 'pKₐ'); _run(eq, ' + f', italic=True); _run(eq, 'ESP')
p = para(); cite(p, 'The model is a logistic regression. In each training fold there are about 13,200 candidate rows, '
     'of which only about 364 (under 3%) are reported pairs. All reported pairs are kept and the unreported ones are '
     'sampled at random so that the training table has 8,000 rows, about one reported pair to 21 others. The limit was '
     'set so that the slower comparison models (the support vector machine and the neural network) could be trained ten '
     'times each on exactly the same rows as the equation, which keeps the comparison fair. Sampling the negatives at '
     'random changes only the intercept of a logistic regression, not the order of the candidates', '33',
     '. A check confirmed that the number of rows does not matter: 4,000, 8,000, 12,000 and all rows gave Hit@10 of '
     '0.725, 0.721, 0.728 and 0.714, all within each other’s intervals. The raw probability is slightly too high '
     'because of the sampling, so it is corrected with Platt calibration using two parameters fitted on held-out pairs.')
para('The reported score is 100 times the calibrated probability, so a score of 25 means about a 25% '
     'chance of being a reported pair; the average candidate scores about 2.8. The spline setting was fixed before the '
     'results were seen. Because the equation is additive, the score splits cleanly into one contribution per group.')

heading('4.5 Input, output and a worked example', 2)
para('The model does not take a list of drug descriptors as input. Its input is a small table with one row for every '
     'candidate pair, that is, the drug together with one possible counter-ion. A drug that is an acid is paired with every '
     'cation in the data, and a drug that is a base is paired with every anion, which gives about 100 to 180 rows per drug. '
     'Each row holds seven numbers: sim2D, sim3D, ΔpKₐ, the salt probability, the two pKₐ flags and the ESP '
     'complementarity. The label of a row is 1 if the pair is a published pair and 0 if it is not. The output is a '
     'score for every row, and the counter-ions of a drug are ranked by this score.')
para('The 2D and 3D similarity of a row are measured to the training drugs that already use that ion. The drug is compared '
     'with each of them, and the highest similarity is kept. The thirteen shape descriptors are therefore not given to the '
     'model directly; they are only used to compute the one number sim3D. Figure 5 shows this for one pair. The steps are:')
figure(FIG + '/fig_similarity_example.png', 'Figure 5. How sim2D and sim3D are found for the pair ibuprofen and lidocainium. '
       '(a) The query drug and the candidate ion. (b) The training drugs reported with lidocainium, with their 2D and 3D '
       'similarity to ibuprofen. (c) The highest values become the pair features.', 6.3)
numbered(1, 'List all candidate counter-ions of the opposite charge for the drug.', lead='Candidates. ')
numbered(2, 'For each candidate, compute the seven numbers using only the training drugs.', lead='Features. ')
numbered(3, 'Pass the seven numbers through the fitted equation and the calibration to get the score.', lead='Score. ')
numbered(4, 'Sort the candidates by score and read the top ten as the suggestions.', lead='Rank. ')
para('Table 4 shows this for ibuprofen, an acid, treated as an unseen drug. It has 158 candidate cations, and four of the '
     'rows are shown. Lidocainium, for example, is used by loxoprofenate and ketoprofen, the drugs most similar to ibuprofen, '
     'so its sim2D is 0.49 and its sim3D is 0.75. Its ΔpKₐ is 4.80, which falls in the salt zone, so it scores 25.4 and '
     'ranks 4 of 158. Prochlorazium has low similarity and a negative ΔpKₐ, so it scores 0.95 and ranks 156. Cholinium '
     'and tetraethylammonium are permanent cations, so no pKₐ is needed for them.')
table_caption('Table 4. Four of the 158 candidate rows for ibuprofen (acid), with the model score and rank. Ibuprofen was '
              'held out, so only the other drugs were used to compute the numbers. * Permanent cation, no pKₐ needed.')
simple_table(['Counter-ion', 'sim2D', 'sim3D', 'ΔpKₐ', 'ESP', 'Reported', 'Score', 'Rank'],
             [['Lidocainium', '0.49', '0.75', '4.80', '311.6', 'No', '25.4', '4'],
              ['Cholinium', '0.42', '0.77', 'none*', '313.2', 'Yes', '23.3', '5'],
              ['Tetraethylammonium', '0.17', '0.58', 'none*', '298.2', 'Yes', '4.3', '70'],
              ['Prochlorazium', '0.19', '0.36', '−0.77', '311.9', 'No', '0.95', '156']],
             widths=[4.0, 1.5, 1.5, 1.5, 1.5, 2.0, 1.5, 1.3])
para('Ibuprofen has 27 reported counter-ions in the data. Ranked as an unseen drug, the first of them (matrinium) comes at '
     'rank 3 and the next ones at ranks 5, 9, 11 and 12. This is the check that Hit@10 counts: whether a reported ion falls '
     'in the top ten. The ions ranked above a reported one, such as lidocainium here, are not wrong; they are the model’s '
     'new suggestions, because the data only says which pairs were published and never which ones fail.')

heading('4.6 Evaluation and leakage control', 2)
para('The model is tested leave-drug-out: each drug is held out in turn, the model trains on the rest, then ranks the '
     'held-out drug’s counter-ions. This is repeated over five folds and two random seeds (Figure 6). Two data leaks were found '
     'and removed:')
numbered(1, 'A few drugs appear as both a cation and an anion form. All forms of one molecule are kept in the same fold, '
         'so a drug cannot see itself during training.', lead='Same molecule in two forms. ')
numbered(2, 'Some counter-ions are reported with only one drug. When that drug is held out, no training drug uses the '
         'ion, so the model has no evidence for it. If such an ion were left in the candidate list, the fact that it has '
         'no other user would itself reveal that it belongs to the held-out drug, which is a leak. These ions are '
         'therefore removed from the candidate list, and a true pair that uses one of them is counted as a miss. For '
         'example, cinnarizine is the only drug reported with 2-ethyl-1-hexyl sulfate and four other alkyl sulfates; '
         'when cinnarizine is held out, those five pairs cannot be found and count as misses. In total 173 of the 652 '
         'pairs are of this kind, one for each of the 173 ions used only once.', lead='Ions with no training user. ')
figure(FIG + '/fig_leave_drug_out.png', 'Figure 6. Leave-drug-out testing with five folds. In each fold one group of drugs is '
       'held out, the model is fitted on the other four, and the held-out drugs are ranked.', 6.0)
para('Performance is measured with Hit@1, Hit@5 and Hit@10 (whether a true counter-ion is in the top one, five or ten), '
     'the mean reciprocal rank (MRR), and a paired bootstrap over drugs for confidence intervals.')
para('Not every reported pair can be found by a similarity method. Of the 652 pairs, 479 (73%) use an ion that at least one '
     'other drug also uses and can therefore be recovered, and 132 of the 138 drug entries have at least one such pair. '
     'The ceiling on Hit@10 is therefore about 96% of drugs, so the unreachable pairs are not what keeps Hit@10 at '
     '0.72 to 0.74.')

# ================= 5. RESULTS =================
heading('5. Results and Discussion')

heading('5.1 Comparison of models', 2)
para('The models were run on the same four feature groups with the same leave-drug-out protocol, against a random '
     'baseline. The additive spline equation and ordinary logistic regression are the most accurate; the boosted-tree '
     'and neural-network models are below them. Figure 7 and Table 5 summarise the results. Every model received exactly the same seven inputs and the same training rows.')
figure(FIG + '/fig_model_comparison.png', 'Figure 7. Hit@10 for each model under the leave-drug-out protocol, with 95% '
       'confidence intervals. The random baseline is shown for reference.', 5.8)
table_caption('Table 5. Ranking accuracy of the final score and of the comparison models and baselines.')
simple_table(['Method', 'Hit@1', 'Hit@5', 'Hit@10 (95% CI)', 'MRR'],
             [['Final score (spline equation)', '0.326', '0.612', '0.717 (0.649–0.790)', '0.449'],
              ['Logistic regression', '0.337', '0.645', '0.728 (0.656–0.801)', '0.472'],
              ['XGBoost', '0.239', '0.504', '0.641 (0.565–0.714)', '0.366'],
              ['Extra trees', '0.257', '0.525', '0.634 (0.565–0.710)', '0.386'],
              ['LightGBM', '0.228', '0.486', '0.630 (0.562–0.699)', '0.357'],
              ['2D similarity only (kNN)', '0.243', '0.547', '0.652 (0.580–0.725)', '0.378'],
              ['Random baseline', '0.043', '0.123', '0.225 (0.178–0.275)', '0.109']],
             widths=[5.4, 1.6, 1.6, 4.4, 1.5])
para('The final score finds a reported counter-ion in the top ten for about 72% of unseen drugs, three times better than '
     'random, and beats the boosted-tree models by about 0.08 to 0.09 in Hit@10. Plain logistic regression scores slightly '
     'higher (0.728 against 0.717), but the difference is far inside the confidence interval, so the two are equally '
     'accurate here. The spline form was kept because it does not force each feature to act as a straight line, and its '
     'curves can be drawn and read (Figure 8); the score also splits into one contribution per feature group.')
heading('5.2 The score equation', 2)
para('The four spline curves show how each feature pushes the score. Two-dimensional similarity has by far the strongest '
     'effect; three-dimensional similarity adds a smaller push; the pKₐ and ESP curves are close to flat over most '
     'of their range. These curves are shown in Figure 8.')
figure(FIG + '/fig_partial_effects.png', 'Figure 8. The learned spline curves for the four feature groups (partial '
       'effects on the score).', 5.8)

heading('5.3 Contribution of each feature group', 2)
para('The contribution of a group is measured by removing it and seeing how much Hit@10 drops. Two-dimensional '
     'similarity accounts for about 71% of the total drop, three-dimensional similarity about 20%, and pKₐ and ESP '
     'about 6% and 4%. The confidence intervals for pKₐ and ESP include zero, so those two groups are not clearly '
     'useful on their own; most of the information is structural similarity. Table 6 gives the numbers and Figure 9 '
     'shows them.')
table_caption('Table 6. Drop in Hit@10 when each feature group is removed (paired bootstrap over drugs).')
simple_table(['Group removed', 'Hit@10 without it', 'Drop (95% CI)', 'Share of total drop'],
             [['2D similarity', '0.576', '0.141 (0.080–0.210)', '71%'],
              ['3D similarity', '0.678', '0.040 (0.000–0.083)', '20%'],
              ['pKₐ', '0.707', '0.011 (−0.025–0.047)', '6%'],
              ['ESP complementarity', '0.710', '0.007 (−0.014–0.025)', '4%']])
figure(FIG + '/fig_contribution.png', 'Figure 9. Share of the Hit@10 drop attributable to each feature group.', 5.2)

para('Hit@10 is not where these two groups act. ESP mainly lifts the true counter-ion to the very top of the list: '
     'adding it raised Hit@1 from 0.268 to 0.326, with an interval that excludes zero (Table 3). The pKₐ group is kept '
     'because it carries the established chemistry of salt formation (the ΔpKₐ rule), removing it did not improve the '
     'ranking, and the pKₐ label shown next to every suggestion lets the user see the acid-base relationship. Its small '
     'contribution to Hit@10 is reported as it is and is not claimed as a gain.')

heading('5.4 Calibration', 2)
para('The score is calibrated so it can be read as a probability. After Platt calibration, the predicted probability '
     'matches the observed rate of reported pairs across the score range, as shown in Figure 10.')
figure(FIG + '/fig_calibration.png', 'Figure 10. Calibration of the score: predicted probability against the observed '
       'fraction of reported pairs.', 4.8)

heading('5.5 Ranking for all drugs and for a new drug', 2)
para('When each of the drugs is ranked in turn with all the others as training, the score reaches Hit@10 0.739 and MRR '
     '0.471. Figure 11 shows two worked examples. The same equation ranks counter-ions for a drug that is not in the '
     'dataset: the user gives the drug’s charged SMILES and an optional measured pKₐ, every candidate ion is '
     'scored, and a ranked list is returned. For a new drug the two similarity groups are computed from its structure, '
     'the ESP group uses the dataset median, and the pKₐ group is used only when a measured value is supplied.')
figure(FIG + '/fig_example_top10.png', 'Figure 11. Example top-ten counter-ions for two drugs, with the four score '
       'contributions shown for each ion.', 5.8)

heading('5.6 Counter-ion suggestions and safety', 2)
para('For any drug the ranked list mixes counter-ions already published for that drug with new suggestions. A new '
     'suggestion is a pair not present in the dataset. Each suggested ion carries a safety tag taken from the FDA '
     'GRAS/SCOGS list and a pharmaceutical-acceptability class (established salt former, biocompatible, dual-active, '
     'research-only, or not classified). The tag is shown as a label next to each suggestion and is not part of the score.')
para('Whether safety should enter the score was tested directly. Two safety numbers were made for every ion: a 1/0 flag '
     'for whether the ion (or its parent acid or base) is on the FDA GRAS or SCOGS lists, and a 1/0 flag for whether it '
     'belongs to an accepted class (established pharmaceutical salt former, excipient or food-derived, or a drug ion). '
     'Each flag, and both together, was added to the equation as an extra input and tested with the same folds. Hit@10 '
     'stayed between 0.710 and 0.725, against 0.717 without them, and no change in Hit@1, Hit@10 or MRR had an interval '
     'that excludes zero. Safety also does not describe which pairs exist: only about 39% of the reported pairs use a '
     'clearly safe ion, and about 10% use a research-only ion. Safety is therefore shown to the user but not added to the score.')

heading('5.7 Do the new suggestions hold up?', 2)
para('Hit@k only tests whether reported pairs are recovered. It does not test whether a pair that is not in the dataset is a '
     'good one, because the data records which pairs were published and never which ones fail. The right test for new '
     'suggestions is a time split, training on older papers and testing on pairs reported later; this has not been done.')
para('As a partial check, five suggestions that were not in the dataset were looked up in the literature after the model '
     'was built (Table 7). They are examples picked from a longer list of suggestions, not a measured success rate. Three '
     'are in the top ten of their drug and two are not. That all five exist in the literature also shows that the '
     'extraction of reported pairs is not complete, so a suggestion that is not in the dataset is not necessarily new.')
table_caption('Table 7. Suggested pairs that were not in the dataset but were later found in the literature.')
simple_table(['Pair', 'Rank', 'Score', 'ΔpKₐ label', 'What the literature reports'],
             [['Ibuprofen + lidocainium', '4 of 158', '25.4', 'salt likely (4.8)', 'Lidocaine ibuprofenate ionic liquid; some report a eutectic mixture [23]'],
              ['Valproate + cholinium', '1 of 182', '11.0', 'permanent cation', 'Valproate ionic liquids with choline, conference abstract [27]'],
              ['Cinnarizine + tosylate', '9 of 104', '5.4', 'salt likely (10.1)', 'Crystal salt with two tosylates per drug [26]'],
              ['Albendazole + tartrate', '28 of 109', '1.5', 'ambiguous (0.06)', 'D-tartrate salt, about 585 times more soluble at pH 2 [28]'],
              ['Propranolol + oxalate', '56 of 107', '3.6', 'salt likely (7.9)', 'Oxalate salt, more soluble than the base [29]']],
             widths=[3.9, 2.0, 1.5, 3.0, 5.2])
para('The albendazole case is also a reminder of why ΔpKₐ is a label and not a filter: its ΔpKₐ falls in the ambiguous zone, '
     'yet the salt exists and is far more soluble.')
para('A second, larger check used patents. Seven patents were read (Table 8): two MEDRx patents on NSAID ionic liquids '
     'for skin patches, one on NSAID–amine salts, one on a lidocaine patch, two from the Rogers group on API ionic '
     'liquids, and one on acyclovir salts. Together they name 102 distinct drug–counter-ion pairs; 27 are already in the dataset and 75 are '
     'new. Of the new pairs, 31 use a drug and an ion that the model can score, and their ranks were read from the '
     'leave-drug-out ranking.')
table_caption('Table 8. Patents read and the pairs they name. A pair named in two patents is counted in both rows.')
simple_table(['Patent', 'Topic', 'Pairs named', 'Already in dataset'],
             [['US8623387B2 (MEDRx)', 'fatty-acid and NSAID ionic liquids for external use', '84', '15'],
              ['AU2008334222B2 (MEDRx)', 'etodolac ionic liquid tape', '4', '1'],
              ['US20100029704A1', 'NSAID salts with organic amines', '4', '2'],
              ['WO2020218249A1', 'lidocaine-containing patch', '3', '1'],
              ['US20160002240A1', 'nucleoside-analogue (acyclovir) salts', '6', '6'],
              ['US8232265B2 / CA2625004C (Rogers et al.)', 'multi-functional ionic liquids', '3', '2'],
              ['WO2010078300A1 / US9278134B2', 'dual-functioning ionic liquids', '2', '1']],
             widths=[5.2, 6.2, 2.2, 2.6])
para('The result is mixed. Two of the 31 patent pairs are in the top ten of their drug (6%), about what chance gives '
     '(7%); ibuprofen with diisopropanolamine is ranked first. Over the whole list the patent pairs sit higher than '
     'chance: their median rank is 55 of about 180 candidates, the average pair is at the 36th percentile (95% interval '
     '29 to 44, against 50 for chance), and 68% are in the top half. So the model places these pairs in roughly the top '
     'third, but not in the top ten. Two things limit what this check shows. Most of the patent pairs come from one '
     'company’s systematic grid of six NSAIDs crossed with fourteen amines, and over half of the grid is only listed as '
     'possible ionic liquids, not shown as made. The check is also a reminder that the model learns which pairs '
     'researchers have chosen to publish, and a patent grid is chosen differently.')

heading('5.8 Properties not predicted: solubility, toxicity and bioavailability', 2)
para('The recommender does not predict the solubility, toxicity or bioavailability of the API-ILs. The reason is a lack '
     'of data, not of method, and it comes from the chemistry of the pair. An API-IL is not a mixture of two independent '
     'ions: the drug and the counter-ion are held together by proton transfer (which may be incomplete), by hydrogen '
     'bonds and by electrostatic attraction, and in water they may stay partly paired. Its solubility and toxicity '
     'therefore belong to the bonded pair and cannot be added up from the two ions. Values for single ions are often '
     'available (a measured water solubility for 135 of the 291 counter-ions and an oral LD50 for 46), but they give '
     'every partner of a drug the same value.')
p = para(); cite(p, 'The literature shows how much the partner matters. Naproxenate is about four times more soluble with '
     'cholinium than with sodium', '36', ', and its cytotoxicity to L929 cells changes about seven-fold with the cation, '
     'from an IC50 of 111 µM with benzalkonium to 783 µM with cholinium'); _run(p, '[34]', sup=True)
_run(p, '. Measured values for the pairs themselves are scarce. A targeted search for pairs in this dataset found numeric '
     'solubility or toxicity values for only seven API-ILs, in three papers'); _run(p, '[34, 35, 36]', sup=True)
_run(p, ' (Table 9; all values are in the project workbook). They are also not comparable with each other: solubility is '
     'given in mol/L, in mM or only as a fold increase, in water or in buffer, at 25 or 37 °C, and toxicity comes from '
     'three different assays (an MTT test on mouse L929 cells, an XTT test on human Caco-2 cells, and an enzyme '
     'inhibition test). This is far too little, and too mixed, to train or validate a model for a two-ion system; '
     'bioavailability has essentially no quantitative API-IL records at all.')
table_caption('Table 9. Numeric solubility and toxicity values found for API-ILs in this dataset.')
simple_table(['API-IL', 'Water solubility', 'Toxicity', 'Ref.'],
             [['Cholinium naproxenate', '3.37 mol/L (25 °C)', 'IC50 783 µM (L929, MTT)', '[34]'],
              ['Benzalkonium naproxenate', '1.10 mol/L (25 °C)', 'IC50 111 µM (L929, MTT)', '[34]'],
              ['1-Octyl-3-methylimidazolium naproxenate', '2.65 mol/L (25 °C)', 'IC50 174 µM (L929, MTT)', '[34]'],
              ['Cholinium ibuprofenate', '8,111 mM (37 °C)*', 'EC50 > 5.5 mM (Caco-2, XTT)', '[35]'],
              ['1-Ethyl-3-methylimidazolium ibuprofenate', 'fold increase only', 'EC50 > 5.5 mM (Caco-2, XTT)', '[35]'],
              ['1-(2-Hydroxyethyl)-3-methylimidazolium ibuprofenate', '9,845 mM*', 'EC50 > 5.5 mM (Caco-2, XTT)', '[35]'],
              ['Cholinium ketoprofenate', '58-fold above ketoprofen', 'enzyme IC50 21.3 mM', '[36]']],
             widths=[6.2, 3.6, 4.6, 1.4])
para('* Value quoted in [35] from earlier work, not measured in that study.')
p = para(); cite(p, 'Drug-level models do exist for related properties — a linear intestinal-permeability equation', '20',
         ' and bioavailability classifiers'); _run(p, '[21]', sup=True); _run(p, ' — but they cannot rank '
     'counter-ions. They take only the drug as input, so every counter-ion of a drug receives the same score; they were '
     'fitted on single neutral drugs and exclude salts and charged species; and none has been validated on ionic '
     'liquids. For this reason the recommender ranks on structure and the pKₐ relationship, and shows measured '
     'solubility, toxicity or safety values only as labels when they exist.')

heading('5.9 What else did not improve the model', 2)
para('Several additional features were tested with the same protocol and did not improve the ranking; they are recorded '
     'here so the limits of the method are clear.')
bullet('Dipole moment, and the few measured solubility and toxicity values available at the ion level, added to the '
       'score gave no improvement (Hit@10 changed from 0.717 to 0.699, 0.696 and 0.717, and every change interval '
       'included zero). A solubility or toxicity value belongs to the ion alone, so it gives every drug the same '
       'ranking and cannot say which ion suits which drug.')
bullet('The two safety flags added to the score gave no gain (Hit@10 between 0.710 and 0.725; Section 5.6).')
bullet('Other electronic descriptors in place of ESP gave no clear gain (Table 3).')
bullet('An older model that used two-dimensional descriptors of both the drug and the ion scored higher (0.775), but '
       'with the ion descriptors alone, which know nothing about the drug and give every drug the same list, it still '
       'reached 0.736. Its higher score therefore did not come from matching the ion to the drug, and it was not kept.')
bullet('Training on ionic-liquid–ionic-liquid pairs did not transfer to the API pairs, so it was dropped.')

# ================= 6. SUMMARY =================
heading('6. Summary')
para('This project built a recommender that, given a drug, ranks the counter-ions most likely to form a reported pair '
     'with it. The pairs were collected from about 4,600 articles; the counter-ion pKₐ values were measured, taken '
     'as not needed for permanent ions, or predicted with the open-source pKasso tool, chosen after density-functional '
     'theory proved slow, expensive and weak for bases, and two machine-learning predictors proved less accurate. The score is one interpretable '
     'additive equation over four feature groups. It places a reported '
     'counter-ion in the top ten for about 72% of unseen drugs, three times better than random, and matches the best of '
     'eight machine-learning models. Most of the signal is structural similarity to drugs that already have a reported '
     'counter-ion; ESP mainly lifts the true ion to the top of the list and pKₐ adds little. The same equation works for a new drug and is delivered as a '
     'web application and a notebook, with a safety label on each suggestion.')
para('The main limitations follow from the data. The labels are reported pairs, not proven good pairs, so a pair nobody '
     'has reported is not a failed pair. About a quarter of the reported pairs use an ion that no other drug uses, so a '
     'similarity method can never suggest those ions. Solubility, toxicity and bioavailability of the API-ILs cannot yet '
     'be predicted for lack of data. The results come from a single dataset with no external test set, and because the '
     'feature groups and the equation form were chosen on the same data, the Hit@10 figure is probably a little '
     'optimistic. The new suggestions have been checked only retrospectively: five suggested pairs were later found in '
     'the literature, and pairs from seven patents were ranked in about the top third of their lists but rarely in the '
     'top ten. A time-split test has not been done.')

# ================= REFERENCES =================
heading('References')
refs = [
 'Rogers, R. D., & Seddon, K. R. (2003). Ionic liquids—solvents of the future? Science, 302(5646), 792–793.',
 'Hough, W. L., Smiglak, M., Rodríguez, H., Swatloski, R. P., Spear, S. K., Daly, D. T., Pernak, J., Grisel, J. E., Carliss, R. D., Soutullo, M. D., Davis, J. H., & Rogers, R. D. (2007). The third evolution of ionic liquids: active pharmaceutical ingredients. New Journal of Chemistry, 31(8), 1429–1436.',
 'Sintra, T. E., Shimizu, K., Ventura, S. P. M., Shimizu, S., Canongia Lopes, J. N., & Coutinho, J. A. P. (2018). Enhanced dissolution of ibuprofen using ionic liquids as catanionic hydrotropes. Physical Chemistry Chemical Physics, 20(3), 2094–2103. (Cholinium-based ionic liquids give the largest solubility gains for ibuprofen.)',
 'Shamshina, J. L., & Rogers, R. D. (2023). Ionic liquids: new forms of active pharmaceutical ingredients with unique, tunable properties. Chemical Reviews, 123(20), 11894–11953.',
 'Cruz-Cabeza, A. J. (2012). Acid–base crystalline complexes and the pKₐ rule. CrystEngComm, 14(20), 6362–6365.',
 'Brittain, H. G. (2007). Strategy for the prediction and selection of drug substance salt forms. Pharmaceutical Technology, 31(10).',
 'Yoshizawa, M., Xu, W., & Angell, C. A. (2003). Ionic liquids by proton transfer: vapor pressure, conductivity, and the relevance of ΔpKₐ from aqueous solutions. Journal of the American Chemical Society, 125(50), 15411–15419.',
 'Krossing, I., Slattery, J. M., Daguenet, C., Dyson, P. J., Oleinikova, A., & Weingärtner, H. (2006). Why are ionic liquids liquid? A simple explanation based on lattice and solvation energies. Journal of the American Chemical Society, 128(41), 13427–13434.',
 'Stoimenovski, J., Izgorodina, E. I., & MacFarlane, D. R. (2010). Ionicity and proton transfer in protic ionic liquids. Physical Chemistry Chemical Physics, 12(35), 10341–10347.',
 'Baltruschat, M., & Czodrowski, P. (2020). Machine learning meets pKₐ. F1000Research, 9, 113. DOI 10.12688/f1000research.22090.2.',
 'Xiong, J., Li, Z., Wang, G., Fu, Z., Zhong, F., Xu, T., Liu, X., Huang, Z., Liu, X., Chen, K., Jiang, H., & Zheng, M. (2022). Multi-instance learning of graph neural networks for aqueous pKₐ prediction. Bioinformatics, 38(3), 792–798.',
 'Sanchez, A. J., & Raghavachari, K. (2023). Development and assessment of a cheminformatics model for accurate pKₐ prediction in aqueous medium. Theoretical Chemistry Accounts, 142, 86.',
 'Rogers, D., & Hahn, M. (2010). Extended-connectivity fingerprints. Journal of Chemical Information and Modeling, 50(5), 742–754.',
 'Landrum, G. RDKit: Open-source cheminformatics. https://www.rdkit.org.',
 'Pedregosa, F., et al. (2011). Scikit-learn: machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.',
 'Bannwarth, C., Ehlert, S., & Grimme, S. (2019). GFN2-xTB—an accurate and broadly parametrized self-consistent tight-binding quantum chemical method. Journal of Chemical Theory and Computation, 15(3), 1652–1671.',
 'Johnson, M. A., & Maggiora, G. M. (Eds.). (1990). Concepts and Applications of Molecular Similarity. Wiley, New York.',
 'Maggiora, G., Vogt, M., Stumpfe, D., & Bajorath, J. (2014). Molecular similarity in medicinal chemistry. Journal of Medicinal Chemistry, 57(8), 3186–3204.',
 'Sethi, S., Sundmacher, K., & Ganzer, C. (2025). A neural recommender system leveraging transfer learning for property prediction of ionic liquids. arXiv:2509.10273.',
 'Zhao, Y. H., Le, J., Abraham, M. H., et al. (2001). Evaluation of human intestinal absorption data and subsequent derivation of a QSAR with the Abraham descriptors. Journal of Pharmaceutical Sciences, 90(6), 749–784.',
 'Beheshti, A., et al. (2020). Machine-learning models relating animal and human oral bioavailability. (As cited in project background.)',
 'Can, E., Jalal, A., Zirhlioglu, I. G., Uzun, A., & Yildirim, R. (2021). Predicting water solubility in ionic liquids using machine learning towards design of hydro-philic/phobic ionic liquids. Journal of Molecular Liquids, 332, 115848.',
 'Panić, J., Tot, A., Janković, N., Drid, P., Gadžurić, S., & Vraneš, M. (2020). Physicochemical and structural properties of lidocaine-based ionic liquids with anti-inflammatory anions. RSC Advances, 10(24), 14089–14098.',
 'Kalepu, S., & Nekkanti, V. (2015). Insoluble drug delivery strategies: review of recent advances and business prospects. Acta Pharmaceutica Sinica B, 5(5), 442–453.',
 'pKasso: an open-source tool for protonation-state and macro-pKₐ prediction (building on the MolGpKa and Uni-pKa micro-pKₐ models). Bind Research. https://github.com/bindresearch/pkasso.',
 'Kavitha, C. N., Butcher, R. J., Jasinski, J. P., Yathirajan, H. S., & Dayananda, A. S. (2013). Cinnarizinium bis(p-toluenesulfonate) dihydrate. Acta Crystallographica Section E, 69, o485–o486. DOI 10.1107/S1600536813003991.',
 'Ferraz, R., Costa, A., Marrucho, I., Branco, L., Prudêncio, C., Noronha, J. P., & Petrovski, Ž. (2012). Development of novel ionic liquids based on valproate anion. Book of Abstracts, 6th Spanish-Portuguese-Japanese Organic Chemistry Symposium, Lisbon, p. 122.',
 'Yan, H., Zhong, X., & Liu, Y. (2024). Improving the solubility, stability, and bioavailability of albendazole through synthetic salts. Molecules, 29(15), 3571. DOI 10.3390/molecules29153571.',
 'Stepanovs, D., Jure, M., Yanichev, A., Belyakov, S., & Mishnev, A. (2015). Molecular salts of propranolol with dicarboxylic acids: diversity of stoichiometry, supramolecular structures and physicochemical properties. CrystEngComm, 17, 9023–9028. DOI 10.1039/C5CE01408E.',
 'Marenich, A. V., Cramer, C. J., & Truhlar, D. G. (2009). Universal solvation model based on solute electron density and on a continuum model of the solvent defined by the bulk dielectric constant and atomic surface tensions. Journal of Physical Chemistry B, 113(18), 6378–6396. DOI 10.1021/jp810292n.',
 'Ho, J., & Coote, M. L. (2010). A universal approach for continuum solvent pKₐ calculations: are we there yet? Theoretical Chemistry Accounts, 125, 3–21.',
 'Šebesta, F., Sovová, Ž., & Burda, J. V. (2024). Determination of amino acids’ pKₐ: importance of cavity scaling within implicit solvation models and choice of DFT functionals. Journal of Physical Chemistry B, 128(7), 1627–1637. DOI 10.1021/acs.jpcb.3c07007.',
 'King, G., & Zeng, L. (2001). Logistic regression in rare events data. Political Analysis, 9(2), 137–163.',
 'Tang, Y., Yang, K., Zhao, S., Chen, Q., Qin, L., & Qin, B. (2023). Evaluation of solubility, physicochemical properties, and cytotoxicity of naproxen-based ionic liquids. ACS Omega, 8, 8332–8340. DOI 10.1021/acsomega.2c07044.',
 'Bastos, J. C., Vieira, N. S. M., Gaspar, M. M., Pereiro, A. B., & Araújo, J. M. M. (2022). Human cytotoxicity, hemolytic activity, anti-inflammatory activity and aqueous solubility of ibuprofen-based ionic liquids. Sustainable Chemistry, 3(3), 358–375. DOI 10.3390/suschem3030023.',
 'Azevedo, A. M. O., Costa, S. P. F., Dias, A. F. V., Marques, A. H. O., Pinto, P. C. A. G., Bica, K., Ressmann, A. K., Passos, M. L. C., Araújo, A. R. T. S., Reis, S., & Saraiva, M. L. M. F. S. (2017). Anti-inflammatory choline based ionic liquids: insights into their lipophilicity, solubility and toxicity parameters. Journal of Molecular Liquids, 232, 20–26. DOI 10.1016/j.molliq.2017.02.027.',
 'IUPAC. Digitized pKₐ dataset (Dissociation-Constants). https://github.com/IUPAC/Dissociation-Constants.']
for i, r in enumerate(refs, 1):
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(8); p.paragraph_format.left_indent = Cm(0.8); p.paragraph_format.first_line_indent = Cm(-0.8)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY; _run(p, f'{i}. '); _run(p, r)

doc.save('DDP_report_Mahima.docx'); print('saved DDP_report_Mahima.docx')
