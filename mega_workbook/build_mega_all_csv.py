"""One Excel workbook with every CSV file of the project (Nov 2026).
Sheet 'Index' lists every sheet; 'Key tables' (K..) come first, then every CSV (F...), identical files merged into one sheet."""
import os, hashlib, re, math, pandas as pd, numpy as np, xlsxwriter
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); OUT = os.path.join(ROOT, 'mega_workbook', 'DDP_all_data_mega_workbook.xlsx')
FR = os.path.join(ROOT, 'final_recommender')
ions = pd.read_csv(f'{FR}/ions.csv'); keep_i = [c for c in ['unique_ion_id', 'name', 'smiles', 'sign', 'pKa', 'perm', 'safety_tag'] if c in ions.columns]
drugs = pd.read_csv(f'{FR}/drugs.csv').iloc[:, :6]
KEY = [  # (sheet, description, dataframe or path)
 ('K01 Pairs 652', 'The 652 reported API–counter-ion pairs with proof quotes, references, pKa values, ΔpKa label and safety tier', f'{ROOT}/Pairs_652_with_dpKa_reliability_and_safety.csv'),
 ('K02 APIs 131', 'The 131 unique APIs: ionic form (80 cations, 51 anions), SMILES, number of pairs, drug pKa', f'{ROOT}/Unique_APIs_131_with_ionic_forms.csv'),
 ('K03 Drug entries 138', 'The 138 drug entries used by the model (5 APIs appear as both cation and anion)', drugs),
 ('K04 Counter-ions cations', 'The 182 cation counter-ions used by the model (perm = 1: permanent cation)', ions[ions.sign > 0][keep_i]),
 ('K05 Counter-ions anions', 'The 109 anion counter-ions used by the model', ions[ions.sign < 0][keep_i]),
 ('K06 Drug pKa sources', 'Where every drug pKa comes from (measured / site not stated / computed database value)', f'{FR}/drug_pKa_sources_138.csv'),
 ('K07 API-IL sol tox values', 'Numeric aqueous solubility and toxicity values found for API-ILs in the literature (Nov 2026 search)', f'{ROOT}/api_il_sol_tox/API_IL_solubility_toxicity_values.csv'),
 ('K08 Ion measured props', 'Measured water solubility (logS), melting point and LD50 of each counter-ion (single ion, not the pair)', f'{ROOT}/measured_properties/counter_ions_measured_properties.csv'),
 ('K09 Drug measured props', 'Measured water solubility, melting point and LD50 of each drug (single drug)', f'{ROOT}/measured_properties/drugs_measured_properties.csv'),
 ('K10 Patent pairs', 'Drug–counter-ion pairs named in 7 patents (Nov 2026 search), matched to the 652 pairs', f'{ROOT}/patents_2026_11/patent_pairs_unique.csv'),
 ('K11 Patent pairs ranked', 'Where the recommender ranks the new patent pairs whose drug and ion are in the data', f'{ROOT}/patents_2026_11/patent_pairs_new_ranked.csv'),
 ('K12 Top-10 all drugs', 'Top-10 recommended counter-ions for every drug (each ranked as if unseen)', f'{FR}/top10_recommendations_all_drugs.csv'),
 ('K13 Model comparison', 'Hit@k and MRR of all models on the same features (leave-drug-out)', f'{FR}/results_setA_all_models.csv'),
 ('K14 ESP choice test', 'Which electrostatic descriptor to use: six xTB descriptors tested in the score equation', f'{FR}/esp_choice_test.csv'),
 ('K15 Training rows test', 'Does the 8,000-row training sample matter? 4,000 / 8,000 / 12,000 / all rows', f'{FR}/training_rows_test.csv'),
]
def h(p): return hashlib.md5(open(p, 'rb').read()).hexdigest()
paths = sorted(p for p in (os.path.join(d, f) for d, _, fs in os.walk(ROOT) for f in fs if f.lower().endswith('.csv'))
               if '__pycache__' not in p and '/mega_workbook/' not in p)
groups = {}
for p in paths: groups.setdefault(h(p), []).append(p)
key_hash = {h(src): sh for sh, _, src in KEY if isinstance(src, str)}
wb = xlsxwriter.Workbook(OUT, {'constant_memory': True, 'strings_to_numbers': False, 'strings_to_formulas': False, 'strings_to_urls': False, 'nan_inf_to_errors': False})
wb.formats[0].set_font_name('Arial'); wb.formats[0].set_font_size(10)
HB = wb.add_format({'bold': True, 'font_name': 'Arial', 'font_size': 10, 'bg_color': '#DDE6F0', 'border': 1})
TB = wb.add_format({'bold': True, 'font_name': 'Arial', 'font_size': 12}); LK = wb.add_format({'font_name': 'Arial', 'font_size': 10, 'font_color': 'blue', 'underline': 1})
WR = wb.add_format({'font_name': 'Arial', 'font_size': 10, 'text_wrap': True, 'valign': 'top'})
idx = wb.add_worksheet('Index')
def read(src):
    if isinstance(src, pd.DataFrame): return src
    try: return pd.read_csv(src, low_memory=False)
    except Exception:
        try: return pd.read_csv(src, low_memory=False, encoding='latin-1')
        except Exception:
            import csv
            rows = list(csv.reader(open(src, encoding='utf-8', errors='replace')))
            w = max(len(r) for r in rows); hdr = rows[0] + [f'extra_col_{k}' for k in range(len(rows[0]), w)]
            return pd.DataFrame([r + [None] * (w - len(r)) for r in rows[1:]], columns=hdr)
def write_sheet(name, df, note):
    ws = wb.add_worksheet(name); ws.freeze_panes(2, 0)
    ws.write_string(0, 0, note[:32000])
    for j, c in enumerate(df.columns): ws.write_string(1, j, str(c)[:32000], HB); ws.set_column(j, j, min(max(10, len(str(c)) + 2), 45))
    for i, row in enumerate(df.itertuples(index=False), start=2):
        for j, v in enumerate(row):
            if v is None or (isinstance(v, float) and math.isnan(v)): continue
            if isinstance(v, (bool, np.bool_)): ws.write_boolean(i, j, bool(v))
            elif isinstance(v, (int, float, np.integer, np.floating)) and math.isfinite(float(v)): ws.write_number(i, j, float(v))
            else: ws.write_string(i, j, str(v)[:32000])
    return len(df), df.shape[1]
rows_idx = []
for sh, desc, src in KEY:
    df = read(src); r, c = write_sheet(sh, df, desc)
    rows_idx.append(('Key table', sh, desc, os.path.relpath(src, ROOT) if isinstance(src, str) else 'derived from final_recommender files', r, c))
used = set(); n = 0
for hh, ps in sorted(groups.items(), key=lambda kv: kv[1][0]):
    rel = [os.path.relpath(p, ROOT) for p in ps]
    if hh in key_hash:
        rows_idx.append(('All CSV files', key_hash[hh], 'same content as the key table', ' | '.join(rel), None, None)); continue
    n += 1; base = re.sub(r'[\[\]\:\*\?\/\\]', '_', os.path.splitext(os.path.basename(ps[0]))[0])
    sh = f'F{n:03d} ' + base[:26]
    while sh.lower() in used: sh = sh[:-1]
    used.add(sh.lower())
    r, c = write_sheet(sh, read(ps[0]), 'Source file(s): ' + ' | '.join(rel)); print(n, sh, r, c, flush=True)
    rows_idx.append(('All CSV files', sh, '', ' | '.join(rel), r, c))
# index (written last in memory order is fine: constant_memory only needs row order per sheet)
idx.write_string(0, 0, 'DDP API–counter-ion recommender: all project data in one workbook', TB)
idx.write_string(1, 0, f'{len(KEY)} key tables, then {n} sheets holding every CSV file of the project ({len(paths)} files; identical copies are merged into one sheet and all their locations are listed). Click a sheet name to open it. Row 1 of every sheet says where it came from; row 2 is the header.')
for j, hd in enumerate(['Section', 'Sheet', 'Description', 'Source file(s) (relative to the outputs folder)', 'Rows', 'Columns']): idx.write_string(3, j, hd, HB)
for i, (sec, sh, desc, src, r, c) in enumerate(rows_idx, start=4):
    idx.write_string(i, 0, sec); idx.write_url(i, 1, f"internal:'{sh}'!A1", LK, sh); idx.write_string(i, 2, desc, WR); idx.write_string(i, 3, src[:32000], WR)
    if r is not None: idx.write_number(i, 4, r); idx.write_number(i, 5, c)
idx.set_column(0, 0, 14); idx.set_column(1, 1, 34); idx.set_column(2, 2, 60); idx.set_column(3, 3, 80); idx.set_column(4, 5, 9); idx.freeze_panes(4, 0)
idx.activate(); idx.set_first_sheet()
wb.close(); print('sheets', len(KEY) + n + 1, 'files', len(paths), 'unique', len(groups), 'size MB', round(os.path.getsize(OUT) / 1e6, 1))
