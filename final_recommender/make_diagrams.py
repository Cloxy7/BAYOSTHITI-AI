"""Report diagrams in a plain paper style: white background, thin square boxes, two accent colours, panel letters,
and small real plots instead of text boxes. All numbers come from features.py and the result files."""
import io, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
from PIL import Image
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 8, 'axes.linewidth': 0.6, 'mathtext.fontset': 'stixsans',
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.spines.top': False, 'axes.spines.right': False})
K = '#222222'; G = '#8a8a8a'; LG = '#e6e6e6'; ACC = '#1f5fa8'; ACC2 = '#c8553d'
DPI = 300


def mol_img(smi, w=420, h=300):
    m = Chem.MolFromSmiles(smi); d = rdMolDraw2D.MolDraw2DCairo(w, h)
    o = d.drawOptions(); o.clearBackground = False; o.bondLineWidth = 1.6; o.padding = 0.06; o.minFontSize = 13
    d.DrawMolecule(m); d.FinishDrawing()
    return Image.open(io.BytesIO(d.GetDrawingText()))


def mol_ax(fig, rect, smi, frame=None, lw=0.6):
    ax = fig.add_axes(rect); ax.imshow(mol_img(smi)); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(frame is not None); s.set_color(frame or K); s.set_linewidth(lw)
    return ax


def farrow(fig, p1, p2, color=K, lw=0.8, ms=8):
    fig.add_artist(FancyArrowPatch(p1, p2, transform=fig.transFigure, arrowstyle='-|>', mutation_scale=ms, lw=lw, color=color))


def rbox(fig, x, y, w, h, ec=K, fc='white', lw=0.7):
    fig.add_artist(Rectangle((x, y), w, h, transform=fig.transFigure, ec=ec, fc=fc, lw=lw))


def ftext(fig, x, y, s, **kw):
    kw.setdefault('ha', 'center'); kw.setdefault('va', 'center'); kw.setdefault('color', K); kw.setdefault('fontsize', 8)
    fig.text(x, y, s, **kw)


def panel(fig, x, y, letter, title):
    fig.text(x, y, f'({letter})', fontsize=9, fontweight='bold', color=K, ha='left', va='top')
    fig.text(x + 0.032, y, title, fontsize=8.5, color=K, ha='left', va='top')


import features; features.inject(globals())
SCORES = pd.read_csv('all_scores_leave_drug_out.csv')
iI = list(D.name).index('Ibuprofen')
SH = {'CC(O)C[NH2+]CC(C)O': 'Diisopropanolammonium'}
def iname(r):
    n = r['ion_name'] if isinstance(r['ion_name'], str) else SH.get(r['ion_smiles'], r['ion_smiles'][:18])
    n = n.split(' (')[0]
    return {'Bis(2-hydroxyethyl)ammonium': 'Diethanolammonium', 'Tris(2-hydroxypropyl)ammonium': 'Triisopropanolammonium'}.get(n, n)

# ======================= Figure 1: API ionic liquid =======================
fig = plt.figure(figsize=(6.2, 1.9))
mol_ax(fig, [0.00, 0.20, 0.36, 0.78], 'CC(C)Cc1ccc(C(C)C(=O)[O-])cc1')
mol_ax(fig, [0.43, 0.24, 0.22, 0.70], 'C[N+](C)(C)CCO')
ftext(fig, 0.395, 0.58, '+', fontsize=16, color=G)
farrow(fig, (0.68, 0.58), (0.76, 0.58), lw=0.9)
ftext(fig, 0.18, 0.12, 'Ibuprofenate', fontweight='bold'); ftext(fig, 0.18, 0.03, 'API, anion', color=G, fontsize=7.5)
ftext(fig, 0.54, 0.12, 'Cholinium', fontweight='bold'); ftext(fig, 0.54, 0.03, 'counter-ion, cation', color=G, fontsize=7.5)
ftext(fig, 0.885, 0.62, '[Cho][Ibu]', fontsize=10, fontweight='bold'); ftext(fig, 0.885, 0.47, 'API ionic liquid', color=G, fontsize=7.5)
fig.savefig('fig_api_il_example.png', dpi=DPI); plt.close(fig)

# ======================= Figure 3: method overview =======================
g = SCORES[SCORES.drug == iI].sort_values('rank')
rows = g.head(6)
fig = plt.figure(figsize=(7.0, 2.75))
panel(fig, 0.005, 0.985, 'a', 'Input')
mol_ax(fig, [0.005, 0.50, 0.15, 0.38], D.smiles[iI], frame=ACC, lw=0.9)
ftext(fig, 0.08, 0.465, 'query drug (held out)', fontsize=7, color=ACC)
for n, smi in enumerate(rows.ion_smiles.values[[3, 4]]):
    mol_ax(fig, [0.015 + n * 0.075, 0.14, 0.065, 0.25], smi, frame=G, lw=0.5)
ftext(fig, 0.165, 0.265, '…', fontsize=10, color=G)
ftext(fig, 0.085, 0.085, f'{len(g)} candidate ions', fontsize=7, color=G)
ftext(fig, 0.085, 0.04, '(opposite charge)', fontsize=7, color=G)
farrow(fig, (0.185, 0.52), (0.225, 0.52))
panel(fig, 0.22, 0.985, 'b', 'Pair features')
axb = fig.add_axes([0.345, 0.19, 0.15, 0.66])
feat = ['sim2D', 'sim3D', 'dpKa_value', 'ESP_comp']; lab = ['2D', '3D', 'ΔpKa', 'ESP']
M = rows[feat].values.astype(float)
lo = g[feat].min().values; hi = g[feat].max().values; Z = (M - lo) / (hi - lo)
axb.imshow(np.nan_to_num(Z, nan=0.0), cmap='Blues', vmin=-0.15, vmax=1.15, aspect='auto')
for a in range(M.shape[0]):
    for b in range(M.shape[1]):
        v = M[a, b]; s = 'perm.' if np.isnan(v) else (f'{v:.0f}' if b == 3 else f'{v:.2f}' if b < 2 else f'{v:.1f}')
        axb.text(b, a, s, ha='center', va='center', fontsize=6.3, color='white' if (not np.isnan(Z[a, b]) and Z[a, b] > 0.6) else K)
axb.set_xticks(range(4)); axb.set_xticklabels(lab, fontsize=6.8)
axb.xaxis.tick_top(); axb.set_yticks(range(len(rows)))
axb.set_yticklabels([iname(r) for r in rows.to_dict('records')], fontsize=6.3)
axb.tick_params(length=0)
for s in axb.spines.values(): s.set_visible(True); s.set_linewidth(0.5)
ftext(fig, 0.36, 0.08, 'one row per (drug, ion) pair;', fontsize=6.8, color=G)
ftext(fig, 0.36, 0.035, 'similarity is to training drugs using the ion', fontsize=6.8, color=G)
farrow(fig, (0.505, 0.52), (0.545, 0.52))
panel(fig, 0.54, 0.985, 'c', 'Additive score model')
PE = pd.read_csv('partial_effects.csv'); ylo, yhi = PE.contribution_logit.min(), PE.contribution_logit.max(); pad = 0.08 * (yhi - ylo); names = {'sim2D': r'$f_{2D}$', 'sim3D': r'$f_{3D}$', 'dpKa': r'$f_{pKa}$', 'ESP_comp': r'$f_{ESP}$'}
for k, f in enumerate(['sim2D', 'sim3D', 'dpKa', 'ESP_comp']):
    r_, c_ = divmod(k, 2)
    ax = fig.add_axes([0.575 + c_ * 0.105, 0.55 - r_ * 0.33, 0.085, 0.22])
    q = PE[PE.feature == f]; ax.plot(q.value, q.contribution_logit, color=ACC, lw=1.1)
    ax.axhline(0, color=LG, lw=0.6, zorder=0); ax.set_title(names[f], fontsize=8, pad=2)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(ylo - pad, yhi + pad)
ftext(fig, 0.677, 0.115, r'$\mathrm{logit}\,P = b_0 + \sum_g f_g(x_g)$', fontsize=8)
ftext(fig, 0.677, 0.035, r'score $= 100 \times$ calibrated $P$', fontsize=7.2, color=G)
farrow(fig, (0.785, 0.52), (0.825, 0.52))
panel(fig, 0.82, 0.985, 'd', 'Ranked list')
axd = fig.add_axes([0.85, 0.19, 0.135, 0.64])
sc = rows.score.values; rep = rows.is_reported_pair.values.astype(bool)
axd.barh(range(len(sc)), sc, color=[ACC2 if r else '#b9b9b9' for r in rep], height=0.62)
axd.invert_yaxis(); axd.set_yticks(range(len(sc))); axd.set_yticklabels([f'#{int(r)}' for r in rows['rank']], fontsize=6.8)
axd.set_xlabel('score', fontsize=7, labelpad=1); axd.set_xlim(0, 30); axd.set_xticks([0, 15, 30])
fig.add_artist(Rectangle((0.85, 0.885), 0.012, 0.03, transform=fig.transFigure, fc=ACC2, ec='none')); ftext(fig, 0.866, 0.90, 'reported', ha='left', fontsize=6.5)
fig.add_artist(Rectangle((0.92, 0.885), 0.012, 0.03, transform=fig.transFigure, fc='#b9b9b9', ec='none')); ftext(fig, 0.936, 0.90, 'new', ha='left', fontsize=6.5)
fig.savefig('fig_model_workflow.png', dpi=DPI); plt.close(fig)

# ======================= Figure 4: dataset construction + skew =======================
fig = plt.figure(figsize=(7.0, 2.3))
panel(fig, 0.005, 0.985, 'a', 'Dataset construction')
for x, y, w, h, t1, t2 in [(0.01, 0.58, 0.14, 0.24, 'PubMed', '1,935 abstracts'), (0.01, 0.20, 0.14, 0.24, 'Europe PMC', '2,707 full texts')]:
    rbox(fig, x, y, w, h); ftext(fig, x + w / 2, y + h * 0.66, t1, fontsize=7.5, fontweight='bold'); ftext(fig, x + w / 2, y + h * 0.32, t2, fontsize=7)
farrow(fig, (0.15, 0.70), (0.185, 0.56)); farrow(fig, (0.15, 0.32), (0.185, 0.46))
rbox(fig, 0.185, 0.36, 0.11, 0.30); ftext(fig, 0.24, 0.55, 'extraction', fontsize=7.5); ftext(fig, 0.24, 0.45, 'and curation', fontsize=7.5)
farrow(fig, (0.295, 0.51), (0.33, 0.51))
rbox(fig, 0.33, 0.36, 0.13, 0.30, ec=ACC, lw=0.9)
ftext(fig, 0.395, 0.57, '652', fontsize=10, fontweight='bold'); ftext(fig, 0.395, 0.47, 'API–counter-ion', fontsize=6.8); ftext(fig, 0.395, 0.41, 'pairs', fontsize=6.8)
farrow(fig, (0.46, 0.60), (0.495, 0.69)); farrow(fig, (0.46, 0.42), (0.495, 0.32))
rbox(fig, 0.495, 0.56, 0.14, 0.26); ftext(fig, 0.565, 0.74, '131 APIs', fontsize=7.5, fontweight='bold')
ftext(fig, 0.565, 0.655, '80 cations', fontsize=7); ftext(fig, 0.565, 0.595, '51 anions', fontsize=7)
rbox(fig, 0.495, 0.18, 0.14, 0.26); ftext(fig, 0.565, 0.36, '291 counter-ions', fontsize=7.5, fontweight='bold')
ftext(fig, 0.565, 0.275, '182 cations', fontsize=7); ftext(fig, 0.565, 0.215, '109 anions', fontsize=7)
panel(fig, 0.68, 0.985, 'b', 'Pairs per counter-ion')
P = pd.read_csv('positives.csv'); cnt = np.sort(P.ion_idx.value_counts().values)[::-1]
ax = fig.add_axes([0.735, 0.22, 0.25, 0.60])
ax.bar(np.arange(1, len(cnt) + 1), cnt, width=1.0, color=ACC, lw=0)
ax.set_xlabel('counter-ions, sorted by use', fontsize=7, labelpad=1); ax.set_ylabel('reported pairs', fontsize=7, labelpad=1)
ax.set_xlim(0, len(cnt) + 1)
one = int((cnt == 1).sum())
ax.text(len(cnt) * 0.97, cnt.max() * 0.9, f'{len(cnt)} ions\n{one} used only once', ha='right', va='top', fontsize=6.8, color=K)
fig.savefig('fig_data_pipeline.png', dpi=DPI); plt.close(fig)

# ======================= Figure 5: similarity for one pair =======================
j = [k for k in range(ni) if isinstance(I.name[k], str) and I.name[k] == 'Lidocainium'][0]
users = sorted([k for k in use if j in pos[k]], key=lambda k: -S2[iI, k])
show = users[:4]
fig = plt.figure(figsize=(7.0, 2.45))
panel(fig, 0.005, 0.985, 'a', 'Query and candidate')
mol_ax(fig, [0.01, 0.50, 0.15, 0.36], D.smiles[iI], frame=ACC, lw=0.9); ftext(fig, 0.085, 0.455, 'ibuprofen (query)', fontsize=7, color=ACC)
mol_ax(fig, [0.01, 0.08, 0.15, 0.32], I.smiles[j], frame=ACC2, lw=0.9); ftext(fig, 0.085, 0.035, 'lidocainium (candidate)', fontsize=7, color=ACC2)
panel(fig, 0.205, 0.985, 'b', f'Training drugs reported with lidocainium ({len(users)}; top {len(show)} shown)')
for n, k in enumerate(show):
    x = 0.22 + n * 0.135; best = n == 0
    mol_ax(fig, [x, 0.40, 0.12, 0.43], D.smiles[k], frame=ACC if best else G, lw=1.0 if best else 0.5)
    ftext(fig, x + 0.06, 0.335, D.name[k], fontsize=7)
    ftext(fig, x + 0.06, 0.255, f'2D {S2[iI, k]:.2f}   3D {S3[iI, k]:.2f}', fontsize=6.8, color=ACC if best else G, fontweight='bold' if best else 'normal')
farrow(fig, (0.165, 0.68), (0.212, 0.68), color=ACC)
fig.add_artist(Rectangle((0.212, 0.20), 0.545, 0.69, transform=fig.transFigure, ec=G, fc='none', lw=0.5, ls=(0, (3, 2))))
panel(fig, 0.78, 0.985, 'c', 'Pair feature')
farrow(fig, (0.757, 0.55), (0.80, 0.55))
fig.add_artist(Circle((0.835, 0.55), 0.032, transform=fig.transFigure, ec=K, fc='white', lw=0.7)); ftext(fig, 0.835, 0.55, 'max', fontsize=7)
farrow(fig, (0.868, 0.55), (0.895, 0.55))
s2 = max(S2[iI, k] for k in users); s3 = max(S3[iI, k] for k in users)
ftext(fig, 0.945, 0.60, f'sim2D = {s2:.2f}', fontsize=7.5, fontweight='bold'); ftext(fig, 0.945, 0.50, f'sim3D = {s3:.2f}', fontsize=7.5, fontweight='bold')
ftext(fig, 0.88, 0.25, 'ibuprofen itself is never', fontsize=6.5, color=G); ftext(fig, 0.88, 0.19, 'among the training drugs', fontsize=6.5, color=G)
fig.savefig('fig_similarity_example.png', dpi=DPI); plt.close(fig)

# ======================= Figure 6: leave-drug-out =======================
fig = plt.figure(figsize=(6.4, 1.9))
ax = fig.add_axes([0.10, 0.17, 0.42, 0.72])
for f in range(5):
    for gg in range(5):
        ax.add_patch(Rectangle((gg + 0.06, f + 0.12), 0.88, 0.76, fc=ACC2 if gg == f else LG, ec='none'))
ax.set_xlim(0, 5); ax.set_ylim(5, 0); ax.set_yticks(np.arange(5) + 0.5); ax.set_yticklabels([f'fold {f + 1}' for f in range(5)], fontsize=7)
ax.set_xticks(np.arange(5) + 0.5); ax.set_xticklabels([f'G{g + 1}' for g in range(5)], fontsize=7); ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
ax.set_xlabel('drug groups (all forms of one molecule in the same group)', fontsize=7, labelpad=2)
fig.add_artist(Rectangle((0.56, 0.70), 0.025, 0.08, transform=fig.transFigure, fc=LG, ec='none')); ftext(fig, 0.595, 0.74, 'train', ha='left', fontsize=7)
fig.add_artist(Rectangle((0.56, 0.56), 0.025, 0.08, transform=fig.transFigure, fc=ACC2, ec='none')); ftext(fig, 0.595, 0.60, 'held out', ha='left', fontsize=7)
ftext(fig, 0.70, 0.86, 'In each fold', ha='left', fontsize=7.5, fontweight='bold')
for n, s in enumerate(['1. fit on the train groups', '2. rank all candidate ions of', '    each held-out drug', '3. record Hit@1, Hit@5, Hit@10, MRR', '4. repeat with 2 random seeds']):
    ftext(fig, 0.70, 0.72 - n * 0.12, s, ha='left', fontsize=7)
fig.savefig('fig_leave_drug_out.png', dpi=DPI); plt.close(fig)
print('ok', len(users), s2, s3, one, len(cnt))
