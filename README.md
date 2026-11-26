# Which counter-ion should this drug be paired with?

An interpretable machine-learning recommender for API ionic liquids, built on a 652-pair literature
dataset I curated from ~4,600 papers because the dataset did not exist.

Given a drug, it ranks the 291 known counter-ions, scores each one, and shows its reasoning. On drugs it
has never seen, **a counter-ion the literature actually reports lands in its top ten 72% of the time**,
against 22.5% for random ranking.

### ▶ Try it live — **[glittery-frangipane-f6cbaf.netlify.app](https://glittery-frangipane-f6cbaf.netlify.app/)**

Pick any of 138 drugs and the model ranks the counter-ions it is most likely to pair with, each with the
reason behind its score. Runs in the browser — no install, no sign-in.

Dual Degree Project · Jilla Sai Mahima (BE22B047) · Dept. of Biotechnology, IIT Madras
Guide: Prof. Sanjib Senapati

**One-page summary:** [`docs/one-pager.pdf`](docs/one-pager.pdf)

---

## The problem

About **40% of marketed drugs** and **70–90% of pipeline candidates** are too insoluble to work as they
are. One standard fix is to pair the drug with a counter-ion so it forms a salt — or, below ~100 °C, an
ionic liquid with no crystal lattice to dissolve at all.

Which counter-ion you choose is the whole decision. For ibuprofen, the cholinium ionic liquid is roughly
**27,000× more soluble** than the free acid, and **4.6×** more soluble than the ordinary sodium salt. Same
drug, same measurement — only the partner ion changes.

But there was no method for choosing it. Ibuprofen alone has 158 candidate cations, the literature reports
291 counter-ions in all, and every pair has to be synthesised and characterised to find out whether it
works. Papers report the pairs that happened to work, not a way to choose.

## What this does

The live site above is the quickest look. Locally, the conversational layer runs with no API key:

```
cd final_recommender && PYTHONPATH=. python -m agent.demo
```

```
-> recommend_counterions({"drug": "tolfenamic acid", "top": 5})          [77 ms]
   Tolfenamate, acid (anion), 174 candidate ions
    1. Cholinium                      31.97   permanent cation        GRAS/SCOGS listed
    2. unnamed quaternary ammonium    31.38   permanent cation        biocompatible
    3. unnamed long-chain quat. amm.  30.45   permanent cation        [already reported]
    4. Lidocainium                    29.85   salt likely (ΔpKa 4.8)  dual-active drug ion

-> get_evidence({"drug": "Ibuprofen", "ion": "Lidocainium"})             [0.4 ms]
   found = false — no reported pairing of this drug and this ion in the 652-pair dataset
```

That second call is the point. The model ranks lidocainium 4th for ibuprofen; the literature tool says
there is no published record. The answer reports both, because there is no code path that returns a
plausible-looking citation. When a pair *does* exist, the quote comes back verbatim with its DOI.

There is also a self-contained web app covering all 138 drugs — one HTML file, no install.

## How it works

**A dataset that did not exist.** 1,935 PubMed abstracts and 2,707 Europe PMC full texts, read down to
**652 drug–counter-ion pairs** over 131 drugs (138 ionic forms) and 291 counter-ions. 384 stated directly
in the text, 268 implied. Every pair carries the sentence it came from and its DOI, so any row traces back
to its paper.

**One idea, tested.** Similar drugs take similar counter-ions. In this data, a drug's nearest structural
neighbour shares a counter-ion with it 53% of the time, against 27% at random.

**Seven features in four groups.** 2D structural similarity (Morgan fingerprints, Tanimoto to training
drugs that use the ion), 3D shape similarity (13 descriptors over ETKDG conformers), the pKa group
(ΔpKa, salt probability from the Cruz-Cabeza fit, two flags), and electrostatic complementarity from
GFN2-xTB surface charges.

**An additive spline model**, cubic B-splines per feature into a regularised logistic regression with
Platt calibration — 28 spline weights, 3 linear terms, one intercept.

**Scored leave-drug-out**, five folds × two seeds, with every protonation state and duplicate drawing of a
molecule held out together so the model is never tested on a drug it trained on.

| | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|
| This model | 0.326 | 0.612 | **0.717** | 0.449 |
| Random baseline | 0.043 | 0.123 | 0.225 | 0.109 |

95% CI on Hit@10: 0.649–0.790. Under leave-*one*-drug-out the same model reads 0.739.

## Three things worth a look

### The simple model won, so I shipped the simple one

| Model (same features, same folds) | Hit@10 | 95% CI |
|---|---|---|
| Plain logistic regression | 0.728 | 0.656–0.801 |
| **Additive spline equation** *(shipped)* | **0.717** | 0.649–0.790 |
| 2D similarity alone (kNN) | 0.652 | 0.580–0.725 |
| XGBoost | 0.641 | 0.565–0.714 |
| Extra trees | 0.634 | 0.565–0.710 |
| LightGBM | 0.630 | 0.562–0.699 |
| Random forest | 0.605 | 0.533–0.681 |
| SVM (RBF) | 0.399 | 0.333–0.467 |
| Random | 0.225 | 0.178–0.275 |
| MLP (neural net) | 0.152 | 0.098–0.210 |

Eight models on identical inputs and folds, plus a graph neural network re-trained on ~5,900 molecules for
the pKa step. With roughly five known counter-ions per drug, the flexible models had too little to learn
from — the gradient-boosted trees all land below a plain linear model, and the neural net scores *below
random*. The advantage over XGBoost is +0.076 with a confidence interval that excludes zero.

Plain logistic regression edges the spline by 0.011, well inside both intervals. I shipped the spline
anyway, because it buys a readable partial-effect curve per feature — the thing that lets a chemist see
*why* an ion ranked where it did — for no measurable accuracy cost. What ships is one line of arithmetic,
not a black box.

The ablation says where the signal is: 2D similarity alone accounts for 71% of the total drop when feature
groups are removed (3D 20%, pKa 6%, ESP 4%).

### The reported number is the one I committed to, not the best one

Four spline settings were tried. The one in the report is the setting fixed **before any result was seen**:

| setting | Hit@10 |
|---|---|
| 6 knots, C=0.3 | 0.739 |
| 4 knots, C=1 | 0.736 |
| 5 knots, C=0.1 | 0.732 |
| **5 knots, C=1 — fixed in advance** | **0.717** |

Tuning would have bought 0.022 and nobody would have known. A test in the suite asserts the reported row
is the pre-registered one *and* that it is not the best of the four, so this cannot quietly drift.

### It reconstructed chemistry it was never shown

Across 138 drugs the top ten gives 1,380 pairs; 186 were already in the dataset, so I searched the other
1,194 by hand. **36 turned out to be real published salts** — 25 of them in ranks 1–5, eight reported as
ionic liquids — plus 450 with related evidence. Separately, the model was checked against seven patents.

Three it ranks first with no publication yet, each with a close analogue already in the literature:
hydroxychloroquine + betulinate, chloroquine + dodecyl sulfate, tolfenamic acid + cholinium.

**Every score reads back.** It decomposes into four contributions and one shared intercept (−3.74),
rebuilding to within 0.04 points across all 18,879 scored pairs — verified by a test, not asserted.

## Repository

```
final_recommender/      the model, the tests, the figures, the web app builder
  features.py           the 7 pair features and the leave-drug-out folds
  agent/                conversational layer: 4 tools + function-calling loop + demo
  tests/                79 tests (66 run without RDKit)
pKa_methods/            how the counter-ion pKa was chosen: pKasso vs random forest vs GNN vs DFT
data_collection/        the PubMed and Europe PMC harvest notebooks
patents_2026_11/        where the model ranks pairs taken from 7 patents
```

```
pip install -r requirements.txt
pytest final_recommender/tests          # 79 passed (66 without RDKit)
```

The suite is regression tests, not unit tests of chemistry: it pins the three leakage guards (every form of
one molecule stays in one CV fold; a drug alone in training produces no rows, so it can never reference
itself; no ion is offered without an eligible training user), recomputes every headline metric from the raw
scores, and asserts the grounding guarantees of the chat layer.

[`docs/REPRODUCING.md`](docs/REPRODUCING.md) has the full run order for reproducing every number and figure in the report.

## Honest limits

- **Sparse data.** 652 pairs, ~5 counter-ions per drug, and 173 of 291 ions appear with one drug only.
  That is why a four-term equation beats gradient boosting here, and it bounds how far this generalises.
- **A fixed candidate list.** Only the 291 reported counter-ions can be suggested. Proposing genuinely new
  ions is Phase 2 — a drug-conditioned diffusion model, filtered and ranked by the model above.
- **It predicts reports, not properties.** The model learns which pairings the literature records. It does
  not predict solubility, toxicity or bioavailability, and the chat layer declines those questions.
- **Unnamed ions.** 87 of the 291 have no name in the source data and currently surface by structural
  class ("unnamed long-chain pyridinium cation"). Naming them is the highest-value fix outstanding.
