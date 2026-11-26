# Reproducing every number in the report

← Back to the [project overview](../README.md) · live site: https://glittery-frangipane-f6cbaf.netlify.app/

# API–counter-ion recommender: code

Code for the DDP report "API–counter-ion recommender" (Jilla Sai Mahima, BE22B047, IIT Madras).
Given a drug, the recommender ranks the counter-ions most likely to form an API salt or ionic liquid with it.

## Install

```
pip install -r requirements.txt
```

Python 3.10 or newer. Everything in `final_recommender/` runs on a normal laptop (no GPU, no quantum chemistry).

## Folder map

| Folder / file | What is in it |
|---|---|
| `final_recommender/` | The model, all tests, the figures, the report builder, the web app builder and the two Colab notebooks. Run every script from inside this folder. |
| `measured_properties/` | Measured solubility, melting point and LD50 of single drugs and single counter-ions (read by the tests). |
| `Pairs_652_with_dpKa_reliability_and_safety.csv` | The 652 reported pairs with proof quotes, references and pKa values. |
| `Ion_safety_GRAS_SCOGS_all_ions.csv`, `ions_acceptability.csv` | Safety tags for the counter-ions. |
| `Unique_APIs_131_with_ionic_forms.csv` | The 131 unique APIs. |
| `patents_2026_11/` | Patent pairs (7 patents) and the check of where the model ranks them (report Section 5.7, Table 8). |
| `api_il_sol_tox/` | API-IL solubility and toxicity values found in the literature (report Section 5.8, Table 9). |
| `mega_workbook/` | Script that puts every CSV of the project into one Excel workbook. |
| `pKa_methods/` | How the counter-ion pKa was chosen: pKasso, the random forest, Graph-pKa, DFT, and the DFT-for-bases error analysis (report Section 4.2). |
| `supporting_notebooks/` | Notebooks behind numbers quoted in the report: the similarity benchmark, the dipole filter test and the xTB interaction-energy test. |
| `data_collection/` | Notebooks that harvested PubMed abstracts and Europe PMC full texts. |

## Main model: run order (inside `final_recommender/`)

The input tables are already included: `drugs.csv`, `ions.csv`, `positives.csv`, `drugs_3D_descriptors.csv`, `ions_3D_descriptors.csv`, `xtb_molecule_properties.csv`.

| Step | Script | What it does | Output |
|---|---|---|---|
| 0 | `features.py` | Shared code: builds the 7 pair features (sim2D, sim3D, ΔpKa, salt probability, two pKa flags, ESP) and the leave-drug-out folds. Imported by every other script. | – |

`features.py` is a normal importable module. It finds its own CSVs (the folder holding the file) and caches
the result, so the similarity matrices are built once per process instead of once per script:

```python
import features
F = features.load()                 # -> D, I, P, S2, S3, pos, use, pair_table, group_folds, ...
T = F.pair_table(train_idx, [drug_idx])

F = features.load('/path/to/final_recommender')    # works from any working directory
```

The scripts in this folder use `import features; features.inject(globals())`, which puts the same names in
the module namespace that the older `exec(open('features.py').read())` line did. New code should prefer
`features.load()` and go through the returned object — it avoids the name collisions that bare globals invite
(the score table read by `build_app_data.py`, `make_diagrams.py`, `new_pair_cases.py` and `final_rank.py` is
now called `SCORES`, because `A` already means the 3D descriptor table inside `features.py`).

Steps 1–5 still read and write their CSVs relative to the working directory, so run them from inside this folder.
| 1 | `run_setA.py` | Trains all models (logistic regression, XGBoost, LightGBM, random forest, SVM, MLP, ...) on the same features and folds. | `results_setA_all_models.csv`, `ablation_setA.csv`, `oof_scores_setA_seed0.csv` |
| 2 | `score_equation_cv.py` | The final spline score equation, leave-drug-out. | `results_score_equation.csv` |
| 3 | `final_equation_fit.py` | Platt calibration, the spline curves, the final fitted model. | `final_model.pkl`, `calibration_params.json`, `ablation_equation.csv`, `calibration_summary.csv`, `equation_coefficients.csv`, `partial_effects.csv`, `fig_partial_effects.png`, `fig_calibration.png` |
| 4 | `final_rank.py` | Ranks the counter-ions of every drug (each treated as unseen). | `all_scores_leave_drug_out.csv`, `top10_recommendations_all_drugs.csv`, `leave_one_drug_out_summary.csv` |
| 5 | `compare_equation.py` | Equation against the other models. | `equation_vs_others.csv` |

The result files of steps 1–5 are already included, so steps 6 onwards can be run without re-training. Steps 1 and 2 take several minutes each.

## Conversational interface (`final_recommender/agent/`)

A function-calling layer over the model, so a chemist can ask in plain English. The LLM routes and phrases;
it contributes no chemistry. Every number comes from the model and every literature claim is a verbatim
quote with its DOI.

```
python -m agent.demo                  # scripted transcript, real data, no API key needed

pip install anthropic                 # the live loop
export ANTHROPIC_API_KEY=...
python -m agent.chat "I'm working with tolfenamic acid. What should I pair it with?"
```

| Tool | What it returns |
|---|---|
| `search_literature(query, k=5, kind=, drug=, ion=)` | BM25 retrieval over a 1,628-passage corpus built from every evidence source in the project (see below). Returns passages with citations, or `found=false`. |
| `recommend_counterions(drug, pKa=None, top=10)` | Ranked ions with scores and the four contributions. Fast path for the 138 known drugs (precomputed ranking, no RDKit, ~5 ms); falls through to `recommend_new_drug` for any other charged SMILES. |
| `get_evidence(drug, ion)` | The verbatim `proof_quote` and `proof_ref` for that pair from the 652, or `found=false`. Never paraphrases, never guesses. |
| `explain_recommendation(drug, ion)` | The plain-English reason, from `reason.make_reason` — the same function behind the web app, so chat and app cannot drift apart. |

### Retrieval corpus (`final_recommender/rag/`)

`rag/corpus.py` turns nine project tables into 1,628 passages, each carrying the citation it came with —
100% of them. Nothing is paraphrased on the way in; a field that is empty is skipped rather than padded.

| Passages | Source | Answers |
|---|---|---|
| 652 + 99 | the reported pairs, with `proof_quote` and `proof_ref` | has this pair been made, and where is it written |
| 353 | FDA GRAS / SCOGS status per ion | is this counter-ion acceptable |
| 262 | measured melting point, logS, LD50 | what is known about this molecule on its own |
| 138 | drug pKa provenance | where did this pKa come from |
| 51 | pairs from 7 patents, with the rank this model gave | does the model find what patents claim |
| 30 | the four pKa methods benchmarked | why pKasso |
| 29 | measured API-IL solubility and toxicity | how much does solubility actually improve |
| 14 | written assessments of unreported suggestions | why was this one rejected |

Retrieval is BM25, not embeddings. The queries are chemical names and the corpus is short passages full of
them, so an exact match on "tolfenamate" matters more than semantic neighbourhood — and a sparse index
needs no model, no GPU and no network. The one piece of domain knowledge is in the tokeniser: every word of
7+ letters also indexes its 5-letter prefix, so choline/cholinium and tolfenamic/tolfenamate land on the
same term. (Suffix stripping was tried first; it collapsed `cholinium` and `cholate` onto one term and put
cholate at the top of a safety query about choline. A test pins the fix.)

Build is ~90 ms over 1,628 passages and 3,342 terms; a query is ~2 ms.

Three behaviours are deliberate rather than incidental:

- A pair with no record returns `found=false` and no quote. The answer reports "ranked 4th, no published
  record" instead of resolving that tension by inventing a citation.
- A property description ("an acidic drug, pKa 4.2, aromatic ring") is refused. Two of the four features are
  similarity to *specific* known molecules, so there is nothing to compute from a description.
- Questions about solubility, toxicity or bioavailability are declined. The caveat rides on every response.

`agent/tools.py` also handles two mismatches between the tables and the way chemists talk: `drugs.csv` stores
the ionic form (`Tolfenamate`, `Lidocainium`) while a chemist types the neutral name, and 87 of the 291 ions
have no name at all, so an unnamed ion is reported by its class ("unnamed long-chain pyridinium cation")
rather than as a bare SMILES.

## Test suite

```
pytest final_recommender/tests          # from the repo root
```

26 tests in two files. `tests/test_outputs.py` needs only pandas and numpy, so it runs in a bare
environment; `tests/test_features.py` needs RDKit and is skipped automatically without it.

| File | What it pins |
|---|---|
| `tests/test_rag.py` | The retrieval layer: every passage carries a citation; every returned string exists verbatim in the corpus; the ionic and neutral spellings of a name share a stem while unrelated names do not; a filtered query cannot leak a similar-looking ion; a miss returns `found=false` with no passages. |
| `tests/test_agent.py` | The grounding guarantees of the chat layer: an unreported pair returns no quote and no citation; a property description is refused; the scope caveat rides on every answer; unnamed ions never surface as a bare SMILES. |
| `tests/test_features.py` | The three leakage guards: every form of one molecule stays in one CV fold; a drug alone in training produces no rows at all (it can never reference itself); no ion is offered without an eligible training user. Plus the ΔpKa sign convention and the permanent-cation rule. |
| `tests/test_outputs.py` | That the shipped CSVs still say what the report says. Hit@1/5/10 and MRR recomputed from `all_scores_leave_drug_out.csv` against `leave_one_drug_out_summary.csv` (over all 138 drugs, so the 6 whose reported ion is unreachable count as misses); the score rebuilt from its four contributions to 0.04 points; the reported spline settings are the pre-registered ones and not the tuned best; the advantage over XGBoost excludes zero. |

These are regression tests, not unit tests of the chemistry: they fail when a change to the pipeline moves a
number the report quotes.

## Tests reported in the report

| Script | Report section | Output |
|---|---|---|
| `esp_choice_test.py` (and `esp_choice_test_rerun.py`, which adds the Hit@1 and MRR intervals) | 4.3, Table 3 (why ESP) and 4.4 (8,000 training rows) | `esp_choice_test.csv`, `training_rows_test.csv` |
| `extra_features_test.py` | 5.9 (dipole, solubility, toxicity as features) | `extra_features_test.csv` |
| `safety_feature_test.py` | 5.6 (safety as a feature) | `safety_feature_test.csv` |
| `old_2D_descriptor_model_test.py` | 5.9 (older descriptor model) | `old_2D_descriptor_model_test.csv` |
| `new_pairs_theory_check.py`, `new_pair_cases.py`, `justify_pairs.py` | 5.7 (new suggestions) | `new_pairs_theory_check.csv`, `new_pairs_evidence.csv`, `justified_recommendations.csv` |
| `../patents_2026_11/patent_pairs_raw.py` → `match_final.py` → `patent_pairs_ranked.py` | 5.7, Table 8 | `patent_pairs_unique.csv`, `patent_pairs_new_ranked.csv` |
| `../api_il_sol_tox/build_sol_tox.py` | 5.8, Table 9 | `API_IL_solubility_toxicity_values.csv` |

## Using the model for a new drug

```python
import recommend_new_drug as r
table, meta = r.recommend('CC(C)Cc1ccc(C(C)C(=O)[O-])cc1', name='my drug', pKa=4.4, top=10)
print(table[['rank', 'ion_name', 'score', 'pKa_label', 'safety']])
```

Give the drug as a charged SMILES (anion for an acid, cation for a base). The pKa is optional and should be a measured value. `reason.py` writes the short explanation for each suggestion. The same thing runs in Colab with `Colab_recommend_for_new_drug.ipynb`; `Colab_reproduce_final_recommender.ipynb` re-runs the whole model.

## Figures, report and web app

| Script | Output |
|---|---|
| `make_figures.py` | `fig_model_comparison.png`, `fig_contribution.png`, `fig_example_top10.png` |
| `make_diagrams.py` | Figures 1, 3, 4, 5 and 6 (API-IL example, method overview, dataset, similarity example, leave-drug-out) |
| `build_report_v2.py` | `DDP_report_Mahima.docx` (convert to PDF with Word or LibreOffice). Needs all `fig_*.png` and `report_assets/iitm_logo_0.png`. |
| `build_app_data.py` → `build_app_html.py` | `app_data.json` → `app.html` (the web app, from `app_template.html`) |

`fig_solubility_boost.png` (Figure 2) is drawn from the literature values in reference [3] and is included as an image.

## Notes

- `mega_workbook/build_mega_all_csv.py` collects every CSV below the folder it is placed in. To rebuild the full project workbook, put the `mega_workbook` folder inside the full project outputs folder and run it there.
- The scripts in `pKa_methods/` are kept as they were run. Some of them read temporary working files (prepared input tables and cloned copies of the pKasso, MolGpKa and QupKake repositories) that were not kept; their results are in the CSV files next to them, and these are the numbers quoted in the report. pKasso is at https://github.com/bindresearch/pkasso. The DFT and the two machine-learning pKa models were run in Colab with the notebooks in `pKa_methods/`.
- `supporting_notebooks/` and `data_collection/` are Colab notebooks; they need internet access and, for xTB, the xtb program.
