"""Consistency of the shipped result files. No RDKit needed, so these run anywhere.

These pin the numbers quoted in the report to the artefacts in the repo: if a future change to the
pipeline moves Hit@10, these fail and say by how much, instead of the report and the code drifting apart.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest


def _csv(rec_dir, name):
    return pd.read_csv(os.path.join(rec_dir, name))


@pytest.fixture(scope='module')
def scores(rec_dir):
    return _csv(rec_dir, 'all_scores_leave_drug_out.csv')


# ---------------------------------------------------------------- inputs

def test_positives_index_into_the_tables(rec_dir):
    D = _csv(rec_dir, 'drugs.csv'); I = _csv(rec_dir, 'ions.csv'); P = _csv(rec_dir, 'positives.csv')
    assert len(P) == 652, 'the dataset is the 652 reported pairs'
    assert P.drug_idx.between(0, len(D) - 1).all()
    assert P.ion_idx.between(0, len(I) - 1).all()


def test_every_reported_pair_is_acid_plus_base(rec_dir):
    """A pair only makes a salt if the two ions carry opposite charge."""
    D = _csv(rec_dir, 'drugs.csv'); I = _csv(rec_dir, 'ions.csv'); P = _csv(rec_dir, 'positives.csv')
    ds = D.sign.values[P.drug_idx.values]; isg = I.sign.values[P.ion_idx.values]
    assert (ds * isg <= 0).all(), 'a drug and its counter-ion must not share a sign'


def test_permanent_ions_are_cations_without_a_pKa(rec_dir):
    I = _csv(rec_dir, 'ions.csv')
    perm = I[I.perm == 1]
    assert len(perm) == 85
    assert (perm.sign == 1).all(), 'a permanent ion is a quaternary cation'


# ---------------------------------------------------------------- ranking table

def test_ranks_are_contiguous_and_scores_descend(scores):
    for drug, g in scores.groupby('drug'):
        g = g.sort_values('rank')
        assert list(g['rank']) == list(range(1, len(g) + 1)), f'drug {drug}: ranks are not 1..N'
        assert g.score.is_monotonic_decreasing, f'drug {drug}: score disagrees with rank'


def test_top10_file_is_exactly_the_first_ten(rec_dir, scores):
    top = _csv(rec_dir, 'top10_recommendations_all_drugs.csv')
    expected = scores[scores['rank'] <= 10]
    assert len(top) == len(expected)
    pd.testing.assert_frame_equal(
        top[['drug', 'rank', 'ion']].sort_values(['drug', 'rank']).reset_index(drop=True),
        expected[['drug', 'rank', 'ion']].sort_values(['drug', 'rank']).reset_index(drop=True),
    )


def test_no_ion_is_offered_twice_to_one_drug(scores):
    dup = scores.groupby('drug').ion.apply(lambda s: s.duplicated().any())
    assert not dup.any(), f'duplicate candidate ions for drugs {list(dup[dup].index)}'


def test_headline_metrics_match_the_summary_file(rec_dir, scores):
    """Hit@k is counted over ALL drugs, so the 6 drugs whose reported ion is unreachable count as misses.
    Scoring only the 132 reachable drugs would read 0.773 instead of 0.739 -- this pins the honest denominator."""
    summary = _csv(rec_dir, 'leave_one_drug_out_summary.csv').set_index('metric').value
    n = scores.drug.nunique()
    assert n == int(summary['drugs']) == 138
    best = scores[scores.is_reported_pair == 1].groupby('drug')['rank'].min()
    assert len(best) == 132, 'drugs holding at least one reachable reported pair'
    for k in (1, 5, 10):
        assert (best <= k).sum() / n == pytest.approx(summary[f'Hit@{k}'], abs=5e-4)
    assert (1 / best).sum() / n == pytest.approx(summary['MRR'], abs=5e-4)


def test_score_rebuilds_from_its_four_contributions(scores):
    """The interpretability claim: score = 100 / (1 + exp(-(b0 + c2D + c3D + cpKa + cESP))), one shared b0.

    Solving for b0 on every row must give the SAME constant -- that is what makes the model additive and
    the per-ion explanation honest. Report quotes b0 = -3.74 and a mean rebuild error of 0.04 points.
    """
    c = scores[['c_2D similarity', 'c_3D similarity', 'c_pKa', 'c_ESP complementarity']].sum(axis=1).values
    p = np.clip(scores.score.values / 100, 1e-9, 1 - 1e-9)
    implied = np.log(p / (1 - p)) - c
    assert implied.std() < 0.05, 'the score is not one additive function of the four contributions'
    assert implied.mean() == pytest.approx(-3.74, abs=0.02)
    rebuilt = 100 / (1 + np.exp(-(implied.mean() + c)))
    assert np.abs(rebuilt - scores.score.values).mean() < 0.06


def test_platt_calibration_improved_the_probabilities(rec_dir):
    cal = json.load(open(os.path.join(rec_dir, 'calibration_params.json')))
    assert {'platt_a', 'platt_b'} <= set(cal)
    s = _csv(rec_dir, 'calibration_summary.csv').set_index('version')
    before, after = s.loc[s.index[0]], s.loc[s.index[1]]
    assert after.ece < before.ece, 'calibration must reduce expected calibration error'
    assert after.mean_pred == pytest.approx(after.base_rate, abs=5e-3), 'calibrated mean should match the base rate'


# ---------------------------------------------------------------- model comparison

def test_the_reported_equation_is_the_pre_registered_one(rec_dir):
    """The settings fixed before any result was seen are reported, even though three other spline
    settings score higher. If this ever flips to the best row, the pre-registration claim is void."""
    eq = _csv(rec_dir, 'results_score_equation.csv')
    fixed = eq[eq.model.str.contains('fixed')]
    assert len(fixed) == 1
    assert fixed.iloc[0].Hit10 == pytest.approx(0.717, abs=5e-4)
    assert fixed.iloc[0].Hit10 <= eq.Hit10.max(), 'the reported setting must not be the tuned best'


def test_equation_beats_xgboost_with_an_interval_clear_of_zero(rec_dir):
    eq = _csv(rec_dir, 'results_score_equation.csv')
    row = eq[eq.model.str.contains('fixed')].iloc[0]
    assert row.vs_XGBoost > 0
    assert row.vs_XGB_CI_low > 0, 'the advantage over XGBoost must exclude zero'


def test_ablation_shares_account_for_the_whole_drop(rec_dir):
    ab = _csv(rec_dir, 'ablation_equation.csv')
    assert ab['share_of_total_drop_%'].sum() == pytest.approx(100.0, abs=0.5)
    top = ab.sort_values('share_of_total_drop_%', ascending=False).iloc[0]
    assert top.family_removed == '2D similarity'
    assert top['share_of_total_drop_%'] > 50, '2D similarity carries the model'


def test_every_model_shares_one_candidate_pool(rec_dir):
    """A comparison is only fair if each model ranked the same rows."""
    m = _csv(rec_dir, 'results_setA_all_models.csv')
    assert m.n_rare.nunique() == 1, 'models were scored on different candidate pools'
    assert m[m.model == 'Random'].iloc[0].Hit10 == pytest.approx(0.225, abs=5e-4)
