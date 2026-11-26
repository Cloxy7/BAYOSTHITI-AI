"""The leakage guards in features.py, pinned.

These are the invariants that make Hit@10 mean what the report says it means. Each one corresponds to a
guard in pair_table / group_folds; if a refactor removes one, the score goes UP and the number becomes a lie.

Needs RDKit (skipped automatically without it): pip install -r requirements.txt
"""
import numpy as np
import pytest

from conftest import needs_rdkit

pytestmark = needs_rdkit


# ---------------------------------------------------------------- folds

def test_folds_cover_every_usable_drug_exactly_once(F):
    folds = F.group_folds(seed=0, k=5)
    allocated = np.concatenate(folds)
    assert sorted(allocated) == sorted(F.use)
    assert len(set(allocated)) == len(allocated), 'a drug lands in two folds'


def test_all_forms_of_one_molecule_share_a_fold(F):
    """Ciprofloxacin appears as a cation and an anion; ethambutol is drawn twice. Splitting those across
    folds would train on one protonation state and test on another -- the same molecule, so a leak."""
    for seed in (0, 1, 7):
        folds = F.group_folds(seed=seed, k=5)
        where = {d: k for k, f in enumerate(folds) for d in f}
        for parent in set(F.PARENT[F.use]):
            same = [d for d in F.use if F.PARENT[d] == parent]
            assert len({where[d] for d in same}) == 1, \
                f'seed {seed}: parent {parent} split across folds {sorted({where[d] for d in same})}'


def test_folds_are_a_real_split_not_one_bucket(F):
    folds = F.group_folds(seed=0, k=5)
    assert len(folds) == 5
    assert all(len(f) > 0 for f in folds)


# ---------------------------------------------------------------- pair_table

def test_a_drug_alone_in_training_yields_nothing(F):
    """The sharpest form of the self-reference guard: if the only training drug IS the held-out drug,
    every reference is masked out by SIB and the ok-filter must empty the table. A non-empty result here
    would mean a drug can recommend an ion on the strength of its own record."""
    for i in F.use[:12]:
        T = F.pair_table([i], [i])
        assert len(T) == 0, f'drug {i} ({F.D.name[i]}) referenced itself'


def test_candidate_ions_always_carry_the_opposite_charge(F):
    train = F.use[:60]
    T = F.pair_table(train, F.use[60:75])
    assert (F.dsign[T.drug.values] * F.isign[T.ion.values] < 0).all()


def test_every_offered_ion_has_a_training_user(F):
    """The subtle guard: an ion nobody in training uses can only have come from the held-out drug."""
    train = set(F.use[:70]); held = F.use[70:85]
    T = F.pair_table(sorted(train), held)
    for row in T.itertuples():
        users = {k for k in train if row.ion in F.pos[k]}
        users -= set(F.SIB[row.drug])
        assert users, f'ion {row.ion} offered to drug {row.drug} with no eligible training user'


def test_sim2D_is_the_best_non_sibling_training_user(F):
    """Re-derive sim2D independently: it must be the highest Tanimoto to a TRAINING drug that uses the ion
    and is not the drug itself or another form of it. This tests the guard rather than relying on the
    happy accident that no two distinct drugs in this dataset hit Tanimoto 1.0."""
    train = F.use
    held = F.use[:10]
    T = F.pair_table(train, held)
    for row in T.itertuples():
        sibs = set(F.SIB[row.drug])
        users = [k for k in train if row.ion in F.pos[k] and k not in sibs]
        expected = max((F.S2[row.drug, k] for k in users), default=0.0)
        assert row.sim2D == pytest.approx(expected, abs=1e-9), \
            f'drug {row.drug} ion {row.ion}: sim2D {row.sim2D} != best non-sibling {expected}'


def test_similarities_stay_in_range(F):
    T = F.pair_table(F.use, F.use)
    assert (T.sim2D.between(0, 1)).all()
    assert (T.sim3D.between(0, 1)).all()
    assert T.sim2D.max() < 1.0, 'a drug matched itself (or its own other form) at similarity 1'


def test_feature_columns_are_present_and_finite(F):
    T = F.pair_table(F.use[:80], F.use[80:95])
    assert list(F.FEATS_A) == ['sim2D', 'sim3D', 'P_salt', 'dpKa', 'pKa_missing', 'perm_cation', 'ESP_comp']
    for col in F.FEATS_A:
        assert col in T.columns
        assert np.isfinite(T[col].values.astype(float)).all(), f'{col} holds NaN or inf'
    assert T.P_salt.between(0, 1).all()
    assert set(T.y.unique()) <= {0, 1}


def test_permanent_cations_get_a_full_salt_probability(F):
    """A quaternary cation needs no proton transfer, so P_salt is 1 regardless of pKa."""
    T = F.pair_table(F.use, F.use)
    assert (T[T.perm_cation == 1].P_salt == 1.0).all()
    assert (T[T.perm_cation == 1].dpKa == 10.0).all()


def test_dpka_follows_the_acid_base_convention(F):
    """dpKa = pKa(conjugate acid of the base) - pKa(the acid), whichever side the drug is on."""
    js = np.array([j for j in range(F.ni) if F.perm[j] == 0 and not np.isnan(F.ip[j])])[:40]
    acid = next(i for i in F.use if F.dsign[i] < 0)          # drug is the anion -> ion is the base
    base = next(i for i in F.use if F.dsign[i] > 0)          # drug is the cation -> ion is the acid
    np.testing.assert_allclose(F.dpka(acid, js), F.ip[js] - F.dp[acid])
    np.testing.assert_allclose(F.dpka(base, js), F.dp[base] - F.ip[js])


# ---------------------------------------------------------------- module shape

def test_load_is_cached_per_folder(rec_dir):
    import features
    assert features.load(rec_dir) is features.load(rec_dir), 'load() rebuilt the matrices on a second call'


def test_inject_populates_a_namespace(rec_dir):
    import features
    g = {}
    returned = features.inject(g, rec_dir)
    for name in ('D', 'I', 'pos', 'use', 'S2', 'PARENT', 'pair_table', 'group_folds'):
        assert name in g, f'inject() did not provide {name}'
    assert returned.D is g['D']
