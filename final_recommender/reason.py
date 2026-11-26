"""One source of truth for the plain-English reason behind each suggested counter-ion.
Used by build_app_data.py (web app), recommend_new_drug.py (new-drug notebook) and justify_pairs.py (table).
No file reads here: the caller passes the pieces it already has (safety_tag comes from the 'safety_tag' column of ions.csv).
make_reason(...) returns a dict:
  chem   : analogy + proton-transfer + driver  (the chemistry reason; no score/rank, no safety)
  full   : chem + strength + safety sentence   (one flat paragraph for a table / notebook)
  safe   : short safety phrase for a chip
"""
SAFE_PHRASE = {
    'GRAS/SCOGS listed': 'on the FDA GRAS / SCOGS safety list',
    'pharma salt former': 'an established pharmaceutical salt former',
    'biocompatible': 'biocompatible (excipient / food-derived)',
    'dual-active drug ion': 'itself an active drug ion (a dual-active pair)',
    'research-only': 'research-only, not yet pharma-acceptable',
    'unclassified': 'not classified for safety in our data',
}
GN = ['2D similarity', '3D similarity', 'pKa', 'ESP']
DRV_TXT = {'2D similarity': 'structural similarity to known pairs', '3D similarity': 'a similar 3D shape to known pairs',
           'pKa': 'the pKa match', 'ESP': 'the charge-surface match'}


def make_reason(drug_name, ion_disp, rank, ncand, score, pKa_label, dpKa_value, contrib, sim_users, n_users_total, safety_tag, is_reported):
    """contrib = [c2D, c3D, cpKa, cESP]; sim_users = list of (drug_name, sim2D) already sorted high->low; safety_tag = value from ions.csv."""
    p = []
    sim_strong = [(n, s) for n, s in (sim_users or []) if s is not None and s >= 0.2]
    if sim_strong:
        names = ', '.join('%s (%.2f)' % (n, s) for n, s in sim_strong[:3])
        p.append('Drugs similar to %s already use %s: %s.' % (drug_name, ion_disp, names))
    elif (n_users_total or 0) >= 3:
        p.append('%s is a widely used counter-ion (%d other drugs use it), though none is very similar to %s.' % (ion_disp, n_users_total, drug_name))
    else:
        p.append('Few known drugs use %s, and none is close to %s, so this suggestion rests on the chemistry below, not on analogy.' % (ion_disp, drug_name))

    lab = str(pKa_label)
    if 'permanent' in lab:
        p.append('%s is a permanent cation, so it needs no proton transfer and can pair with any acid.' % ion_disp)
    elif 'salt likely' in lab:
        p.append('A proton should transfer (ΔpKa %.1f, salt likely).' % dpKa_value)
    elif 'co-crystal' in lab:
        p.append('ΔpKa is low (%.1f), so this is more likely a co-crystal than a salt.' % dpKa_value)
    elif 'ambiguous' in lab or 'grey' in lab:
        p.append('ΔpKa is in the grey zone (%.1f), so salt vs co-crystal is uncertain.' % dpKa_value)
    else:
        p.append('The drug pKa is missing, so the proton transfer could not be scored.')

    drv = GN[max(range(4), key=lambda i: contrib[i])]
    p.append('The score is driven mainly by %s.' % DRV_TXT[drv])
    chem = ' '.join(p)
    if is_reported:
        chem = 'This pair is already published for this drug in our data. ' + chem

    strength = 'Score %.1f, rank %d of %d possible ions for %s (compare scores only within this drug).' % (score, rank, ncand, drug_name)
    safe_phrase = SAFE_PHRASE.get(str(safety_tag), 'not classified for safety in our data')
    safety_sentence = 'Safety: %s is %s.' % (ion_disp, safe_phrase)
    return {'chem': chem, 'full': chem + ' ' + strength + ' ' + safety_sentence, 'safe': safe_phrase}
