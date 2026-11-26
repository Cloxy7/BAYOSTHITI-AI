import pandas as pd, re
P = pd.read_csv('../Pairs_652_with_dpKa_reliability_and_safety.csv')
txt = (P.drug.astype(str) + ' | ' + P.counter_ion.astype(str) + ' | ' + P.pair_name.astype(str) + ' | ' + P.ion_name.astype(str)).str.lower()
PAT = {'indomethacin': 'indometacin|indomethacin', 'diisopropanolamine': 'diisopropanol|bis\\(2-hydroxypropyl\\)', 'diethanolamine': 'diethanol|bis\\(2-hydroxyethyl\\)',
 'triisopropanolamine': 'triisopropanol|tris\\(2-hydroxypropyl\\)', 'triethanolamine': 'triethanol|tris\\(2-hydroxyethyl\\)', 'flurbiprofen': 'flurbiprofen',
 'lidocaine': 'lidocain', 'etodolac': 'etodolac', 'ibuprofen': 'ibuprofen', 'loxoprofen': 'loxoprofen', 'ketoprofen': 'ketoprofen', 'diclofenac': 'diclofenac',
 'isostearic acid': 'isostear', 'capric acid': 'capr|(?<!do)decanoat|(?<!do)decanoic', 'oleic acid': 'oleat|oleic', 'tramadol': 'tramadol', '2-ethylhexanoic acid': 'ethylhexano',
 'benzoic acid': 'benzoat|benzoic', 'morphine': 'morphin', 'N-hydroxyethylpyrrolidine': 'hydroxyethyl\\)?pyrrolid|epolamin', 'dibucaine': 'dibucain|cinchocain',
 'bupivacaine': 'bupivacain', 'diphenhydramine': 'diphenhydramin', 'eperisone': 'eperison', 'tolperisone': 'tolperison', 'dextromethorphan': 'dextromethorphan',
 'donepezil': 'donepezil', '3-dimethylamino-1-propylamine': 'dimethylamino-?1?-?propyl|dimethylaminopropyl', 'acyclovir': 'acyclovir|aciclovir', 'choline': 'cholin',
 'tributylmethylammonium': 'tributylmethyl|methyltributyl', 'hexadecyltrimethylammonium': 'hexadecyltrimethyl|cetyltrimethyl|cetrimon', 'tetrabutylphosphonium': 'tetrabutylphosph',
 'tetrabutylammonium': 'tetrabutylammon', 'docusate': 'docusat|dioctyl ?sulfosuccin|sulfosuccinat', 'sulfacetamide': 'sulfacetamid', 'hexadecylpyridinium': 'cetylpyrid|hexadecylpyrid',
 'didecyldimethylammonium': 'didecyldimethyl', 'salicylic acid': '(?<!acetyl)salicyl', 'propantheline': 'propanthelin', 'p-toluenesulfonic acid': 'tosyl|toluenesulfon',
 'lactic acid': 'lactat|lactic', 'tartaric acid': 'tartr', 'citric acid': 'citr', 'ketorolac': 'ketorolac'}
new = pd.read_csv('patent_pairs_new_search.csv')
def find(d, i):
    m = txt.str.contains(PAT[d]) & txt.str.contains(PAT[i]); return '; '.join(P.pair_name[m].astype(str).unique()[:3])
new['matching_pair_in_652'] = [find(d, i) for d, i in zip(new.drug, new.counter_ion)]
new['in_our_652'] = new.matching_pair_in_652 != ''
# is each molecule known to the recommender at all (so the pair could be scored)?
def known(x): return bool(txt.str.contains(PAT[x]).any())
new['drug_in_dataset'] = new.drug.map(known); new['ion_in_dataset'] = new.counter_ion.map(known)
new.to_csv('patent_pairs_new_search.csv', index=False)
print(new.groupby('evidence').in_our_652.agg(['size', 'sum']))
print(new[new.in_our_652][['drug', 'counter_ion', 'matching_pair_in_652']].to_string())
