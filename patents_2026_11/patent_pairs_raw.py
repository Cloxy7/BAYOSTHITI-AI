import pandas as pd
R = []
def add(pat, url, drug, ion, where, level):
    R.append(dict(patent=pat, url=url, drug=drug, counter_ion=ion, where_in_patent=where, evidence=level))
U1 = 'https://patents.google.com/patent/US8623387B2/en'; P1 = 'US8623387B2 (MEDRx; fatty acid-based ionic liquid external preparation)'
for d, i, w in [('indomethacin', 'diisopropanolamine', 'Table 1 / tape examples'), ('indomethacin', 'diethanolamine', 'Table 3 discussion'),
                ('indomethacin', 'triisopropanolamine', 'Table 3 discussion'), ('indomethacin', 'triethanolamine', 'Table 3 discussion'),
                ('flurbiprofen', 'diisopropanolamine', 'Table 10'), ('flurbiprofen', 'lidocaine', 'Tests 151, 118, 122'),
                ('etodolac', 'diisopropanolamine', 'Table 11'), ('etodolac', 'lidocaine', 'Tests 139, 140'),
                ('ibuprofen', 'diisopropanolamine', 'Table 12'), ('loxoprofen', 'diisopropanolamine', 'Table 13'),
                ('ketoprofen', 'diisopropanolamine', 'Table 14'), ('diclofenac', 'diisopropanolamine', 'Table 15'),
                ('lidocaine', 'isostearic acid', 'Table 17'), ('lidocaine', 'capric acid', 'Tables 18-21'), ('lidocaine', 'oleic acid', 'Table 24'),
                ('tramadol', 'capric acid', 'Tables 26-30'), ('tramadol', '2-ethylhexanoic acid', 'Example 11(2)'), ('tramadol', 'benzoic acid', 'Example 11(2)'),
                ('morphine', 'capric acid', 'Tables 35-41')]:
    add(P1, U1, d, i, w, 'prepared (example or table)')
add(P1, U1, 'diclofenac', 'N-hydroxyethylpyrrolidine', 'background (Flector patch salt)', 'cited prior art')
AM = ['lidocaine', 'dibucaine', 'bupivacaine', 'diphenhydramine', 'tramadol', 'eperisone', 'tolperisone', 'dextromethorphan', 'donepezil',
      'diethanolamine', 'triethanolamine', 'diisopropanolamine', 'triisopropanolamine', '3-dimethylamino-1-propylamine']
LST = {'indomethacin': [a for a in AM if a not in ('lidocaine', 'bupivacaine')], 'ketoprofen': [a for a in AM if a != 'bupivacaine'],
       'flurbiprofen': [a for a in AM if a != 'bupivacaine'], 'diclofenac': [a for a in AM if a not in ('lidocaine', 'tramadol', 'bupivacaine')],
       'etodolac': [a for a in AM if a != 'bupivacaine'], 'loxoprofen': [a for a in AM if a != 'lidocaine']}
for d, ams in LST.items():
    for a in ams: add(P1, U1, d, a, f'list "Ionic liquid of {d}"', 'listed as an ionic liquid (not shown as prepared)')
U2 = 'https://patents.google.com/patent/AU2008334222B2/en'; P2 = 'AU2008334222B2 (MEDRx; etodolac ionic liquid tape)'
for i in ['lidocaine', 'triisopropanolamine', 'triethanolamine', 'diisopropanolamine']: add(P2, U2, 'etodolac', i, 'Example 1, Table 1', 'prepared (example or table)')
U3 = 'https://patents.google.com/patent/US20160002240A1/en'; P3 = 'US20160002240A1 (nucleoside analog salts with improved solubility)'
for i in ['choline', 'tributylmethylammonium', 'hexadecyltrimethylammonium', 'tetrabutylphosphonium', 'tetrabutylammonium', 'docusate']:
    add(P3, U3, 'acyclovir', i, 'compounds 1-7, figures', 'prepared (example or table)')
U4 = 'https://patents.google.com/patent/US8232265B2/en'; P4 = 'US8232265B2 / CA2625004C (Rogers et al.; multi-functional ionic liquid compositions)'
for d, i, w in [('lidocaine', 'docusate', 'Figs. 4A-4B'), ('sulfacetamide', 'hexadecylpyridinium', 'Fig. 1'), ('sulfacetamide', 'didecyldimethylammonium', 'Fig. 2')]:
    add(P4, U4, d, i, w, 'prepared (example or table)')
U5 = 'https://patents.google.com/patent/WO2010078300A1/en'; P5 = 'WO2010078300A1 / US9278134B2 (dual functioning ionic liquids)'
add(P5, U5, 'lidocaine', 'salicylic acid', 'Fig. 4', 'prepared (example or table)'); add(P5, U5, 'propantheline', 'p-toluenesulfonic acid', 'Fig. 7', 'prepared (example or table)')
U6 = 'https://patents.google.com/patent/WO2020218249A1/en'; P6 = 'WO2020218249A1 (lidocaine-containing patch)'
for i in ['lactic acid', 'tartaric acid', 'citric acid']: add(P6, U6, 'lidocaine', i, 'claims and examples', 'prepared (example or table)')
U7 = 'https://patents.google.com/patent/US20100029704A1/en'; P7 = 'US20100029704A1 (salt of NSAID and organic amine compound)'
for d, i in [('indomethacin', 'lidocaine'), ('diclofenac', 'lidocaine'), ('ketorolac', 'lidocaine'), ('diclofenac', 'tramadol')]:
    add(P7, U7, d, i, 'background (prior patents)', 'cited prior art')
pd.DataFrame(R).to_csv('patent_pairs_new_search.csv', index=False); print(len(R))
