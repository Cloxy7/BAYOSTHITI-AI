"""API-IL aqueous solubility and toxicity values found in the literature (Nov 2026 search). One row per reported value.
Values are copied from the paper text or tables as read; 'fold only' rows have no absolute number in the text."""
import pandas as pd
B = ('Bastos, J. C., Vieira, N. S. M., Gaspar, M. M., Pereiro, A. B., & Araújo, J. M. M. (2022). Human cytotoxicity, hemolytic activity, '
     'anti-inflammatory activity and aqueous solubility of ibuprofen-based ionic liquids. Sustainable Chemistry, 3(3), 358–375.', 'https://doi.org/10.3390/suschem3030023')
T = ('Tang, Y., Yang, K., Zhao, S., Chen, Q., Qin, L., & Qin, B. (2023). Evaluation of solubility, physicochemical properties, and cytotoxicity '
     'of naproxen-based ionic liquids. ACS Omega, 8, 8332–8340.', 'https://doi.org/10.1021/acsomega.2c07044')
A = ('Azevedo, A. M. O., et al. (2017). Anti-inflammatory choline based ionic liquids: insights into their lipophilicity, solubility and '
     'toxicity parameters. Journal of Molecular Liquids, 232, 20–26.', 'https://doi.org/10.1016/j.molliq.2017.02.027')
R = []
def add(drug, ion, prop, value, unit, cond, where, src, kind='API-IL'):
    R.append(dict(drug=drug, counter_ion=ion, compound_type=kind, property=prop, value=value, unit=unit, conditions=cond, where_in_paper=where, reference=src[0], doi=src[1]))
# --- solubility
add('Naproxen', 'cholinium', 'water solubility', 3.37, 'mol/L', '25 °C, water', 'Table 1', T); add('Naproxen', 'cholinium', 'water solubility', 4.32, 'mol/L', '37 °C, water', 'Table 1', T)
add('Naproxen', 'benzalkonium', 'water solubility', 1.10, 'mol/L', '25 °C, water', 'Table 1', T); add('Naproxen', 'benzalkonium', 'water solubility', 1.18, 'mol/L', '37 °C, water', 'Table 1', T)
add('Naproxen', '1-octyl-3-methylimidazolium', 'water solubility', 2.65, 'mol/L', '25 °C, water', 'Table 1', T); add('Naproxen', '1-octyl-3-methylimidazolium', 'water solubility', 3.03, 'mol/L', '37 °C, water', 'Table 1', T)
add('Naproxen', 'sodium', 'water solubility', 0.7298, 'mol/L', '25 °C, water', 'Table 1', T, 'reference salt'); add('Naproxen', '(free acid)', 'water solubility', 0.00115, 'mol/L', '25 °C, water', 'Table 1', T, 'parent drug')
add('Ibuprofen', 'cholinium', 'water solubility', 8111.43, 'mM', '37 °C, water (value cited from Santos et al., not measured by Bastos)', 'Section 3.2, ref. 24', B)
add('Ibuprofen', 'cholinium', 'solubility', 8085.57, 'mM', '37 °C, 0.1 M phosphate buffer (cited from Santos et al.)', 'Section 3.2, ref. 24', B)
add('Ibuprofen', '1-(2-hydroxyethyl)-3-methylimidazolium', 'water solubility', 9845.39, 'mM', 'water (cited from Viciosa et al.)', 'Section 3.2, ref. 44', B)
add('Ibuprofen', '1-ethyl-3-methylimidazolium', 'water solubility', 'up to ~28,000-fold above ibuprofen (absolute value only in Table 4)', '', '25 °C, water', 'Section 3.2', B)
add('Ibuprofen', '(free acid)', 'water solubility', 0.297, 'mM', '25 °C, water', 'Section 3.2', B, 'parent drug'); add('Ibuprofen', 'sodium', 'water solubility', 1761.97, 'mM', '25 °C, water', 'Section 3.2', B, 'reference salt')
add('Ketoprofen', 'cholinium', 'solubility', '58-fold above ketoprofen (absolute value only in a figure)', '', '37 °C, phosphate buffer pH 7.4', 'Fig. 4', A)
add('Naproxen', 'cholinium', 'solubility', 'about 4-fold above sodium naproxen (absolute value only in a figure)', '', '37 °C, phosphate buffer pH 7.4', 'Fig. 4', A)
# --- toxicity
add('Naproxen', 'cholinium', 'cytotoxicity IC50', 783.3, 'µM', 'L929 mouse fibroblasts, MTT', 'Cytotoxicity section, Fig. 6a', T)
add('Naproxen', 'benzalkonium', 'cytotoxicity IC50', 111.0, 'µM', 'L929 mouse fibroblasts, MTT', 'Fig. 6b', T)
add('Naproxen', '1-octyl-3-methylimidazolium', 'cytotoxicity IC50', 173.5, 'µM', 'L929 mouse fibroblasts, MTT', 'Fig. 6c', T)
for ion in ['cholinium', '1-ethyl-3-methylimidazolium', '1-(2-hydroxyethyl)-3-methylimidazolium']:
    add('Ibuprofen', ion, 'cytotoxicity EC50', '> 5.5', 'mM', 'Caco-2 human cells, 24 h, XTT', 'Sections 3.3-3.4, Table 5', B)
    add('Ibuprofen', ion, 'hemolysis', 'none detected up to 3', 'mM', 'human erythrocytes, PBS, 37 °C, 1 h', 'Section 3.4', B)
add('Ibuprofen', '(free acid)', 'cytotoxicity EC50', '> 4.0', 'mM', 'Caco-2 human cells, 24 h, XTT', 'Section 3.4', B, 'parent drug')
add('Ketoprofen', 'cholinium', 'enzyme inhibition IC50 (cytochrome c oxidase)', 21.32, 'mM', 'bovine heart CytCox, room temperature', 'Table 2', A)
add('Naproxen', 'cholinium', 'enzyme inhibition IC50 (cytochrome c oxidase)', 0.47, 'mM', 'bovine heart CytCox, room temperature', 'Table 2', A)
add('Naproxen', 'sodium', 'enzyme inhibition IC50 (cytochrome c oxidase)', 5.91, 'mM', 'bovine heart CytCox, room temperature', 'Table 2', A, 'reference salt')
D = pd.DataFrame(R); D.to_csv('API_IL_solubility_toxicity_values.csv', index=False)
api = D[D.compound_type == 'API-IL']
print(len(D), 'rows;', len(api), 'API-IL rows;', api.groupby(['drug', 'counter_ion']).ngroups, 'API-ILs;',
      (api.property.str.contains('solub')).sum(), 'solubility rows;', (~api.property.str.contains('solub')).sum(), 'toxicity rows')
