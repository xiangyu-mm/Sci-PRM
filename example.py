from rdkit import Chem
from rdkit.Chem import AllChem

target_smiles = 'CCN(C(=O)NC(C)(C)c1ccc(F)cc1)C1CCCCC1'
target_mol = Chem.MolFromSmiles(target_smiles)

# Option A reactants
reactants_a = ['CC(C)(N=C=O)c1ccc(F)cc1', 'CCNC1CCCCC1']
mols_a = [Chem.MolFromSmiles(s) for s in reactants_a]

# Reaction: Isocyanate + Amine -> Urea
rxn = AllChem.ReactionFromSmarts('[N:1]=[C:2]=[O:3].[N:4][H:5]>>[N:4]-[C:2](=[O:3])-[N:1]-[H:5]')
products = rxn.RunReactants(mols_a)

# Check if any product matches the target
match_found = False
if products:
    for prod_set in products:
        for prod in prod_set:
            if Chem.MolToInchi(prod) == Chem.MolToInchi(target_mol):
                match_found = True
                break

print(f"Target SMILES: {target_smiles}")
print(f"Option A reactants: {reactants_a}")
print(f"Match found for Option A: {match_found}")

# Quick check of other options' complexity
options = {
    'B': ['CC(C)(C)c1ccc2c(Cl)nccc2n1', '[NH4+]'],
    'C': ['C1CNC1', 'Clc1nc2ccc(Br)cc2c(Cl)c1Cc1ccc(-n2cccn2)cc1'],
    'D': ['C1CCNC1', 'COc1nc(SC)ncc1-c1cc(C)c(Oc2ccnc(-c3cnn(C)c3)c2)cn1']
}

for opt, smiles_list in options.items():
    print(f"Option {opt} reactants: {smiles_list}")
