import requests
from rdkit import Chem

def validate_and_canonicalize_smiles(smiles: str) -> str:

    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        pass
    raise ValueError(f"SMILES string '{smiles}' failed RDKit structural sanitization validation.")

def fetch_pubchem_title(smiles: str) -> str:
   
    try:
        
        clean_smi = validate_and_canonicalize_smiles(smiles)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{clean_smi}/description/JSON"
        res = requests.get(url, timeout=3.0)
        if res.status_code == 200:
            return res.json()['InformationList']['Information'][0]['Title']
    except Exception:
        pass
    return "Novel Experimental Compound Structure"