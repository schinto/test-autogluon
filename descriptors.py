import numpy as np
import networkx as nx
from itertools import combinations

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.ML.Descriptors import MoleculeDescriptors

nms = [
    "MinAbsPartialCharge",
    "NumRadicalElectrons",
    "HeavyAtomMolWt",
    "MaxAbsEStateIndex",
    "MaxAbsPartialCharge",
    "MaxEStateIndex",
    "MinPartialCharge",
    "ExactMolWt",
    "MolWt",
    "NumValenceElectrons",
    "MinEStateIndex",
    "MinAbsEStateIndex",
    "MaxPartialCharge",
    "BalabanJ",
    "BertzCT",
    "Chi0",
    "Chi0n",
    "Chi0v",
    "Chi1",
    "Chi1n",
    "Chi1v",
    "Chi2n",
    "Chi2v",
    "Chi3n",
    "Chi3v",
    "Chi4n",
    "Chi4v",
    "HallKierAlpha",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "LabuteASA",
    "PEOE_VSA1",
    "PEOE_VSA10",
    "PEOE_VSA11",
    "PEOE_VSA12",
    "PEOE_VSA13",
    "PEOE_VSA14",
    "PEOE_VSA2",
    "PEOE_VSA3",
    "PEOE_VSA4",
    "PEOE_VSA5",
    "PEOE_VSA6",
    "PEOE_VSA7",
    "PEOE_VSA8",
    "PEOE_VSA9",
    "SMR_VSA1",
    "SMR_VSA10",
    "SMR_VSA2",
    "SMR_VSA3",
    "SMR_VSA4",
    "SMR_VSA5",
    "SMR_VSA6",
    "SMR_VSA7",
    "SMR_VSA8",
    "SMR_VSA9",
    "SlogP_VSA1",
    "SlogP_VSA10",
    "SlogP_VSA11",
    "SlogP_VSA12",
    "SlogP_VSA2",
    "SlogP_VSA3",
    "SlogP_VSA4",
    "SlogP_VSA5",
    "SlogP_VSA6",
    "SlogP_VSA7",
    "SlogP_VSA8",
    "SlogP_VSA9",
    "TPSA",
    "EState_VSA1",
    "EState_VSA10",
    "EState_VSA11",
    "EState_VSA2",
    "EState_VSA3",
    "EState_VSA4",
    "EState_VSA5",
    "EState_VSA6",
    "EState_VSA7",
    "EState_VSA8",
    "EState_VSA9",
    "VSA_EState1",
    "VSA_EState10",
    "VSA_EState2",
    "VSA_EState3",
    "VSA_EState4",
    "VSA_EState5",
    "VSA_EState6",
    "VSA_EState7",
    "VSA_EState8",
    "VSA_EState9",
    "FractionCSP3",
    "HeavyAtomCount",
    "NHOHCount",
    "NOCount",
    "NumAliphaticCarbocycles",
    "NumAliphaticHeterocycles",
    "NumAliphaticRings",
    "NumAromaticCarbocycles",
    "NumAromaticHeterocycles",
    "NumAromaticRings",
    "NumHAcceptors",
    "NumHDonors",
    "NumHeteroatoms",
    "NumRotatableBonds",
    "NumSaturatedCarbocycles",
    "NumSaturatedHeterocycles",
    "NumSaturatedRings",
    "RingCount",
    "MolLogP",
    "MolMR",
    "fr_Al_COO",
    "fr_Al_OH",
    "fr_Al_OH_noTert",
    "fr_ArN",
    "fr_Ar_COO",
    "fr_Ar_N",
    "fr_Ar_NH",
    "fr_Ar_OH",
    "fr_COO",
    "fr_COO2",
    "fr_C_O",
    "fr_C_O_noCOO",
    "fr_C_S",
    "fr_HOCCN",
    "fr_Imine",
    "fr_NH0",
    "fr_NH1",
    "fr_NH2",
    "fr_N_O",
    "fr_Ndealkylation1",
    "fr_Ndealkylation2",
    "fr_Nhpyrrole",
    "fr_SH",
    "fr_aldehyde",
    "fr_alkyl_carbamate",
    "fr_alkyl_halide",
    "fr_allylic_oxid",
    "fr_amide",
    "fr_amidine",
    "fr_aniline",
    "fr_aryl_methyl",
    "fr_azide",
    "fr_azo",
    "fr_barbitur",
    "fr_benzene",
    "fr_benzodiazepine",
    "fr_bicyclic",
    "fr_diazo",
    "fr_dihydropyridine",
    "fr_epoxide",
    "fr_ester",
    "fr_ether",
    "fr_furan",
    "fr_guanido",
    "fr_halogen",
    "fr_hdrzine",
    "fr_hdrzone",
    "fr_imidazole",
    "fr_imide",
    "fr_isocyan",
    "fr_isothiocyan",
    "fr_ketone",
    "fr_ketone_Topliss",
    "fr_lactam",
    "fr_lactone",
    "fr_methoxy",
    "fr_morpholine",
    "fr_nitrile",
    "fr_nitro",
    "fr_nitro_arom",
    "fr_nitro_arom_nonortho",
    "fr_nitroso",
    "fr_oxazole",
    "fr_oxime",
    "fr_para_hydroxylation",
    "fr_phenol",
    "fr_phenol_noOrthoHbond",
    "fr_phos_acid",
    "fr_phos_ester",
    "fr_piperdine",
    "fr_piperzine",
    "fr_priamide",
    "fr_prisulfonamd",
    "fr_pyridine",
    "fr_quatN",
    "fr_sulfide",
    "fr_sulfonamd",
    "fr_sulfone",
    "fr_term_acetylene",
    "fr_tetrazole",
    "fr_thiazole",
    "fr_thiocyan",
    "fr_thiophene",
    "fr_unbrch_alkane",
    "fr_urea",
]
rdkit_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(nms)


def calc_rdkit_properties(mols):
    calc_desc = [rdkit_calculator.CalcDescriptors(mol) for mol in mols]
    calc_desc = np.asarray(calc_desc)
    calc_desc[~np.isfinite(calc_desc)] = 0
    return calc_desc, [f"RDKIT:{x}" for x in nms]


def calc_morgan_fingerprints(mols, nBits=1024, radius=3):
    calc_fp = []
    for mol in mols:
        arr = np.zeros((1,))
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius, useFeatures=True, nBits=nBits
        )
        DataStructs.ConvertToNumpyArray(fp, arr)
        calc_fp.append(arr)
    calc_fp = np.asarray(calc_fp)
    calc_fp[~np.isfinite(calc_fp)] = 0
    return calc_fp, [f"MORGANF_{radius}_{nBits}:{x}" for x in range(nBits)]


def calc_MACCS_keys_fingerprint(mols):
    calc_fp = []
    for mol in mols:
        arr = np.zeros((1,))
        fp = AllChem.GetMACCSKeysFingerprint(mol)
        DataStructs.ConvertToNumpyArray(fp, arr)
        calc_fp.append(arr)
    calc_fp = np.asarray(calc_fp)
    nBits = calc_fp[0].size
    return calc_fp, [f"MACCS:{x}" for x in range(nBits)]


def flat_ring_system(mol):
    if mol is None:
        print("Error: Molecule is None")
        return np.nan

    # Loop over all rings and remember flat ones
    # A ring is considered flat when all atoms have SP2 hybridization
    rings = mol.GetRingInfo().AtomRings()
    flat_rings = []
    for ring in rings:
        is_flat = all(
            [
                mol.GetAtomWithIdx(a).GetHybridization()
                == Chem.rdchem.HybridizationType.SP2
                for a in ring
            ]
        )
        if is_flat:
            flat_rings.append(ring)

    # Generate the connectivity Graph1 for fused rings
    #   Two rings are considered fused if they share exactly 2 common atoms
    # Also Generate the connectivity Graph2 for flat-single-flat rings
    #  Two rings are considered flat-single-flat if there is exactly one single bond between their atoms
    #  and they dont have any atoms in common
    G1 = nx.path_graph(0)
    G2 = nx.path_graph(0)
    for combo in combinations(range(len(flat_rings)), 2):
        common_atoms = set(flat_rings[combo[0]]).intersection(set(flat_rings[combo[1]]))

        # if there are exactly 2 shared atoms between two rings -> fused
        if len(common_atoms) == 2:
            nx.add_path(G1, combo)

        # if there are no shared atoms
        if len(common_atoms) == 0:
            bonds_between = []
            # Get all bonds between all atoms in both rings
            for a1 in flat_rings[combo[0]]:
                for a2 in flat_rings[combo[1]]:
                    bond = mol.GetBondBetweenAtoms(a1, a2)
                    if bond is not None:
                        bonds_between.append(bond)
            # if there is exactly 1 bond
            if len(bonds_between) == 1:
                # it needs to be a single bond
                if bonds_between[0].GetBondType() == Chem.rdchem.BondType.SINGLE:
                    nx.add_path(G2, combo)

    # connected components does the clustering and returns groups of indices corresponding to the flat ring indices
    clusters1 = list(
        nx.connected_components(G1)
    )  # has to be list because these are generators
    clusters2 = list(nx.connected_components(G2))  # that need to be used multiple times
    fused_lengths = [len(x) for x in clusters1]  # length of fused flat ring clusters

    # Single flat rings have to be detected separately because they dont count as connected component
    rings_in_fused = set([ring for cls in clusters1 for ring in cls])
    nSingleRings = len(
        [ring for ring, _ in enumerate(flat_rings) if ring not in rings_in_fused]
    )
    Num_FlatSingleFlat = np.sum([len(x) - 1 for x in clusters2], dtype=np.int32)

    Num_AtomsInFlatRings = len(set([a for ring in flat_rings for a in ring]))
    Ratio_AtomsInFlatRings = Num_AtomsInFlatRings / mol.GetNumAtoms()

    Num_2_Fused_Flat_Rings = np.sum([r == 2 for r in fused_lengths], dtype=np.int32)
    Num_3_Fused_Flat_Rings = np.sum([r == 3 for r in fused_lengths], dtype=np.int32)
    Num_Greater_3_Fused_Flat_Rings = np.sum(
        [r > 3 for r in fused_lengths], dtype=np.int32
    )

    # Total number of conjugated bonds
    conjugated = []
    for i in range(mol.GetNumBonds()):
        conjugated.append(mol.GetBondWithIdx(i).GetIsConjugated())
    Num_ConjugatedBonds = sum(conjugated)

    # Number of conjugated single and double bonds
    Num_ConjugatedSingleBonds = 0
    Num_ConjugatedDoubleBonds = 0
    for i in range(mol.GetNumBonds()):
        bond = mol.GetBondWithIdx(i)
        if bond.GetIsConjugated():
            if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                Num_ConjugatedSingleBonds += 1
            if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                Num_ConjugatedDoubleBonds += 1

    # Return results
    res = {
        "Ratio_AtomsInFlatRings": Ratio_AtomsInFlatRings,
        "Num_AtomsInFlatRings": Num_AtomsInFlatRings,
        "Num_1_Flat_Rings": nSingleRings,
        "Num_2_Fused_Flat_Rings": Num_2_Fused_Flat_Rings,
        "Num_3_Fused_Flat_Rings": Num_3_Fused_Flat_Rings,
        "Num_Greater_3_Fused_Flat_Rings": Num_Greater_3_Fused_Flat_Rings,
        "Num_FlatSingleFlat": Num_FlatSingleFlat,
        "Num_ConjugatedBonds": Num_ConjugatedBonds,
        "Num_ConjugatedSingleBonds": Num_ConjugatedSingleBonds,
        "Num_ConjugatedDoubleBonds": Num_ConjugatedDoubleBonds,
    }
    return res


def calc_flat_ring_properties(mols):
    calc_desc = []
    names = []
    for mol in mols:
        desc = flat_ring_system(mol)
        names = desc.keys()
        calc_desc.append(list(desc.values()))
    calc_desc = np.asarray(calc_desc)
    return calc_desc, [f"FLATRING:{x}" for x in names]
