import copy
import itertools
import networkx as nx
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.Descriptors import MolLogP, qed
from rdkit.Chem.FilterCatalog import FilterCatalogParams, FilterCatalog
from .sascorer import calculateScore


def convert_radical_electrons_to_hydrogens(mol):
    """
    Convert radical electrons in a molecule into bonds to hydrogens. Only
    use this if molecule is valid. Return a new mol object.

    Args:
        mol: Rdkit mol object

    :rtype:
        Rdkit mol object
    """

    m = copy.deepcopy(mol)
    if Chem.Descriptors.NumRadicalElectrons(m) == 0:  # not a radical
        return m
    else:  # a radical
        print('converting radical electrons to H')
        for a in m.GetAtoms():
            num_radical_e = a.GetNumRadicalElectrons()
            if num_radical_e > 0:
                a.SetNumRadicalElectrons(0)
                a.SetNumExplicitHs(num_radical_e)
    return m


def check_chemical_validity(mol):
    """
    Check the chemical validity of the mol object. Existing mol object is
    not modified. Radicals pass this test.

    Args:
        mol: Rdkit mol object
    
    :rtype:
        :class:`bool`, True if chemically valid, False otherwise
    """

    s = Chem.MolToSmiles(mol, isomericSmiles=True)
    m = Chem.MolFromSmiles(s)  # implicitly performs sanitization
    if m:
        return True
    else:
        return False


def check_valency(mol):
    """
    Check that no atoms in the mol have exceeded their possible
    valency.

    Args:
        mol: Rdkit mol object

    :rtype:
        :class:`bool`, True if no valency issues, False otherwise
    """

    try:
        s = Chem.MolToSmiles(mol, isomericSmiles=True)
        m = Chem.MolFromSmiles(s)
        Chem.SanitizeMol(m, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True
    except:
        return False


def penalized_logp(mol):
    """
    Calculate the reward that consists of log p penalized by SA and # long cycles,
    as described in (Kusner et al. 2017). Scores are normalized based on the
    statistics of 250k_rndm_zinc_drugs_clean.smi dataset.

    Args:
        mol: Rdkit mol object
    
    :rtype:
        :class:`float`
    """

    # normalization constants, statistics from 250k_rndm_zinc_drugs_clean.smi
    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = MolLogP(mol)
    SA = -calculateScore(mol)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(
        Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std

    return normalized_log_p + normalized_SA + normalized_cycle


def calculate_min_plogp(mol):
    """
    Calculate the eward that consists of log p penalized by SA and # long cycles,
    as described in (Kusner et al. 2017). Scores are normalized based on the
    statistics of 250k_rndm_zinc_drugs_clean.smi dataset.

    Args:
        mol: Rdkit mol object
    
    :rtype:
        :class:`float`
    """

    p1 = penalized_logp(mol)
    s1 = Chem.MolToSmiles(mol, isomericSmiles=True)
    s2 = Chem.MolToSmiles(mol, isomericSmiles=False)
    mol1 = Chem.MolFromSmiles(s1)
    mol2 = Chem.MolFromSmiles(s2)    
    p2 = penalized_logp(mol1)
    p3 = penalized_logp(mol2)    
    final_p = min(p1, p2)
    final_p = min(final_p, p3)
    return final_p


def steric_strain_filter(mol, cutoff=0.82, max_attempts_embed=20, max_num_iters=200):
    """
    Flag molecules based on a steric energy cutoff after max_num_iters
    iterations of MMFF94 forcefield minimization. Cutoff is based on average
    angle bend strain energy of molecule

    Args:
        mol: Rdkit mol object
        cutoff (float, optional): Kcal/mol per angle . If minimized energy is above this
            threshold, then molecule fails the steric strain filter. (default: :obj:`0.82`)
        max_attempts_embed (int, optional): Number of attempts to generate initial 3d
            coordinates. (default: :obj:`20`)
        max_num_iters (int, optional): Number of iterations of forcefield minimization. (default: :obj:`200`)

    :rtype:
        :class:`bool`, True if molecule could be successfully minimized, and resulting
        energy is below cutoff, otherwise False.
    """

    # check for the trivial cases of a single atom or only 2 atoms, in which
    # case there is no angle bend strain energy (as there are no angles!)
    if mol.GetNumAtoms() <= 2:
        return True

    # make copy of input mol and add hydrogens
    m = copy.deepcopy(mol)
    m_h = Chem.AddHs(m)

    # generate an initial 3d conformer
    try:
        flag = AllChem.EmbedMolecule(m_h, maxAttempts=max_attempts_embed)
        if flag == -1:
            # print("Unable to generate 3d conformer")
            return False
    except: # to catch error caused by molecules such as C=[SH]1=C2OC21ON(N)OC(=O)NO
        # print("Unable to generate 3d conformer")
        return False

    # set up the forcefield
    AllChem.MMFFSanitizeMolecule(m_h)
    if AllChem.MMFFHasAllMoleculeParams(m_h):
        mmff_props = AllChem.MMFFGetMoleculeProperties(m_h)
        try:    # to deal with molecules such as CNN1NS23(=C4C5=C2C(=C53)N4Cl)S1
            ff = AllChem.MMFFGetMoleculeForceField(m_h, mmff_props)
        except:
            # print("Unable to get forcefield or sanitization error")
            return False
    else:
        # print("Unrecognized atom type")
        return False

    # minimize steric energy
    try:
        ff.Minimize(maxIts=max_num_iters)
    except:
        # print("Minimization error")
        return False

    # ### debug ###
    # min_e = ff.CalcEnergy()
    # print("Minimized energy: {}".format(min_e))
    # ### debug ###

    # get the angle bend term contribution to the total molecule strain energy
    mmff_props.SetMMFFBondTerm(False)
    mmff_props.SetMMFFAngleTerm(True)
    mmff_props.SetMMFFStretchBendTerm(False)
    mmff_props.SetMMFFOopTerm(False)
    mmff_props.SetMMFFTorsionTerm(False)
    mmff_props.SetMMFFVdWTerm(False)
    mmff_props.SetMMFFEleTerm(False)

    ff = AllChem.MMFFGetMoleculeForceField(m_h, mmff_props)

    min_angle_e = ff.CalcEnergy()
    # print("Minimized angle bend energy: {}".format(min_angle_e))

    # find number of angles in molecule
    # from molecule... This is too hacky
    num_atoms = m_h.GetNumAtoms()
    atom_indices = range(num_atoms)
    angle_atom_triplets = itertools.permutations(atom_indices, 3)  # get all
    # possible 3 atom indices groups. Currently, each angle is represented by
    #  2 duplicate groups. Should remove duplicates here to be more efficient
    double_num_angles = 0
    for triplet in list(angle_atom_triplets):
        if mmff_props.GetMMFFAngleBendParams(m_h, *triplet):
            double_num_angles += 1
    num_angles = double_num_angles / 2  # account for duplicate angles

    # print("Number of angles: {}".format(num_angles))

    avr_angle_e = min_angle_e / num_angles

    if avr_angle_e < cutoff:
        return True
    else:
        return False


### YES/NO filters ###
def zinc_molecule_filter(mol):
    """
    Flag molecules based on problematic functional groups as
    provided set of ZINC rules from
    http://blaster.docking.org/filtering/rules_default.txt.

    Args:
        mol: Rdkit mol object
    
    :rtype:
        :class:`bool`, returns True if molecule is okay (ie does not match any of
        therules), False if otherwise.
    """
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.ZINC)
    catalog = FilterCatalog(params)
    return not catalog.HasMatch(mol)


def reward_target_molecule_similarity(mol, target, radius=2, nBits=2048,
                                      useChirality=True):
    """
    Calculate the similarity, based on tanimoto similarity
    between the ECFP fingerprints of the x molecule and target molecule.

    Args:
        mol: Rdkit mol object
        target: Rdkit mol object
    
    :rtype:
        :class:`float`, [0.0, 1.0]
    """
    x = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, useChirality=useChirality)
    target = rdMolDescriptors.GetMorganFingerprintAsBitVect(target, radius=radius, nBits=nBits, useChirality=useChirality)
    return DataStructs.TanimotoSimilarity(x, target)