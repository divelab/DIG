# Code adapted from chainer_chemistry\dataset\preprocessors\common
from rdkit import Chem
import numpy
from rdkit import Chem
from rdkit.Chem import rdmolops


class GGNNPreprocessor(object):
    """GGNN Preprocessor

    Args:
        max_atoms (int): Max number of atoms for each molecule, if the
            number of atoms is more than this value, this data is simply
            ignored.
            Setting negative value indicates no limit for max atoms.
        out_size (int): It specifies the size of array returned by
            `get_input_features`.
            If the number of atoms in the molecule is less than this value,
            the returned arrays is padded to have fixed size.
            Setting negative value indicates do not pad returned array.
        add_Hs (bool): If True, implicit Hs are added.
        kekulize (bool): If True, Kekulizes the molecule.

    """

    def __init__(self, max_atoms=-1, out_size=-1, add_Hs=False,
                 kekulize=False):
        super(GGNNPreprocessor, self).__init__()
        self.add_Hs = add_Hs
        self.kekulize = kekulize

        if max_atoms >= 0 and out_size >= 0 and max_atoms > out_size:
            raise ValueError('max_atoms {} must be less or equal to '
                             'out_size {}'.format(max_atoms, out_size))
        self.max_atoms = max_atoms
        self.out_size = out_size

    def get_input_features(self, mol):
        """get input features

        Args:
            mol (Mol):

        Returns:

        """
        type_check_num_atoms(mol, self.max_atoms)
        atom_array = construct_atomic_number_array(mol, out_size=self.out_size)
        adj_array = construct_discrete_edge_matrix(mol, out_size=self.out_size)
        return atom_array, adj_array

    def prepare_smiles_and_mol(self, mol):
        """Prepare `smiles` and `mol` used in following preprocessing.

        This method is called before `get_input_features` is called, by parser
        class.
        This method may be overriden to support custom `smile`/`mol` extraction

        Args:
            mol (mol): mol instance

        Returns (tuple): (`smiles`, `mol`)
        """
        # Note that smiles expression is not unique.
        # we obtain canonical smiles which is unique in `mol`
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False,
                                            canonical=True)
        mol = Chem.MolFromSmiles(canonical_smiles)
        if self.add_Hs:
            mol = Chem.AddHs(mol)
        if self.kekulize:
            Chem.Kekulize(mol)
        return canonical_smiles, mol

    def get_label(self, mol, label_names=None):
        """Extracts label information from a molecule.

        This method extracts properties whose keys are
        specified by ``label_names`` from a molecule ``mol``
        and returns these values as a list.
        The order of the values is same as that of ``label_names``.
        If the molecule does not have a
        property with some label, this function fills the corresponding
        index of the returned list with ``None``.

        Args:
            mol (rdkit.Chem.Mol): molecule whose features to be extracted
            label_names (None or iterable): list of label names.

        Returns:
            list of str: label information. Its length is equal to
            that of ``label_names``. If ``label_names`` is ``None``,
            this function returns an empty list.

        """
        if label_names is None:
            return []

        label_list = []
        for label_name in label_names:
            if mol.HasProp(label_name):
                label_list.append(mol.GetProp(label_name))
            else:
                label_list.append(None)

                # TODO(Nakago): Review implementation
                # Label -1 work in case of classification.
                # However in regression, assign -1 is not a good strategy...
                # label_list.append(-1)

                # Failed to GetProp for label, skip this case.
                # print('no label')
                # raise MolFeatureExtractionError

        return label_list


class MolFeatureExtractionError(Exception):
    pass


# --- Type check ---
def type_check_num_atoms(mol, num_max_atoms=-1):
    """Check number of atoms in `mol` does not exceed `num_max_atoms`

    If number of atoms in `mol` exceeds the number `num_max_atoms`, it will
    raise `MolFeatureExtractionError` exception.

    Args:
        mol (Mol):
        num_max_atoms (int): If negative value is set, not check number of
            atoms.

    """
    num_atoms = mol.GetNumAtoms()
    if num_max_atoms >= 0 and num_atoms > num_max_atoms:
        # Skip extracting feature. ignore this case.
        raise MolFeatureExtractionError(
            'Number of atoms in mol {} exceeds num_max_atoms {}'
            .format(num_atoms, num_max_atoms))


# --- Atom preprocessing ---
def construct_atomic_number_array(mol, out_size=-1):
    """Returns atomic numbers of atoms consisting a molecule.

    Args:
        mol (rdkit.Chem.Mol): Input molecule.
        out_size (int): The size of returned array.
            If this option is negative, it does not take any effect.
            Otherwise, it must be larger than the number of atoms
            in the input molecules. In that case, the tail of
            the array is padded with zeros.

    Returns:
        numpy.ndarray: an array consisting of atomic numbers
            of atoms in the molecule.
    """

    atom_list = [a.GetAtomicNum() for a in mol.GetAtoms()]
    n_atom = len(atom_list)

    if out_size < 0:
        return numpy.array(atom_list, dtype=numpy.int32)
    elif out_size >= n_atom:
        # 'empty' padding for atom_list
        # 0 represents empty place for atom
        atom_array = numpy.zeros(out_size, dtype=numpy.int32)
        atom_array[:n_atom] = numpy.array(atom_list, dtype=numpy.int32)
        return atom_array
    else:
        raise ValueError('`out_size` (={}) must be negative or '
                         'larger than or equal to the number '
                         'of atoms in the input molecules (={})'
                         '.'.format(out_size, n_atom))


# --- Adjacency matrix preprocessing ---
def construct_adj_matrix(mol, out_size=-1, self_connection=True):
    """Returns the adjacent matrix of the given molecule.

    This function returns the adjacent matrix of the given molecule.
    Contrary to the specification of
    :func:`rdkit.Chem.rdmolops.GetAdjacencyMatrix`,
    The diagonal entries of the returned matrix are all-one.

    Args:
        mol (rdkit.Chem.Mol): Input molecule.
        out_size (int): The size of the returned matrix.
            If this option is negative, it does not take any effect.
            Otherwise, it must be larger than the number of atoms
            in the input molecules. In that case, the adjacent
            matrix is expanded and zeros are padded to right
            columns and bottom rows.
        self_connection (bool): Add self connection or not.
            If True, diagonal element of adjacency matrix is filled with 1.

    Returns:
        adj_array (numpy.ndarray): The adjacent matrix of the input molecule.
            It is 2-dimensional array with shape (atoms1, atoms2), where
            atoms1 & atoms2 represent from and to of the edge respectively.
            If ``out_size`` is non-negative, the returned
            its size is equal to that value. Otherwise,
            it is equal to the number of atoms in the the molecule.
    """

    adj = rdmolops.GetAdjacencyMatrix(mol)
    s0, s1 = adj.shape
    if s0 != s1:
        raise ValueError('The adjacent matrix of the input molecule'
                         'has an invalid shape: ({}, {}). '
                         'It must be square.'.format(s0, s1))

    if self_connection:
        adj = adj + numpy.eye(s0)
    if out_size < 0:
        adj_array = adj.astype(numpy.float32)
    elif out_size >= s0:
        adj_array = numpy.zeros((out_size, out_size),
                                dtype=numpy.float32)
        adj_array[:s0, :s1] = adj
    else:
        raise ValueError(
            '`out_size` (={}) must be negative or larger than or equal to the '
            'number of atoms in the input molecules (={}).'
            .format(out_size, s0))
    return adj_array


def construct_discrete_edge_matrix(mol, out_size=-1):
    """Returns the edge-type dependent adjacency matrix of the given molecule.

    Args:
        mol (rdkit.Chem.Mol): Input molecule.
        out_size (int): The size of the returned matrix.
            If this option is negative, it does not take any effect.
            Otherwise, it must be larger than the number of atoms
            in the input molecules. In that case, the adjacent
            matrix is expanded and zeros are padded to right
            columns and bottom rows.

    Returns:
        adj_array (numpy.ndarray): The adjacent matrix of the input molecule.
            It is 3-dimensional array with shape (edge_type, atoms1, atoms2),
            where edge_type represents the bond type,
            atoms1 & atoms2 represent from and to of the edge respectively.
            If ``out_size`` is non-negative, its size is equal to that value.
            Otherwise, it is equal to the number of atoms in the the molecule.
    """
    if mol is None:
        raise MolFeatureExtractionError('mol is None')
    N = mol.GetNumAtoms()

    if out_size < 0:
        size = N
    elif out_size >= N:
        size = out_size
    else:
        raise ValueError(
            'out_size {} is smaller than number of atoms in mol {}'
            .format(out_size, N))
    adjs = numpy.zeros((4, size, size), dtype=numpy.float32)

    bond_type_to_channel = {
        Chem.BondType.SINGLE: 0,
        Chem.BondType.DOUBLE: 1,
        Chem.BondType.TRIPLE: 2,
        Chem.BondType.AROMATIC: 3
    }
    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()
        ch = bond_type_to_channel[bond_type]
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adjs[ch, i, j] = 1.0
        adjs[ch, j, i] = 1.0
    return adjs
