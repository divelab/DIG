from pyscf import gto, dft
from rdkit import Chem
from scipy.constants import physical_constants
EH2EV = physical_constants['Hartree energy in eV'][0]


def geom2gap(geom):
    mol = gto.Mole()
    mol.atom = geom
    mol.basis = '6-31G(2df,p)' #QM9
    mol.nelectron += mol.nelectron % 2 # Make it full shell? Otherwise the density matrix will be 3D
    mol.build(0, 0)

    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    mf.kernel()

    nocc = mol.nelectron // 2
    homo = mf.mo_energy[nocc - 1] * EH2EV
    lumo = mf.mo_energy[nocc] * EH2EV
    gap = lumo - homo
    return gap


def geom2alpha(geom):
    mol = gto.Mole()
    mol.atom = geom
    mol.basis = '6-31G(2df,p)' #QM9
    # mol.basis = '6-31G*' # Kddcup
    mol.nelectron += mol.nelectron % 2 # Make it full shell? Otherwise the density matrix will be 3D
    mol.build(0, 0)

    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    mf.kernel()

    polar = mf.Polarizability().polarizability()
    xx, yy, zz = polar.diagonal()
    return (xx + yy + zz) / 3


def compute_prop(atomic_number, position, prop_name):
    ptb = Chem.GetPeriodicTable()
    geom = [[ptb.GetElementSymbol(int(z)), position[i]] for i, z in enumerate(atomic_number)]

    if prop_name == 'gap':
        prop = geom2gap(geom)
    elif prop_name == 'alpha':
        prop = geom2alpha(geom)
    
    return prop