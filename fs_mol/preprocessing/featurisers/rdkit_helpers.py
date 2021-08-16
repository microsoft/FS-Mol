"""Helper functions for things that are not in rdkit."""

from rdkit.Chem import Atom


def get_atom_symbol(atom: Atom) -> str:
    """Get the chemical symbol of the given atom with the formal charge appended to it.

    Args:
        atom: An RDKit atom for which we need the symbol.

    Returns:
        The chemical symbol of the atom with the atom's formal charge appended to it as a sequence of + or - characters.

    Examples:
        >>> atom = Atom("N")
        >>> atom.SetFormalCharge(1)
        >>> get_atom_symbol(atom)
        'N+'
        >>> atom.SetFormalCharge(-2)
        >>> get_atom_symbol(atom)
        'N--'
    """
    atom_symbol = atom.GetSymbol()
    charge = atom.GetFormalCharge()
    assert isinstance(charge, int)
    charge_symbol = ""
    abs_charge = charge
    if charge < 0:
        charge_symbol = "-"
        abs_charge *= -1
    elif charge > 0:
        charge_symbol = "+"
    charge_string = charge_symbol * abs_charge
    return atom_symbol + charge_string
