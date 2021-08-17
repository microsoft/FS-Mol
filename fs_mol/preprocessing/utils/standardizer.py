## These MolStandardizer classes are due to Paolo Tosco
## They ensure that a sequence of standardization operations are applied
## https://gist.github.com/ptosco/7e6b9ab9cc3e44ba0919060beaed198e

import logging

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize


class BaseLogger:
    """
    Simple logging base class.

    Inherit from this class and call self.get_logger() to
    get a logger bearing the class name.
    """

    DEFAULT_LOG_LEVEL = logging.WARNING

    def __init__(self):
        self._log_level = self.DEFAULT_LOG_LEVEL

    def set_log_level(self, log_level):
        if not getattr(logging, log_level):
            raise TypeError(f"log_level {log_level} does not exist in logging")
        self._log_level = log_level

    def get_logger(self):
        """Return a logger bearing the class name."""
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[%(asctime)s:%(name)s:%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(self._log_level)
        return logger


class Standardizer(BaseLogger):
    """
    Simple wrapper class around rdkit Standardizer.
    """

    DEFAULT_CANON_TAUT = False
    DEFAULT_METAL_DISCONNECT = False
    MAX_TAUTOMERS = 100
    MAX_TRANSFORMS = 100
    MAX_RESTARTS = 200
    PREFER_ORGANIC = True

    def __init__(
        self,
        metal_disconnect=None,
        canon_taut=None,
    ):
        """
        Constructor.

        All parameters are optional.
        :param metal_disconnect:    if True, metallorganic complexes are
                                    disconnected
        :param canon_taut:          if True, molecules are converted to their
                                    canonical tautomer
        """
        super().__init__()
        if metal_disconnect is None:
            metal_disconnect = self.DEFAULT_METAL_DISCONNECT
        if canon_taut is None:
            canon_taut = self.DEFAULT_CANON_TAUT
        self._canon_taut = canon_taut
        self._metal_disconnect = metal_disconnect
        self._taut_enumerator = None
        self._rdlogger = None
        self._uncharger = None
        self._lfrag_chooser = None
        self._metal_disconnector = None
        self._normalizer = None
        self._reionizer = None
        self._params = None

    @property
    def params(self):
        """Return the MolStandardize CleanupParameters."""
        if self._params is None:
            self._params = rdMolStandardize.CleanupParameters()
            self._params.maxTautomers = self.MAX_TAUTOMERS
            self._params.maxTransforms = self.MAX_TRANSFORMS
            self._params.maxRestarts = self.MAX_RESTARTS
            self._params.preferOrganic = self.PREFER_ORGANIC
            self._params.tautomerRemoveSp3Stereo = False
        return self._params

    @property
    def canon_taut(self):
        """Return whether tautomer canonicalization will be done."""
        return self._canon_taut

    @property
    def metal_disconnect(self):
        """Return whether metallorganic complexes will be disconnected."""
        return self._metal_disconnect

    @property
    def taut_enumerator(self):
        """Return the TautomerEnumerator object."""
        if self._taut_enumerator is None:
            self._taut_enumerator = rdMolStandardize.TautomerEnumerator(self.params)
        return self._taut_enumerator

    @property
    def uncharger(self):
        """Return the Uncharger object."""
        if self._uncharger is None:
            self._uncharger = rdMolStandardize.Uncharger()
        return self._uncharger

    @property
    def lfrag_chooser(self):
        """Return the LargestFragmentChooser object."""
        if self._lfrag_chooser is None:
            self._lfrag_chooser = rdMolStandardize.LargestFragmentChooser(self.params.preferOrganic)
        return self._lfrag_chooser

    @property
    def metal_disconnector(self):
        """Return the MetalDisconnector object."""
        if self._metal_disconnector is None:
            self._metal_disconnector = rdMolStandardize.MetalDisconnector()
        return self._metal_disconnector

    @property
    def normalizer(self):
        """Return the Normalizer object."""
        if self._normalizer is None:
            self._normalizer = rdMolStandardize.Normalizer(
                self.params.normalizationsFile, self.params.maxRestarts
            )
        return self._normalizer

    @property
    def reionizer(self):
        """Return the Reionizer object."""
        if self._reionizer is None:
            self._reionizer = rdMolStandardize.Reionizer(self.params.acidbaseFile)
        return self._reionizer

    def charge_parent(self, mol_in):
        """Sequentially apply a series of MolStandardize operations:

        * MetalDisconnector
        * Normalizer
        * Reionizer
        * LargestFragmentChooser
        * Uncharger

        The net result is that a desalted, normalized, neutral
        molecule with implicit Hs is returned.
        """
        params = Chem.RemoveHsParameters()
        params.removeAndTrackIsotopes = True
        mol_in = Chem.RemoveHs(mol_in, params, sanitize=False)
        if self._metal_disconnect:
            mol_in = self.metal_disconnector.Disconnect(mol_in)
        normalized = self.normalizer.normalize(mol_in)
        Chem.SanitizeMol(normalized)
        normalized = self.reionizer.reionize(normalized)
        Chem.AssignStereochemistry(normalized)
        normalized = self.lfrag_chooser.choose(normalized)
        normalized = self.uncharger.uncharge(normalized)
        # need this to reassess aromaticity on things like
        # cyclopentadienyl, tropylium, azolium, etc.
        Chem.SanitizeMol(normalized)
        return Chem.RemoveHs(Chem.AddHs(normalized))

    def standardize_mol(self, mol_in):
        """
        Standardize a single molecule.

        :param mol_in:  a Chem.Mol
        :return:        * (standardized Chem.Mol, n_taut) tuple
                          if success. n_taut will be negative if
                          tautomer enumeration was aborted due
                          to reaching a limit
                        * (None, error_msg) if failure

        This calls self.charge_parent() and, if self._canon_taut
        is True, runs tautomer canonicalization.
        """
        logger = self.get_logger()
        if self._rdlogger is None:
            self._rdlogger = RDLogger.logger()
            self._rdlogger.setLevel(RDLogger.CRITICAL)
        n_tautomers = 0
        if isinstance(mol_in, Chem.Mol):
            name = None
            try:
                name = mol_in.GetProp("_Name")
            except KeyError:
                pass
            if not name:
                name = "NONAME"
        else:
            error = f"Expected SMILES or Chem.Mol as input, got {str(type(mol_in))}"
            logger.critical(error)
            return None, error
        try:
            mol_out = self.charge_parent(mol_in)
        except Exception as e:
            error = f"charge_parent FAILED: {str(e).strip()}"
            logger.critical(error)
            return None, error
        if self._canon_taut:
            try:
                res = self.taut_enumerator.Enumerate(mol_out, False)
            except TypeError:
                # we are still on the pre-2021 RDKit API
                res = self.taut_enumerator.Enumerate(mol_out)
            except Exception as e:
                # something else went wrong
                error = f"canon_taut FAILED: {str(e).strip()}"
                logger.critical(error)
                return None, error
            n_tautomers = len(res)
            if hasattr(res, "status"):
                completed = res.status == rdMolStandardize.TautomerEnumeratorStatus.Completed
            else:
                # we are still on the pre-2021 RDKit API
                completed = len(res) < 1000
            if not completed:
                n_tautomers = -n_tautomers
            try:
                mol_out = self.taut_enumerator.PickCanonical(res)
            except AttributeError:
                # we are still on the pre-2021 RDKit API
                mol_out = max([(self.taut_enumerator.ScoreTautomer(m), m) for m in res])[1]
            except Exception as e:
                # something else went wrong
                error = f"canon_taut FAILED: {str(e).strip()}"
                logger.critical(error)
                return None, error
        mol_out.SetProp("_Name", name)
        return mol_out, n_tautomers
