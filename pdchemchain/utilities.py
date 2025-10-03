import inspect

import pandas as pd

from pdchemchain import Link, links


class LinkToolbox:
    """Toolbox with information and references to all link classes"""

    def __init__(self):
        class_tuples = inspect.getmembers(
            links, lambda cls: inspect.isclass(cls) and issubclass(cls, Link)
        )
        classes = [class_tuple[1] for class_tuple in class_tuples]
        self._parse_classes(classes)
        self.register_main_scope_links()

    def _get_class_info(self, cls):
        info = {}
        if cls:
            if cls.__doc__:
                info["Tooltip"] = cls.__doc__.split("\n")[0]
            info["Api"] = cls.__name__ + str(inspect.signature(cls)).replace(
                " -> None", ""
            )
            info["Klass"] = cls
        return info

    def _parse_classes(self, classes):
        class_dict = {}
        for cls in classes:
            compound_key = (cls.__module__.split(".")[-1], cls.__name__)
            class_dict[compound_key] = self._get_class_info(cls)
        class_df = pd.DataFrame(class_dict).T
        class_df.index.names = ["Module", "Class"]
        self.class_df = class_df.sort_index()

    def __repr__(self):
        return repr(self.class_df)

    def _ipython_display_(self):
        max_column_width = pd.get_option("max_colwidth")
        pd.set_option("max_colwidth", 0)
        display(self.class_df[["Tooltip", "Api"]])
        pd.set_option("max_colwidth", max_column_width)

    def __getitem__(self, item):
        return self._get_info("Link").iloc[0].Klass

    def _get_info(self, class_name):
        return self.class_df.query(f"Class == '{class_name}'")

    @property
    def modules(self):
        module_list = list(set(self.class_df.index.droplevel(1)))
        return sorted(module_list)

    @property
    def class_names(self):
        class_list = list(set(self.class_df.index.droplevel(0)))
        return sorted(class_list)

    def register_class(self, cls):
        self.class_df.loc[(cls.__module__.split(".")[-1], cls.__name__), :] = (
            self._get_class_info(cls)
        )

    def register_main_scope_links(self):
        subclasses = []
        main_module = __import__("__main__")
        for name, obj in inspect.getmembers(main_module):
            if (
                inspect.isclass(obj)
                and issubclass(obj, Link)
                and obj.__module__ == "__main__"
            ):
                subclasses.append(obj)
        for cls in subclasses:
            self.register_class(cls)


def assert_import_dependency(package_name: str, install_name=None):
    try:
        # Attempt to import the package
        __import__(package_name)
        return True
    except ImportError:
        if not install_name:
            install_name = package_name
        raise ImportError(f"""The '{package_name}' package is required but not installed.
                          Please install it using 'pip install {install_name}'.""")


def get_error_placeholder_molecule():
    """
    Create a placeholder molecule for error rows in SDF files.

    This function creates a wildcard atom (*) with "ERROR" as the atom alias,
    used as a placeholder when writing error DataFrames to SDF files where the
    molecule column contains None values (e.g., when SMILES parsing failed).

    Returns
    -------
    rdkit.Chem.rdchem.Mol
        A molecule with a single wildcard atom (*) labeled as "ERROR"

    Notes
    -----
    The placeholder molecule has these properties:
    - Atomic number: 0 (wildcard/any atom)
    - MolWt: 0.00 (clearly distinguishable from real molecules)
    - Atom alias: "ERROR" (displays in molecular viewers)
    - Molecule name: "ERROR_NO_MOLECULE"

    This ensures that:
    - Error molecules are visually obvious in molecular viewers
    - Property calculations return distinguishable values (e.g., MolWt=0)
    - Downstream tools won't mistake them for real molecules

    Examples
    --------
    >>> from pdchemchain.utilities import get_error_placeholder_molecule
    >>> error_mol = get_error_placeholder_molecule()
    >>> from rdkit.Chem import Descriptors
    >>> Descriptors.MolWt(error_mol)
    0.0
    >>> error_mol.GetProp('_Name')
    'ERROR_NO_MOLECULE'
    """
    from rdkit import Chem

    # Create wildcard atom molecule
    dummy_mol = Chem.MolFromSmiles("*")
    dummy_mol.SetProp("_Name", "ERROR_NO_MOLECULE")

    # Set atom alias to "ERROR" - this displays in molecular viewers
    atom = dummy_mol.GetAtomWithIdx(0)
    atom.SetProp("atomLabel", "ERROR")
    atom.SetProp("molFileAlias", "ERROR")

    return dummy_mol
