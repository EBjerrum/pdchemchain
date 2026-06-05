import click
import pandas as pd

from pdchemchain import Link
from pdchemchain.io_utilities import load_dict
from pdchemchain.links import FromFile, FromSDF, StripErrors, ToFile, ToSDF
from pdchemchain.logging import logger


@click.group()
def pdchemchain():
    pass


def detect_file_format(filename: str) -> str:
    """
    Detect file format from file extension.

    Parameters
    ----------
    filename : str
        Path to the file

    Returns
    -------
    str
        Detected format: 'sdf' or 'csv'

    Notes
    -----
    Falls back to 'csv' for unknown extensions.
    """
    if filename is None:
        return "csv"

    ext = filename.lower().split(".")[-1]
    if ext in ["sdf", "sd"]:
        return "sdf"
    else:
        # Default to csv for .csv, .tsv, .txt, and unknown extensions
        return "csv"


def import_all_from_path(file_path: str) -> None:
    """
    Import all contents from a Python file into the current __main__ scope.

    Parameters:
    - file_path (str): The path to the Python file to be imported.

    Returns:
    None

    Example:
    ```python
    file_path = '/path/to/your/file.py'
    import_all_from_path(file_path)

    # Assuming your file contains variables or functions
    result = some_function_or_variable_defined_in_file()
    print(result)
    ```
    """
    with open(file_path, "rb") as file:
        code = compile(file.read(), file_path, "exec")
        exec(code, globals())


def process_data(
    in_file,
    in_format,
    out_format,
    mol_column,
    mol_from_smiles_column,
    sep,
    config_file,
    error_file,
    debug_level,
    out_file,
    custom_links,
    pd_read_options,
    pd_write_options,
):
    if pd_read_options:
        logger.info("Parsing options for file reading.")
        pd_read_options = dict(kv.split("=") for kv in pd_read_options)
        logger.debug(f"pandas read options {pd_read_options}")
    else:
        pd_read_options = {}

    if pd_write_options:
        logger.info("Parsing options for file writing.")
        pd_write_options = dict(kv.split("=") for kv in pd_write_options)
        logger.debug(f"pandas write options {pd_write_options}")
    else:
        pd_write_options = {}

    if custom_links:
        logger.info(f"Loading custom python code (links) from {custom_links}")
        import_all_from_path(custom_links)

    chain = Link.from_config_file(config_file)

    if debug_level:
        chain.set_log_level(debug_level)

    df = pd.DataFrame()

    # Auto-detect input format if not specified
    if in_file:
        if in_format is None:
            in_format = detect_file_format(in_file)
            logger.info(f"Auto-detected input format: {in_format}")

        if in_format.lower() == "csv":
            # For CSV, use sep if provided, otherwise let pandas auto-detect
            if sep and "sep" not in pd_read_options:
                pd_read_options["sep"] = sep
            elif "sep" not in pd_read_options:
                # Let pandas auto-detect separator using csv.Sniffer
                pd_read_options["sep"] = None
                logger.info("No separator specified, pandas will auto-detect")

            read_file = FromFile(in_file, mol_from_smiles_column=mol_from_smiles_column, pd_readcsv_options=pd_read_options)
        elif in_format.lower() == "sdf":
            # Extract SDF-specific options (prefixed with sdf_)
            sdf_read_opts = {
                k.replace("sdf_", ""): v
                for k, v in pd_read_options.items()
                if k.startswith("sdf_")
            }
            read_file = FromSDF(
                in_file, mol_column=mol_column, mol_from_smiles_column=mol_from_smiles_column, sdf_load_options=sdf_read_opts
            )
        else:
            raise ValueError(f"Unsupported input format: {in_format}")

        df = read_file(df)

    df = chain(df)

    if error_file:
        strip = StripErrors()
        df = strip(df)
        if not strip.error_df.empty:
            # Auto-detect error file format
            error_format = detect_file_format(error_file)
            logger.warning(
                f"{len(strip.error_df)} rows with errors found after processing, saving to {error_file} as {error_format}"
            )

            if error_format.lower() == "csv":
                strip.error_df.to_csv(error_file)
            elif error_format.lower() == "sdf":
                # For SDF, need molecule column
                if mol_column not in strip.error_df.columns:
                    raise ValueError(
                        f"Cannot write error file as SDF: molecule column '{mol_column}' not found. "
                        f"Available columns: {list(strip.error_df.columns)}. "
                        f"Please use a CSV error file instead (e.g., --error_file errors.csv)"
                    )

                # Prepare error dataframe - rename __error__ to ERROR for SDF compatibility
                # (RDKit filters out properties starting with underscore)
                error_df_copy = strip.error_df.copy()
                if "__error__" in error_df_copy.columns:
                    error_df_copy["ERROR"] = error_df_copy["__error__"]
                    error_df_copy = error_df_copy.drop(columns=["__error__"])

                # Select properties to write (exclude mol column and __log__)
                error_props = [
                    col
                    for col in error_df_copy.columns
                    if col != mol_column and col != "__log__"
                ]

                # Use ToSDF with handle_none="warn" to replace None molecules
                error_write_link = ToSDF(
                    error_file,
                    mol_column=mol_column,
                    properties=error_props,
                    handle_none="warn",  # Replace None with placeholder molecules and log
                )
                error_write_link(error_df_copy)
            else:
                strip.error_df.to_csv(error_file)

    # Auto-detect output format if not specified
    if out_file:
        if out_format is None:
            out_format = detect_file_format(out_file)
            logger.info(f"Auto-detected output format: {out_format}")

        if out_format.lower() == "csv":
            # For CSV, use sep if provided
            if sep and "sep" not in pd_write_options:
                pd_write_options["sep"] = sep

            write_file = ToFile(out_file, pd_tocsv_options=pd_write_options)
        elif out_format.lower() == "sdf":
            # Extract SDF-specific options (prefixed with sdf_)
            sdf_write_opts = {
                k.replace("sdf_", ""): v
                for k, v in pd_write_options.items()
                if k.startswith("sdf_")
            }
            write_file = ToSDF(
                out_file, mol_column=mol_column, sdf_write_options=sdf_write_opts
            )
        else:
            raise ValueError(f"Unsupported output format: {out_format}")

        write_file(df)


@pdchemchain.command()
@click.argument("config_file", type=str)
@click.option(
    "--in_file",
    default=None,
    type=click.Path(exists=True, readable=True),
    help="Input file path (CSV or SDF)",
)
@click.option(
    "--out_file",
    default=None,
    type=click.Path(writable=True),
    help="Output file path (CSV or SDF)",
)
@click.option(
    "--in_format",
    default=None,
    type=click.Choice(["csv", "sdf"], case_sensitive=False),
    help="Input format (csv or sdf). Auto-detected from extension if not specified.",
)
@click.option(
    "--out_format",
    default=None,
    type=click.Choice(["csv", "sdf"], case_sensitive=False),
    help="Output format (csv or sdf). Auto-detected from extension if not specified.",
)
@click.option(
    "--mol_column",
    default="ROMol",
    type=str,
    help='Molecule column name for SDF files (default: "ROMol")',
)
@click.option(
    "--mol_from_smiles_column",
    default=None,
    type=str,
    help='(Re)generate ROMol from the named SMILES column after loading. Works for both CSV and SDF input. '
         'For CSV: generates ROMol so pipelines built for SDF work unchanged. '
         'For SDF: substitutes the loaded 3D molecule with one from SMILES, e.g. to strip coordinates before redocking.',
)
@click.option(
    "--error_file",
    default=None,
    type=click.Path(writable=True),
    help="Error file path for rows with errors (CSV or SDF, auto-detected from extension)",
)
@click.option(
    "--sep",
    type=str,
    default=None,
    help='Separator for CSV files (default: auto-detect). Common values: "," "\\t" ";"',
)
@click.option("--debug_level", type=str, default=None, help="Optional debug level")
@click.option(
    "--custom_links",
    default=None,
    type=click.Path(exists=True, readable=True),
    help="Optional file with custom links configured as belonging to __main__ scope in config",
)
@click.option(
    "--pd_read_option",
    multiple=True,
    default=None,
    help="Extra options for file reading written as keyword=value. For SDF options, prefix with 'sdf_' (e.g., sdf_removeHs=False). Multiple options can be specified by using --pd_read_option multiple times.",
)
@click.option(
    "--pd_write_option",
    multiple=True,
    default=None,
    help="Extra options for file writing written as keyword=value. For SDF options, prefix with 'sdf_' (e.g., sdf_allNumeric=True). Multiple options can be specified by using --pd_write_option multiple times.",
)
def run(
    config_file,
    in_file,
    out_file,
    in_format,
    out_format,
    mol_column,
    mol_from_smiles_column,
    error_file,
    sep,
    debug_level,
    custom_links,
    pd_read_option,
    pd_write_option,
):
    """CONFIG_FILE: the json/yaml file with the specification for the pdchemchain chain or link.

    Examples:

    \b
    # Read SDF, process, write SDF (auto-detected)
    pdchemchain run pipeline.yaml --in_file mols.sdf --out_file results.sdf

    \b
    # Read SDF, write CSV (format conversion)
    pdchemchain run pipeline.yaml --in_file mols.sdf --out_file results.csv

    \b
    # Override auto-detection
    pdchemchain run pipeline.yaml --in_file data.txt --in_format csv --sep "\\t"

    \b
    # Custom molecule column name
    pdchemchain run pipeline.yaml --in_file mols.sdf --mol_column Molecule --out_file results.sdf

    \b
    # Pass SDF-specific options
    pdchemchain run pipeline.yaml --in_file mols.sdf --pd_read_option sdf_removeHs=False
    """
    process_data(
        in_file,
        in_format,
        out_format,
        mol_column,
        mol_from_smiles_column,
        sep,
        config_file,
        error_file,
        debug_level,
        out_file,
        custom_links,
        pd_read_option,
        pd_write_option,
    )


# TODO, make it possible to customize if version and debug level should be included
def io_config(in_config, out_config, defaults, version, loglevel):
    params = load_dict(in_config)

    # if version is None:
    #     if "__version__" in params:
    #         version = True
    #     else:
    #         version = False

    # if loglevel is None:
    #     if "__loglevel__" in params:
    #         loglevel = True
    #     else:
    #         loglevel = False

    # Check parameters for keys if the keywords in the function call is not specified
    # However We can't reliable check if a parameter object was written with the defaults switch.
    keywords = {"__version__": version, "__loglevel__": loglevel}

    for key, variable in keywords.items():
        if variable is None:
            if key in params:
                logger.debug(
                    f"Option {key.strip('_')} not specified, but found in in_config. Will include in out_config"
                )
                keywords[key] = True
            else:
                logger.debug(
                    f"Option {key.strip('_')} not specified, and not found in in_config. Will exclude in out_config"
                )
                keywords[key] = False

    chain = Link.from_params(params)

    chain.to_config_file(
        out_config, defaults=defaults, version=version, log_level=loglevel
    )


@pdchemchain.command()
@click.argument("in_config_file", type=click.Path(exists=True, readable=True))
@click.argument("out_config_file", type=click.Path(writable=True))
@click.option(
    "--defaults/--no-defaults",
    default=True,
    help="Whether to include settings which are default. Defaults to True",
)
@click.option(
    "--version/--no-version",
    default=None,
    help="Whether to include/update version info. If not specified will use setting of input config.",
)
@click.option(
    "--loglevel/--no-loglevel",
    default=None,
    help="Whether to include/exclude loglevel setting. If not specified will use setting of input config.",
)
def config(in_config_file, out_config_file, defaults, version, loglevel):
    """IN_CONFIG_FILE The json/yaml file with the input configuration
    OUT_CONFIG_FILE The json/yaml file to write"""
    io_config(in_config_file, out_config_file, defaults, version, loglevel)


if __name__ == "__main__":
    pdchemchain()
