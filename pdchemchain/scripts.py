import click
import pandas as pd

from pdchemchain import Link
from pdchemchain.io_utilities import load_dict
from pdchemchain.links import FromFile, StripErrors, ToFile
from pdchemchain.logging import logger


@click.group()
def pdchemchain():
    pass


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
        logger.info("Parsing options for csv reading.")
        pd_read_options = dict(kv.split("=") for kv in pd_read_options)
        logger.debug(f"pandas read options {pd_read_options}")
    else:
        pd_read_options = {}

    if pd_write_options:
        logger.info("Parsing options for csv writing.")
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

    if in_file:
        read_file = FromFile(in_file, pd_readcsv_options=pd_read_options)
        df = read_file(df)

    df = chain(df)

    if error_file:
        strip = StripErrors()
        df = strip(df)
        if not strip.error_df.empty:
            logger.warning(
                f"{len(strip.has_errors)} rows with errors found after processing, saving to {error_file}"
            )
            strip.error_df.to_csv(error_file)

    if out_file:
        write_file = ToFile(out_file, pd_tocsv_options=pd_write_options)
        write_file(df)


@pdchemchain.command()
@click.argument("config_file", type=str)
@click.option(
    "--in_file",
    default=None,
    type=click.Path(exists=True, readable=True),
    help="Optional input file path",
)
@click.option(
    "--out_file",
    default=None,
    type=click.Path(writable=True),
    help="Optional output file path",
)
@click.option(
    "--error_file",
    default=None,
    type=click.Path(writable=True),
    help="Optional error file path",
)
@click.option(
    "--sep", type=str, default=",", help='Seperator for the input file, default=","'
)
@click.option("--debug_level", type=str, default=None, help="Optional debug level")
@click.option(
    "--custom_links",
    default=None,
    type=click.Path(exists=True, readable=True),
    help="Optional file with custom links configured as belonging to __main__. scope in config",
)
@click.option(
    "--pd_read_option",
    multiple=True,
    default=None,
    help="Extra options for Pandas read_csv() written as keyword=value. Multiple options can be repeated by using --pd_read_option multiple times.",
)
@click.option(
    "--pd_write_option",
    multiple=True,
    default=None,
    help="Extra options for Pandas to_csv() written as single keyword=value. Multiple options can be set by using --pd_write_option multiple times.",
)
def run(
    config_file,
    in_file,
    out_file,
    error_file,
    sep,
    debug_level,
    custom_links,
    pd_read_option,
    pd_write_option,
):
    """CONFIG_FILE: the json/yaml file with the specification for the pdchemchain chain or link."""
    process_data(
        in_file,
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
