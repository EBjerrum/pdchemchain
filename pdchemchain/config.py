"""Tool path discovery for external dependencies.

Precedence chain:
    explicit parameter > environment variable > user config file > system default

User config file: ~/.pdchemchain/config.yaml
"""

import logging
import os
import subprocess
import time
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# System defaults — can be overridden at any level
_DEFAULTS = {
}

# Environment variable mapping
_ENV_VARS = {
    "schrodinger": "SCHRODINGER",
}

_USER_CONFIG_PATH = Path.home() / ".pdchemchain" / "config.yaml"


def _load_user_config() -> dict:
    """Load user config from ~/.pdchemchain/config.yaml if it exists."""
    if _USER_CONFIG_PATH.is_file():
        with open(_USER_CONFIG_PATH) as f:
            config = yaml.safe_load(f) or {}
        return config.get("tools", {})
    return {}


def get_tool_path(tool_name: str, explicit_path: str = None) -> str:
    """Discover the path for an external tool.

    Parameters
    ----------
    tool_name : str
        Tool identifier, e.g. "schrodinger".
    explicit_path : str, optional
        If provided, used directly (highest priority).

    Returns
    -------
    str
        Resolved tool path.

    Raises
    ------
    FileNotFoundError
        If no path could be resolved through any source.
    """
    # 1. Explicit parameter
    if explicit_path is not None:
        logger.debug(f"{tool_name}: using explicit path '{explicit_path}'")
        return explicit_path

    # 2. Environment variable
    env_var = _ENV_VARS.get(tool_name)
    if env_var:
        env_value = os.environ.get(env_var)
        if env_value:
            logger.debug(f"{tool_name}: using ${env_var}='{env_value}'")
            return env_value

    # 3. User config file
    user_config = _load_user_config()
    if tool_name in user_config:
        path = user_config[tool_name]
        logger.debug(f"{tool_name}: using user config '{path}'")
        return path

    # 4. System default
    if tool_name in _DEFAULTS:
        path = _DEFAULTS[tool_name]
        logger.debug(f"{tool_name}: using system default '{path}'")
        return path

    raise FileNotFoundError(
        f"Could not find path for tool '{tool_name}'. "
        f"Set the ${_ENV_VARS.get(tool_name, tool_name.upper())} environment variable, "
        f"add it to {_USER_CONFIG_PATH}, or pass it explicitly."
    )


def get_num_physical_cores() -> int:
    """Get the number of cores this process may actually use.

    Respects, in order of precedence:

    1. ``SCHRODINGER_HOST`` of the form ``localhost:N`` -- an explicit override.
    2. The CPU affinity mask (``sched_getaffinity``), which is what a batch scheduler
       such as SLURM restricts the process to.
    3. Physical cores from ``/proc/cpuinfo``, excluding hyperthreads.
    4. ``os.cpu_count()``.

    Why this is not just a core count. ``/proc/cpuinfo`` describes the MACHINE, not the
    allocation, so on a shared cluster node it over-reports badly: a 112-core node with
    ``--cpus-per-task=16`` returned 112, Schrodinger's job server was told to run 112
    subjobs inside a 16-cpu cgroup, and Glide exited rc=1 after ~3 seconds with nothing
    but a hosts-file warning on stderr. LigPrep survived it (it degrades gracefully),
    which made the failure look Glide-specific rather than a slot-count problem.
    """
    env_host = os.environ.get("SCHRODINGER_HOST", "")
    if env_host.startswith("localhost:"):
        try:
            n = int(env_host.split(":", 1)[1])
            if n > 0:
                return n
        except ValueError:
            pass

    # The scheduler's allocation, when there is one.
    if hasattr(os, "sched_getaffinity"):
        try:
            allowed = len(os.sched_getaffinity(0))
            if allowed > 0:
                return allowed
        except OSError:
            pass

    try:
        with open("/proc/cpuinfo") as f:
            cores = set()
            phys_id = core_id = None
            for line in f:
                if line.startswith("physical id"):
                    phys_id = line.split(":")[1].strip()
                elif line.startswith("core id"):
                    core_id = line.split(":")[1].strip()
                    if phys_id is not None:
                        cores.add((phys_id, core_id))
                        phys_id = core_id = None
            if cores:
                return len(cores)
    except OSError:
        pass
    return os.cpu_count() or 1


def get_schrodinger_path(explicit_path: str = None) -> str:
    """Convenience wrapper for Schrodinger path discovery."""
    return get_tool_path("schrodinger", explicit_path)


def ensure_schrodinger_job_server(schrodinger_path: str = None) -> None:
    """Ensure a local Schrodinger job server is running, starting one if needed.

    Parameters
    ----------
    schrodinger_path : str, optional
        Explicit Schrodinger installation path. If None, discovered via config.
    """
    schrodinger = get_schrodinger_path(schrodinger_path)
    jsc = str(Path(schrodinger) / "jsc")

    if not Path(jsc).exists():
        raise RuntimeError(
            f"Schrodinger jsc not found at {jsc}. "
            f"Check your SCHRODINGER path ({schrodinger})."
        )

    result = subprocess.run(
        [jsc, "local-server-status"], capture_output=True, text=True
    )
    if result.returncode == 0 and "RUNNING" in result.stdout:
        logger.debug("Schrodinger job server already running")
        return

    logger.info("Starting local Schrodinger job server...")
    start_result = subprocess.run(
        [jsc, "local-server-start"], capture_output=True, text=True
    )
    if start_result.returncode != 0:
        raise RuntimeError(
            f"Failed to start Schrodinger job server: {start_result.stderr}"
        )

    # Poll until server is actually ready (start returns before it accepts jobs)
    for _ in range(10):
        time.sleep(1)
        check = subprocess.run(
            [jsc, "local-server-status"], capture_output=True, text=True
        )
        if check.returncode == 0 and "RUNNING" in check.stdout:
            logger.info("Schrodinger job server started successfully")
            return

    raise RuntimeError(
        "Schrodinger job server did not become ready within 10 seconds. "
        "Try running '$SCHRODINGER/jsc local-server-start' manually."
    )
