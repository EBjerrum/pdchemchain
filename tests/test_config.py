"""Tests for config module."""

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import yaml

from pdchemchain.config import (
    get_tool_path,
    get_schrodinger_path,
    ensure_schrodinger_job_server,
    get_num_physical_cores,
)


class TestGetToolPath:
    """Test tool path discovery precedence chain."""

    def test_explicit_path_wins(self):
        result = get_tool_path("schrodinger", explicit_path="/explicit/path")
        assert result == "/explicit/path"

    def test_env_var_second(self):
        with mock.patch.dict(os.environ, {"SCHRODINGER": "/env/schrodinger"}):
            result = get_tool_path("schrodinger")
            assert result == "/env/schrodinger"

    def test_explicit_overrides_env(self):
        with mock.patch.dict(os.environ, {"SCHRODINGER": "/env/schrodinger"}):
            result = get_tool_path("schrodinger", explicit_path="/explicit")
            assert result == "/explicit"

    def test_user_config_third(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"tools": {"schrodinger": "/user/config/path"}}))

        with mock.patch("pdchemchain.config._USER_CONFIG_PATH", config_file):
            with mock.patch.dict(os.environ, {}, clear=True):
                # Remove SCHRODINGER from env if present
                env = os.environ.copy()
                env.pop("SCHRODINGER", None)
                with mock.patch.dict(os.environ, env, clear=True):
                    result = get_tool_path("schrodinger")
                    assert result == "/user/config/path"

    def test_no_config_raises(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            env = os.environ.copy()
            env.pop("SCHRODINGER", None)
            with mock.patch.dict(os.environ, env, clear=True):
                with mock.patch("pdchemchain.config._USER_CONFIG_PATH", Path("/nonexistent")):
                    with pytest.raises(FileNotFoundError, match="Could not find path"):
                        get_tool_path("schrodinger")

    def test_unknown_tool_raises(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("pdchemchain.config._USER_CONFIG_PATH", Path("/nonexistent")):
                with pytest.raises(FileNotFoundError, match="Could not find path"):
                    get_tool_path("nonexistent_tool")


class TestGetSchrodingerPath:
    """Test convenience wrapper."""

    def test_returns_path(self):
        result = get_schrodinger_path(explicit_path="/test/path")
        assert result == "/test/path"


class TestEnsureSchrodingerJobServer:
    """Test job server auto-start logic."""

    def test_already_running(self):
        """No-op when server is already running."""
        with mock.patch("pdchemchain.config.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(
                returncode=0, stdout="Status of jobserverd: RUNNING\nJob server available at localhost:36231"
            )
            ensure_schrodinger_job_server(schrodinger_path="/test/schrodinger")
            mock_run.assert_called_once()
            assert "local-server-status" in mock_run.call_args[0][0]

    def test_starts_when_not_running(self):
        """Starts server when status check shows not running."""
        status_result = mock.Mock(returncode=1, stdout="", stderr="")
        start_result = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch("pdchemchain.config.subprocess.run", side_effect=[status_result, start_result]) as mock_run:
            ensure_schrodinger_job_server(schrodinger_path="/test/schrodinger")
            assert mock_run.call_count == 2
            assert "local-server-start" in mock_run.call_args_list[1][0][0]

    def test_raises_on_start_failure(self):
        """Raises RuntimeError when server fails to start."""
        status_result = mock.Mock(returncode=1, stdout="", stderr="")
        start_result = mock.Mock(returncode=1, stdout="", stderr="bind: address already in use")
        with mock.patch("pdchemchain.config.subprocess.run", side_effect=[status_result, start_result]):
            with pytest.raises(RuntimeError, match="Failed to start"):
                ensure_schrodinger_job_server(schrodinger_path="/test/schrodinger")


class TestGetNumPhysicalCores:
    """Slot count must reflect the ALLOCATION, not the machine.

    /proc/cpuinfo describes the host, so on a shared cluster node it over-reports badly.
    A 112-core node under --cpus-per-task=16 reported 112, Schrodinger was told to run
    112 subjobs inside a 16-cpu cgroup, and Glide exited rc=1 within seconds on every
    call -- with nothing but a hosts-file warning on stderr.
    """

    def test_schrodinger_host_wins(self):
        with mock.patch.dict(os.environ, {"SCHRODINGER_HOST": "localhost:16"}):
            assert get_num_physical_cores() == 16

    def test_malformed_schrodinger_host_falls_back(self):
        with mock.patch.dict(os.environ, {"SCHRODINGER_HOST": "localhost:notanumber"}):
            assert get_num_physical_cores() > 0

    def test_zero_slots_ignored(self):
        with mock.patch.dict(os.environ, {"SCHRODINGER_HOST": "localhost:0"}):
            assert get_num_physical_cores() > 0

    def test_non_localhost_host_ignored(self):
        """A remote host string says nothing about local slot count."""
        with mock.patch.dict(os.environ, {"SCHRODINGER_HOST": "somehost:8"}):
            with mock.patch("os.sched_getaffinity", return_value=set(range(4))):
                assert get_num_physical_cores() == 4

    def test_falls_back_to_cpu_affinity(self):
        """What SLURM actually restricts the process to."""
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("os.sched_getaffinity", return_value=set(range(16))):
                assert get_num_physical_cores() == 16

    def test_never_returns_zero(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            assert get_num_physical_cores() >= 1
