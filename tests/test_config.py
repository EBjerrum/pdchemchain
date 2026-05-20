"""Tests for config module."""

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import yaml

from pdchemchain.config import get_tool_path, get_schrodinger_path


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
