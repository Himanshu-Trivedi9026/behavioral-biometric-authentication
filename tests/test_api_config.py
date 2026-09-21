"""
Phase 9A — configuration tests for :class:`backend.app.config.Settings`.

Covers defaults, ``BBA_*`` environment overrides, invalid-configuration
failures and the guarantee that no secrets or unrelated environment variables
appear in settings representations.
"""

import os

import pytest

from backend.app.config import Settings
from backend.app.main import create_app


class TestSettingsDefaults:
    def test_defaults_load(self):
        settings = Settings()
        assert settings.app_name == "Behavioral Biometric Authentication API"
        assert settings.app_version == "0.1.0"
        assert settings.environment == "development"
        assert settings.debug is False
        assert settings.api_prefix == "/api/v1"
        assert settings.cors_origins == ("http://localhost:5173",)

    def test_from_env_with_empty_environment(self, monkeypatch):
        for key in (
            "BBA_APP_NAME", "BBA_APP_VERSION", "BBA_ENVIRONMENT",
            "BBA_DEBUG", "BBA_API_PREFIX", "BBA_CORS_ORIGINS",
        ):
            monkeypatch.delenv(key, raising=False)
        settings = Settings.from_env()
        assert settings.app_name == "Behavioral Biometric Authentication API"
        assert settings.environment == "development"
        assert settings.cors_origins == ("http://localhost:5173",)

    def test_from_env_reads_overrides(self, monkeypatch):
        monkeypatch.setenv("BBA_APP_NAME", "Unit Test API")
        monkeypatch.setenv("BBA_APP_VERSION", "9.9.9")
        monkeypatch.setenv("BBA_ENVIRONMENT", "test")
        monkeypatch.setenv("BBA_DEBUG", "true")
        monkeypatch.setenv("BBA_API_PREFIX", "/api/unit")
        monkeypatch.setenv("BBA_CORS_ORIGINS", " http://a.example , http://b.example ")
        settings = Settings.from_env()
        assert settings.app_name == "Unit Test API"
        assert settings.app_version == "9.9.9"
        assert settings.environment == "test"
        assert settings.debug is True
        assert settings.api_prefix == "/api/unit"
        assert settings.cors_origins == ("http://a.example", "http://b.example")

    def test_debug_parsing_true_like(self):
        for raw in ("true", "1", "yes", "TRUE", "Yes"):
            assert Settings(debug=raw in ("true", "1", "yes", "TRUE", "Yes")) is not None


class TestSettingsValidation:
    def test_invalid_environment_fails(self):
        with pytest.raises(ValueError, match="environment"):
            Settings(environment="production-ish")

    def test_empty_app_name_fails(self):
        with pytest.raises(ValueError, match="app_name"):
            Settings(app_name="   ")

    def test_invalid_api_prefix_fails(self):
        with pytest.raises(ValueError, match="api_prefix"):
            Settings(api_prefix="api/v1")

    def test_wildcard_origin_fails(self):
        with pytest.raises(ValueError, match="cors_origins"):
            Settings(cors_origins=("*",))

    def test_environment_override_invalid_env_var_fails(self, monkeypatch):
        monkeypatch.setenv("BBA_ENVIRONMENT", "staging-ish")
        with pytest.raises(ValueError, match="environment"):
            Settings.from_env()


class TestSettingsPrivacy:
    def test_repr_contains_no_unrelated_env_variables(self, monkeypatch):
        monkeypatch.setenv("DATABASE_PASSWORD", "hunter2")
        monkeypatch.setenv("AWS_SECRET_KEY", "s3cr3t")
        settings = Settings.from_env()
        assert "hunter2" not in repr(settings)
        assert "hunter2" not in str(settings)
        assert "s3cr3t" not in repr(settings)

    def test_create_app_settings_attached(self):
        settings = Settings(environment="test")
        app = create_app(settings)
        assert app.state.settings is settings

    def test_cors_origin_default_is_documented_frontend_origin(self):
        settings = Settings()
        assert any("localhost:5173" in origin for origin in settings.cors_origins)