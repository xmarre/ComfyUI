from types import SimpleNamespace

import comfy.model_management as model_management


class TestModelManagement:
    def test_is_wsl_detects_microsoft_suffix(self, monkeypatch):
        monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
        monkeypatch.setattr(
            model_management.platform,
            "uname",
            lambda: SimpleNamespace(release="6.6.87.2-Microsoft"),
        )

        assert model_management.is_wsl() is True

    def test_is_wsl_detects_wsl2_suffix(self, monkeypatch):
        monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
        monkeypatch.setattr(
            model_management.platform,
            "uname",
            lambda: SimpleNamespace(release="5.15.167.4-microsoft-standard-WSL2"),
        )

        assert model_management.is_wsl() is True

    def test_is_wsl_detects_distro_env_for_custom_kernel(self, monkeypatch):
        monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
        monkeypatch.setattr(
            model_management.platform,
            "uname",
            lambda: SimpleNamespace(release="6.6.87.2-custom"),
        )

        assert model_management.is_wsl() is True

    def test_is_wsl_returns_false_without_env_or_known_suffix(self, monkeypatch):
        monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
        monkeypatch.setattr(
            model_management.platform,
            "uname",
            lambda: SimpleNamespace(release="6.6.87.2-generic"),
        )

        assert model_management.is_wsl() is False
