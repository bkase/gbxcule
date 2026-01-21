"""Tests for package importability."""


def test_import_gbxcule() -> None:
    """Test that gbxcule can be imported."""
    import gbxcule

    assert gbxcule is not None


def test_version_exists() -> None:
    """Test that __version__ exists and is a string."""
    import gbxcule

    assert hasattr(gbxcule, "__version__")
    assert isinstance(gbxcule.__version__, str)
    assert gbxcule.__version__ == "0.0.0"


def test_import_backends() -> None:
    """Test that backends subpackage can be imported."""
    import gbxcule.backends

    assert gbxcule.backends is not None


def test_import_backends_common() -> None:
    """Test that common types can be imported."""
    from gbxcule.backends.common import (
        ActionSpec,
        ObsSpec,
        StepOutput,
        VecBackend,
    )

    assert ActionSpec is not None
    assert ObsSpec is not None
    assert StepOutput is not None
    assert VecBackend is not None


def test_import_core() -> None:
    """Test that core subpackage can be imported."""
    import gbxcule.core

    assert gbxcule.core is not None


def test_import_core_abi() -> None:
    """Test that abi module can be imported."""
    from gbxcule.core.abi import ABI_VERSION

    assert ABI_VERSION == 0


def test_import_kernels() -> None:
    """Test that kernels subpackage can be imported."""
    import gbxcule.kernels

    assert gbxcule.kernels is not None
