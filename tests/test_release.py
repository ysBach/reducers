"""Release metadata checks without importing the native extension."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

if sys.version_info < (3, 11):
    pytest.skip(
        "release checker requires Python 3.11+ tomllib", allow_module_level=True
    )

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_release.py"
_SPEC = importlib.util.spec_from_file_location("reducers_check_release", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
release = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(release)

_VERSION = "0.3.2"


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """Create a minimal, extension-free reducers release tree."""
    files = {
        "Cargo.toml": f"""\
[package]
name = "reducers"
version = "{_VERSION}"
edition = "2021"

[dependencies]
rayon = "1.10"
""",
        "pyproject.toml": f"""\
[build-system]
requires = ["maturin>=1.5,<2"]
build-backend = "maturin"

[project]
name = "reducers"
version = "{_VERSION}"
dependencies = ["numpy>=1.24"]

[project.optional-dependencies]
test = ["pytest>=7"]
""",
        "Cargo.lock": f"""\
version = 3

[[package]]
name = "rayon"
version = "1.10.0"
source = "registry+https://github.com/rust-lang/crates.io-index"

[[package]]
name = "reducers"
version = "{_VERSION}"
dependencies = ["rayon"]
""",
        "uv.lock": f"""\
version = 1
requires-python = ">=3.10"

[[package]]
name = "numpy"
version = "2.4.6"
source = {{ registry = "https://pypi.org/simple" }}

[[package]]
name = "reducers"
version = "{_VERSION}"
source = {{ editable = "." }}
""",
        "CHANGELOG.md": f"""\
# Changelog

## Unreleased

- Pending release work.

## {_VERSION} - 2026-09-05

### Added

- Reducer improvements.

## 0.3.1 - 2026-07-14

- Older release.
""",
    }
    for name, contents in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    return tmp_path


def _make_tag_ready(repository: Path) -> None:
    path = repository / "CHANGELOG.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "## Unreleased\n\n- Pending release work.\n",
            "## Unreleased\n\n",
        ),
        encoding="utf-8",
    )


def test_rehearsal_reads_pending_notes_without_importing_extension(
    repository: Path,
) -> None:
    assert release.check_release(repository) == (
        _VERSION,
        "- Pending release work.\n",
    )
    assert release.check_release(repository, tag="") == release.check_release(
        repository
    )


def test_tag_rejects_nonempty_unreleased_section(repository: Path) -> None:
    with pytest.raises(release.ReleaseError, match="nonempty.*Unreleased"):
        release.check_release(repository, tag=f"v{_VERSION}")


def test_tag_accepts_dated_nonempty_version_section(repository: Path) -> None:
    _make_tag_ready(repository)
    assert release.check_release(repository, tag=f"v{_VERSION}") == (
        _VERSION,
        "### Added\n\n- Reducer improvements.\n",
    )


@pytest.mark.parametrize("tag", ["v0.3.1", "0.3.2", "v0.3.2rc1", " "])
def test_tag_must_match_stable_version(repository: Path, tag: str) -> None:
    with pytest.raises(release.ReleaseError, match="tag .* must match v0.3.2"):
        release.check_release(repository, tag=tag)


@pytest.mark.parametrize("version", ["1.2", "1.2.3rc1", "1.2.3-dev", "01.2.3"])
def test_only_stable_manifest_versions_are_accepted(
    repository: Path, version: str
) -> None:
    path = repository / "Cargo.toml"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            f'version = "{_VERSION}"', f'version = "{version}"'
        ),
        encoding="utf-8",
    )
    with pytest.raises(release.ReleaseError, match="stable X.Y.Z"):
        release.check_release(repository)


@pytest.mark.parametrize("filename", ["pyproject.toml", "Cargo.lock", "uv.lock"])
def test_manifest_and_lock_versions_must_agree(repository: Path, filename: str) -> None:
    path = repository / filename
    path.write_text(
        path.read_text(encoding="utf-8").replace(_VERSION, "0.3.3"),
        encoding="utf-8",
    )
    with pytest.raises(release.ReleaseError, match="version"):
        release.check_release(repository)


@pytest.mark.parametrize("filename", ["Cargo.lock", "uv.lock"])
def test_lock_must_contain_one_local_root_entry(
    repository: Path, filename: str
) -> None:
    path = repository / filename
    contents = path.read_text(encoding="utf-8")
    if filename == "Cargo.lock":
        contents = contents.replace(
            f'name = "reducers"\nversion = "{_VERSION}"\n',
            f'name = "reducers"\nversion = "{_VERSION}"\n'
            'source = "registry+https://github.com/rust-lang/crates.io-index"\n',
        )
    else:
        contents = contents.replace(
            'source = { editable = "." }',
            'source = { registry = "https://pypi.org/simple" }',
        )
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(release.ReleaseError, match="exactly one local"):
        release.check_release(repository)


@pytest.mark.parametrize(
    "status",
    ["Unreleased", "2026-02-30", "2026-9-5", "later"],
)
def test_tag_requires_valid_dated_changelog_section(
    repository: Path, status: str
) -> None:
    _make_tag_ready(repository)
    path = repository / "CHANGELOG.md"
    contents = path.read_text(encoding="utf-8").replace(
        f"## {_VERSION} - 2026-09-05", f"## {_VERSION} - {status}"
    )
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(release.ReleaseError, match="(date|YYYY-MM-DD)"):
        release.check_release(repository, tag=f"v{_VERSION}")


def test_tag_requires_nonempty_matching_section(repository: Path) -> None:
    _make_tag_ready(repository)
    path = repository / "CHANGELOG.md"
    contents = path.read_text(encoding="utf-8").replace(
        "### Added\n\n- Reducer improvements.\n", ""
    )
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(release.ReleaseError, match="nonempty"):
        release.check_release(repository, tag=f"v{_VERSION}")

    path.write_text(
        contents.replace(f"## {_VERSION} - 2026-09-05", "## 0.3.3 - 2026-09-05"),
        encoding="utf-8",
    )
    with pytest.raises(release.ReleaseError, match=r"expected ## 0\.3\.2"):
        release.check_release(repository, tag=f"v{_VERSION}")


@pytest.mark.parametrize(
    "source",
    ['path = "../external"', 'git = "https://example.invalid/external.git"'],
)
def test_cargo_path_and_git_overrides_are_rejected(
    repository: Path, source: str
) -> None:
    path = repository / "Cargo.toml"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            'rayon = "1.10"',
            f'rayon = "1.10"\nexternal = {{ version = "1", {source} }}',
        ),
        encoding="utf-8",
    )
    with pytest.raises(release.ReleaseError, match="external uses a (path|git)"):
        release.check_release(repository)


def test_cargo_dependency_requires_registry_version(repository: Path) -> None:
    path = repository / "Cargo.toml"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            'rayon = "1.10"', 'rayon = "1.10"\nexternal = { default-features = false }'
        ),
        encoding="utf-8",
    )
    with pytest.raises(release.ReleaseError, match="requires a registry version"):
        release.check_release(repository)


def test_python_source_overrides_are_rejected(repository: Path) -> None:
    path = repository / "pyproject.toml"
    path.write_text(
        path.read_text(encoding="utf-8")
        + '\n[tool.uv.sources]\nreducers = { path = ".", editable = true }\n',
        encoding="utf-8",
    )
    with pytest.raises(release.ReleaseError, match="tool.uv.sources"):
        release.check_release(repository)


@pytest.mark.parametrize(
    "requirement", ["../external", "numpy @ https://example.invalid/numpy.whl"]
)
def test_python_direct_local_or_url_dependencies_are_rejected(
    repository: Path, requirement: str
) -> None:
    path = repository / "pyproject.toml"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            'dependencies = ["numpy>=1.24"]', f'dependencies = ["{requirement}"]'
        ),
        encoding="utf-8",
    )
    with pytest.raises(release.ReleaseError, match="direct local/URL"):
        release.check_release(repository)


def test_cli_writes_version_and_notes_to_requested_files(
    repository: Path, tmp_path: Path
) -> None:
    notes = tmp_path / "release-notes.md"
    github_output = tmp_path / "github-output"
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--root",
            str(repository),
            "--tag",
            "",
            "--notes-output",
            str(notes),
            "--github-output",
            str(github_output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert notes.read_text(encoding="utf-8") == "- Pending release work.\n"
    assert github_output.read_text(encoding="utf-8") == "version=0.3.2\n"


@pytest.fixture
def clean_repository(repository: Path) -> Path:
    """Commit the synthetic tree so the Git cleanliness check is meaningful."""
    subprocess.run(["git", "init", "--quiet"], cwd=repository, check=True)
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Release Test",
            "-c",
            "user.email=release@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "-m",
            "Fixture",
        ],
        cwd=repository,
        check=True,
    )
    return repository


def test_require_clean_includes_untracked_files(clean_repository: Path) -> None:
    assert release.check_release(clean_repository, require_clean=True)[0] == _VERSION
    (clean_repository / "untracked.txt").write_text("pending\n", encoding="utf-8")
    with pytest.raises(release.ReleaseError, match="clean tree"):
        release.check_release(clean_repository, require_clean=True)
