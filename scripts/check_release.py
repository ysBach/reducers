"""Validate reducers release metadata without importing the native extension."""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import tomllib

_PACKAGE_NAME = "reducers"
_VERSION = r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
_SECTION_RE = re.compile(r"^##[ \t]+(?P<title>.+?)[ \t]*$", re.MULTILINE)
_VERSION_SECTION_RE = re.compile(
    rf"^(?P<version>{_VERSION})[ \t]+-[ \t]+(?P<status>.+)$"
)
_CARGO_DEPENDENCY_KEYS = ("dependencies", "dev-dependencies", "build-dependencies")


class ReleaseError(ValueError):
    """Raised when release metadata does not satisfy the publishing contract."""


def _load_toml(root: Path, filename: str) -> dict[str, Any]:
    try:
        with (root / filename).open("rb") as source:
            value = tomllib.load(source)
    except tomllib.TOMLDecodeError as error:
        raise ReleaseError(f"{filename}: invalid TOML: {error}") from error
    if not isinstance(value, dict):
        raise ReleaseError(f"{filename}: expected a TOML table")
    return value


def _check_cargo_spec(name: str, specification: object, label: str) -> None:
    if isinstance(specification, str):
        if specification.strip():
            return
        raise ReleaseError(f"{label}: {name} requires a registry version")
    if not isinstance(specification, Mapping):
        raise ReleaseError(f"{label}: {name} requires a registry version")
    if "path" in specification:
        raise ReleaseError(f"{label}: {name} uses a path dependency override")
    if "git" in specification:
        raise ReleaseError(f"{label}: {name} uses a git dependency override")
    registry = specification.get("registry")
    if registry is not None and registry != "crates-io":
        raise ReleaseError(f"{label}: {name} uses a non-crates.io registry")
    version = specification.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ReleaseError(f"{label}: {name} requires a registry version")


def _check_cargo_scope(
    scope: Mapping[str, object],
    *,
    label: str,
    workspace_dependencies: Mapping[str, object] | None,
) -> None:
    if "replace" in scope or scope.get("patch"):
        raise ReleaseError(f"{label}: source overrides are not release sources")
    for key in _CARGO_DEPENDENCY_KEYS:
        dependencies = scope.get(key, {})
        if not isinstance(dependencies, Mapping):
            raise ReleaseError(f"{label} {key}: expected a table")
        for name, specification in dependencies.items():
            if not isinstance(name, str):
                raise ReleaseError(f"{label} {key}: dependency names must be strings")
            if (
                isinstance(specification, Mapping)
                and specification.get("workspace") is True
            ):
                if workspace_dependencies is None or name not in workspace_dependencies:
                    raise ReleaseError(
                        f"{label} {key}: {name} has no workspace registry version"
                    )
                specification = workspace_dependencies[name]
            _check_cargo_spec(name, specification, f"{label} {key}")
    targets = scope.get("target", {})
    if not isinstance(targets, Mapping):
        raise ReleaseError(f"{label} target: expected a table")
    for target_name, target_scope in targets.items():
        if not isinstance(target_scope, Mapping):
            raise ReleaseError(f"{label} target {target_name}: expected a table")
        _check_cargo_scope(
            target_scope,
            label=f"{label} target {target_name}",
            workspace_dependencies=workspace_dependencies,
        )


def _check_cargo_dependencies(cargo: Mapping[str, object]) -> None:
    workspace = cargo.get("workspace", {})
    if workspace is not None and not isinstance(workspace, Mapping):
        raise ReleaseError("Cargo.toml workspace: expected a table")
    workspace_dependencies = (
        workspace.get("dependencies") if isinstance(workspace, Mapping) else None
    )
    if workspace_dependencies is not None and not isinstance(
        workspace_dependencies, Mapping
    ):
        raise ReleaseError("Cargo.toml workspace.dependencies: expected a table")
    _check_cargo_scope(
        cargo,
        label="Cargo.toml",
        workspace_dependencies=workspace_dependencies,
    )
    if isinstance(workspace, Mapping):
        _check_cargo_scope(
            workspace,
            label="Cargo.toml workspace",
            workspace_dependencies=workspace_dependencies,
        )


def _check_python_requirement(requirement: object, label: str) -> None:
    if not isinstance(requirement, str) or not requirement.strip():
        raise ReleaseError(f"pyproject.toml {label}: expected a registry requirement")
    reference = requirement.split(";", 1)[0].strip()
    if (
        "@" in reference
        or re.match(
            r"(?i)^(?:[A-Za-z][A-Za-z0-9+.-]*://|(?:git|hg|svn|bzr)\+)",
            reference,
        )
        or re.match(r"^(?:[./\\]|[A-Za-z]:[\\/])", reference)
    ):
        raise ReleaseError(
            f"pyproject.toml {label}: direct local/URL dependency {requirement!r}"
        )


def _check_python_dependencies(python: Mapping[str, object]) -> None:
    tool = python.get("tool", {})
    if not isinstance(tool, Mapping):
        raise ReleaseError("pyproject.toml tool: expected a table")
    uv = tool.get("uv", {})
    if uv is not None and not isinstance(uv, Mapping):
        raise ReleaseError("pyproject.toml tool.uv: expected a table")
    if isinstance(uv, Mapping) and uv.get("sources"):
        raise ReleaseError(
            "pyproject.toml tool.uv.sources overrides are not release sources"
        )
    project = python.get("project", {})
    if not isinstance(project, Mapping):
        raise ReleaseError("pyproject.toml project: expected a table")
    build_system = python.get("build-system", {})
    if not isinstance(build_system, Mapping):
        raise ReleaseError("pyproject.toml build-system: expected a table")
    groups: list[tuple[str, object]] = [
        ("project.dependencies", project.get("dependencies", [])),
        ("build-system.requires", build_system.get("requires", [])),
    ]
    optional = project.get("optional-dependencies", {})
    if not isinstance(optional, Mapping):
        raise ReleaseError(
            "pyproject.toml project.optional-dependencies: expected a table"
        )
    groups.extend(
        (f"project.optional-dependencies.{name}", requirements)
        for name, requirements in optional.items()
    )
    if isinstance(uv, Mapping) and uv.get("dev-dependencies"):
        groups.append(("tool.uv.dev-dependencies", uv["dev-dependencies"]))
    dependency_groups = python.get("dependency-groups", {})
    if dependency_groups is not None and not isinstance(dependency_groups, Mapping):
        raise ReleaseError("pyproject.toml dependency-groups: expected a table")
    if isinstance(dependency_groups, Mapping):
        groups.extend(
            (f"dependency-groups.{name}", requirements)
            for name, requirements in dependency_groups.items()
        )
    for label, requirements in groups:
        if not isinstance(requirements, Sequence) or isinstance(
            requirements, (str, bytes)
        ):
            raise ReleaseError(f"pyproject.toml {label}: expected a list")
        for requirement in requirements:
            if isinstance(requirement, Mapping) and set(requirement) == {
                "include-group"
            }:
                continue
            _check_python_requirement(requirement, label)


def _local_lock_entry(
    data: Mapping[str, object], *, filename: str
) -> Mapping[str, object]:
    packages = data.get("package")
    if not isinstance(packages, list):
        raise ReleaseError(f"{filename}: expected a package list")
    local: list[Mapping[str, object]] = []
    for package in packages:
        if not isinstance(package, Mapping):
            raise ReleaseError(f"{filename}: package entries must be tables")
        if package.get("name") != _PACKAGE_NAME:
            continue
        if filename == "Cargo.lock" and "source" not in package:
            local.append(package)
        elif filename == "uv.lock" and package.get("source") in (
            {"editable": "."},
            {"virtual": "."},
        ):
            local.append(package)
    if len(local) != 1:
        raise ReleaseError(
            f"{filename}: expected exactly one local {_PACKAGE_NAME} entry"
        )
    return local[0]


def _changelog_sections(text: str) -> list[tuple[str, str]]:
    headings = list(_SECTION_RE.finditer(text))
    sections: list[tuple[str, str]] = []
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        sections.append(
            (heading.group("title").strip(), text[heading.end() : end].strip())
        )
    return sections


def _changelog_notes(root: Path, version: str, tag: str | None) -> str:
    sections = _changelog_sections((root / "CHANGELOG.md").read_text(encoding="utf-8"))
    unreleased = [body for title, body in sections if title == "Unreleased"]
    if len(unreleased) > 1:
        raise ReleaseError("CHANGELOG.md: expected at most one ## Unreleased section")
    version_sections = [
        (match.group("status").strip(), body)
        for title, body in sections
        if (match := _VERSION_SECTION_RE.fullmatch(title)) is not None
        and match.group("version") == version
    ]
    if len(version_sections) > 1:
        raise ReleaseError(
            f"CHANGELOG.md: expected exactly one ## {version} - ... section"
        )
    if tag:
        if unreleased and unreleased[0]:
            raise ReleaseError(
                "CHANGELOG.md: tagged release has nonempty ## Unreleased notes"
            )
        if not version_sections:
            raise ReleaseError(
                f"CHANGELOG.md: expected ## {version} - YYYY-MM-DD section"
            )
        status, notes = version_sections[0]
        if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", status) is None:
            raise ReleaseError(
                "CHANGELOG.md: tagged release requires a YYYY-MM-DD section"
            )
        try:
            date.fromisoformat(status)
        except ValueError as error:
            raise ReleaseError(
                f"CHANGELOG.md: invalid release date {status!r}"
            ) from error
        if not notes:
            raise ReleaseError("CHANGELOG.md: tagged release requires nonempty notes")
        return f"{notes}\n"
    if unreleased:
        return f"{unreleased[0]}\n" if unreleased[0] else ""
    if version_sections:
        return f"{version_sections[0][1]}\n" if version_sections[0][1] else ""
    raise ReleaseError(
        "CHANGELOG.md: expected ## Unreleased or a matching version section"
    )


def _check_clean(root: Path) -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = result.stderr.strip()
        suffix = f": {detail}" if detail else ""
        raise ReleaseError(f"git status failed{suffix}")
    if result.stdout:
        raise ReleaseError(
            "release requires a clean tree (tracked and untracked files)"
        )


def check_release(
    root: Path, *, tag: str | None = None, require_clean: bool = False
) -> tuple[str, str]:
    """Validate reducers release metadata and return version and release notes.

    Parameters
    ----------
    root : pathlib.Path
        Repository containing the manifests, lockfiles, and changelog.
    tag : str or None, optional
        Exact ``vX.Y.Z`` tag for publication. `None` or an empty string performs
        a branch rehearsal and permits a plain `## Unreleased` section.
    require_clean : bool, optional
        Reject tracked, staged, and untracked files by querying Git.

    Returns
    -------
    version : str
        Stable version shared by the manifests and local lock entries.
    notes : str
        Notes from the pending rehearsal or dated release section, normalized
        to one final newline.

    Raises
    ------
    ReleaseError
        Release metadata, dependency sources, changelog, tag, or Git checks
        fail.
    OSError
        A required file or Git executable cannot be accessed.
    """
    root = root.resolve()
    cargo_toml = _load_toml(root, "Cargo.toml")
    pyproject = _load_toml(root, "pyproject.toml")
    cargo_lock = _load_toml(root, "Cargo.lock")
    uv_lock = _load_toml(root, "uv.lock")
    cargo = cargo_toml.get("package")
    project = pyproject.get("project")
    if not isinstance(cargo, Mapping) or cargo.get("name") != _PACKAGE_NAME:
        raise ReleaseError(f"Cargo.toml: package name must be {_PACKAGE_NAME!r}")
    if not isinstance(project, Mapping) or project.get("name") != _PACKAGE_NAME:
        raise ReleaseError(f"pyproject.toml: package name must be {_PACKAGE_NAME!r}")
    version = cargo.get("version")
    if not isinstance(version, str) or re.fullmatch(_VERSION, version) is None:
        raise ReleaseError("Cargo.toml: version must be a stable X.Y.Z release")
    if project.get("version") != version:
        raise ReleaseError(f"pyproject.toml: version must match {version}")
    _check_cargo_dependencies(cargo_toml)
    _check_python_dependencies(pyproject)
    for filename, data in (("Cargo.lock", cargo_lock), ("uv.lock", uv_lock)):
        entry = _local_lock_entry(data, filename=filename)
        if entry.get("version") != version:
            raise ReleaseError(f"{filename}: local version must match {version}")
    normalized_tag = tag or None
    if normalized_tag and (
        re.fullmatch(r"v\d+\.\d+\.\d+", normalized_tag) is None
        or normalized_tag != f"v{version}"
    ):
        raise ReleaseError(f"tag {normalized_tag!r} must match v{version}")
    notes = _changelog_notes(root, version, normalized_tag)
    if require_clean:
        _check_clean(root)
    return version, notes


def main(argv: Sequence[str] | None = None) -> int:
    """Validate release metadata and write optional machine-readable outputs.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command-line arguments; `None` reads process arguments.

    Returns
    -------
    int
        Zero on success and one after a validation, file, or Git error.
        ``--github-output`` appends ``version=X.Y.Z`` and ``--notes-output``
        overwrites the extracted changelog notes.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--tag", help="exact vX.Y.Z tag; omit or empty for rehearsal")
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--notes-output", type=Path)
    args = parser.parse_args(argv)
    try:
        version, notes = check_release(
            args.root, tag=args.tag, require_clean=args.require_clean
        )
        if args.notes_output is not None:
            with args.notes_output.open("w", encoding="utf-8", newline="\n") as output:
                output.write(notes)
        if args.github_output is not None:
            with args.github_output.open("a", encoding="utf-8", newline="\n") as output:
                output.write(f"version={version}\n")
    except (ReleaseError, OSError) as error:
        logging.error("Release check failed: %s", error)
        return 1
    logging.info("Release metadata verified: %s", version)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
