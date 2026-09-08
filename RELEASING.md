# Releasing reducers

Push a matching `vX.Y.Z` tag only after a successful manual rehearsal. CI,
five platform wheels, and an installed sdist must pass before publishing to
PyPI and crates.io. The GitHub release follows both registries and includes
the Python distributions and validated changelog entry.

## One-time setup

Create GitHub environments `pypi` and `crates-io`, restricted to **tags** matching
`v*`. Configure these trusted publishers in the registry projects:

| Setting | PyPI | crates.io |
| --- | --- | --- |
| Project/crate | `reducers` | `reducers` |
| Repository owner/name | `ysBach/reducers` | `ysBach/reducers` |
| Workflow filename | `release.yml` | `release.yml` |
| Environment | `pypi` | `crates-io` |

Follow the [PyPI setup](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
and [crates.io setup](https://crates.io/docs/trusted-publishing). Confirm any
first-publication requirements if the project does not exist. Authentication
uses temporary OIDC credentials; no permanent publishing secret is needed.
Require CI on `main`. Dependabot opens Cargo, uv, and Actions update PRs weekly;
updates are reviewed manually and never automatically merged or released.

## Each release

1. Set the same version in `Cargo.toml` and `pyproject.toml`; refresh `Cargo.lock`
   with `cargo check --no-default-features` and `uv.lock` with `uv lock --no-sources`.
   Move pending changes into `## X.Y.Z - YYYY-MM-DD` in `CHANGELOG.md` and remove
   the `Unreleased` section. Use the project `.venv` with Python 3.11+ for release
   tooling; never synchronize `~/.venvs/ysbpy`. Commit the intended files.
2. Push the commit and rehearse:

   ```sh
   uv run --no-sync python scripts/check_release.py --require-clean
   git push origin main
   gh workflow run release.yml --ref main
   gh run list --workflow release.yml --event workflow_dispatch
   ```

   Inspect the run: all checks must pass and publishing must be skipped. Manual
   runs using the current workflow never publish, including on tags. Older tags
   may contain the previous workflow; use `main` for rehearsals.
3. Confirm local `HEAD` matches the successful run's `headSha`
   (`gh run view RUN_ID --json conclusion,headSha`). For the planned `0.4.0`:

   ```sh
   uv run --no-sync python scripts/check_release.py --tag v0.4.0 --require-clean
   git tag -a v0.4.0 -m "Release 0.4.0"
   git push origin v0.4.0
   ```

   **Pushing the tag publishes.** Change the example version for later releases.
4. Check the Actions run, both registry versions, and the GitHub release.
   On an external failure, rerun **failed jobs** in the original tag-push run
   (`gh run rerun RUN_ID --failed`), preserving successful jobs and artifacts.
   PyPI retries skip identical files. If crates.io or GitHub already accepted
   the release before a job failed, verify the existing result before retrying.
   Source changes require a new version; never move a published tag.

CI checks Rust 1.84/stable, formatting, linting, Rust-only dependencies, crate
packaging, and installed Python artifacts using published dependencies. Release
wheels target Linux/macOS x86-64 and ARM64, plus Windows x86-64, with CPython
3.10+ ABI3. Wheel tests use Python 3.10; sdist tests use 3.13. Documentation
publishing remains in `pages.yml`; multi-repository benchmarks stay opt-in.
