DOCS_DIR := docs/quarto
DOCS_PORT ?= 8000
DOCS_PYTHON ?= $(CURDIR)/.venv/bin/python
DOCS_CACHE := $(CURDIR)/$(DOCS_DIR)/.cache
DOCS_HOME := $(CURDIR)/$(DOCS_DIR)/.home
DOCS_DENO := $(CURDIR)/$(DOCS_DIR)/.deno-cache
DOCS_GRAINS := $(CURDIR)/benchmarks/recorded_parallel_grains.json
DOCS_RUSTDOC_SITE := $(CURDIR)/$(DOCS_DIR)/_site/rustdoc
DOCS_ENV = HOME="$(DOCS_HOME)" XDG_CACHE_HOME="$(DOCS_CACHE)" DENO_DIR="$(DOCS_DENO)" MPLBACKEND=module://matplotlib_inline.backend_inline QUARTO_PYTHON="$(DOCS_PYTHON)" REDUCERS_PARALLEL_GRAINS_FILE="$(DOCS_GRAINS)"

.PHONY: docs-api docs-rust docs-rust-copy docs-render docs-build docs-preview docs-serve docs-clean

docs-api:  ## Regenerate docstring/API reference qmd files only.
	cd $(DOCS_DIR) && "$(DOCS_PYTHON)" -m quartodoc build

docs-rust:  ## Build the full Rust API reference with rustdoc.
	cargo doc --no-default-features --no-deps --document-private-items

docs-rust-copy:  ## Copy rustdoc output into the rendered Quarto site.
	test -d target/doc
	rm -rf "$(DOCS_RUSTDOC_SITE)"
	mkdir -p "$(DOCS_RUSTDOC_SITE)"
	cp -R target/doc/. "$(DOCS_RUSTDOC_SITE)"

docs-render:  ## Render the whole Quarto website from existing qmd files.
	cd $(DOCS_DIR) && $(DOCS_ENV) quarto render

docs-build: docs-api docs-rust docs-render docs-rust-copy  ## Build complete Python, Rust, and Quarto docs in one command.

docs-preview: docs-build docs-serve  ## Build the site, then serve docs/quarto/_site locally.

docs-serve:  ## Serve the existing rendered site without rebuilding.
	cd $(DOCS_DIR) && python -m http.server $(DOCS_PORT) -d _site

docs-clean:
	rm -rf docs/quarto/_site docs/quarto/.quarto docs/quarto/api docs/quarto/objects.json docs/quarto/.cache docs/quarto/.home docs/quarto/.deno-cache
	find docs -name .DS_Store -delete
