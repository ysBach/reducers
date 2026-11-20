"""Link documented API names in rendered inline code, types, and Python examples."""

from __future__ import annotations

import ast
import json
import os
import re
from html import escape, unescape
from html.parser import HTMLParser
from pathlib import Path

# A complete dotted name must resolve as a unit. Pandoc sometimes highlights
# one component separately, for example np.<span class="bu">sum</span>.
NAME = re.compile(r"""'[^'\n]*'|"[^"\n]*"|(?<![\w.])[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*""")
CALL = re.compile(r"\s*\(")
HIGHLIGHTED = re.compile(r'\.<span class="\w+">([A-Za-z_]\w*)</span>')
TRACKED_TAGS = {"main", "pre", "code", "a", "span"}
SKIP_CLASSES = {
    "co",
    "st",
    "ss",
    "ch",
    "no-api-link",
    "parameter-name",
}


ALIASES = {"rd": "reducers", "rdl": "reducers.lowlevel"}


class CodeLinks(HTMLParser):
    """Record link insertions without serializing or changing the source HTML."""

    def __init__(self, html: str, targets: dict[str, str]) -> None:
        super().__init__(convert_charrefs=False)
        self.html = html
        self.targets = targets
        self.context: list[tuple[str, set[str]]] = []
        self.line_offsets = [0, *(m.end() for m in re.finditer("\n", html))]
        self.edits: list[tuple[int, int, str]] = []

    def handle_starttag(self, tag, attrs):
        if tag in TRACKED_TAGS:
            classes = (dict(attrs).get("class") or "").split()
            self.context.append((tag, set(classes)))

    def handle_endtag(self, tag):
        if self.context and self.context[-1][0] == tag:
            self.context.pop()

    def handle_data(self, data):
        tags = {tag for tag, _ in self.context}
        if not {"main", "code"} <= tags or "a" in tags:
            return
        classes = set().union(*(classes for _, classes in self.context))
        if classes & SKIP_CLASSES:
            return
        block = "pre" in tags
        if block and "python" not in classes:
            return
        line, col = self.getpos()
        offset = self.line_offsets[line - 1] + col
        for match in NAME.finditer(data):
            name = match.group()
            start = offset + match.start()
            end = offset + match.end()
            if self.edits and start < self.edits[-1][1]:
                continue
            highlighted = HIGHLIGHTED.match(self.html, end)
            if highlighted:
                name += "." + highlighted[1]
                end = highlighted.end()
            href = self.targets.get(name)
            if not href:
                continue
            call = CALL.match(self.html, end) is not None
            if not call and "parameter-annotation" not in classes:
                if block and "." not in name and name[0].islower():
                    continue
                if not block and "." not in name and data.strip() != name:
                    continue
            markup = self.html[start:end]
            link = (
                f'<a class="api-symbol" href="{escape(href, quote=True)}">{markup}</a>'
            )
            self.edits.append((start, end, link))


def example_methods(
    html: str, targets: dict[str, str], classes: set[str]
) -> dict[str, str]:
    """Resolve methods when a page assigns a variable one documented class."""
    bindings = {}
    members = set()
    for block in re.findall(
        r'<code class="sourceCode python[^"]*">(.*?)</code>', html, re.S
    ):
        source = unescape(re.sub(r"<[^>]*>", "", block))
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for statement in tree.body:
            if not isinstance(statement, ast.Assign):
                continue
            value = statement.value
            constructor = ast.unparse(value.func) if isinstance(value, ast.Call) else ""
            for variable in statement.targets:
                if isinstance(variable, ast.Name):
                    bindings.setdefault(variable.id, set()).add(constructor)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                members.add((node.value.id, node.attr))
    resolved = {}
    for variable, member in members:
        owners = bindings.get(variable, set())
        if len(owners) == 1:
            owner = next(iter(owners))
            method = f"{owner}.{member}"
            if targets.get(owner) in classes and method in targets:
                resolved[f"{variable}.{member}"] = targets[method]
    return resolved


def link_page(page: Path, output: Path, objects: dict, external: dict) -> None:
    """Resolve local links relative to the page and retain absolute upstream URLs."""
    project = objects["project"]
    paths = {}
    for item in objects["items"]:
        name = item["name"]
        uri = item["uri"].replace("$", name)
        if item["role"] == "attribute":
            uri = uri.partition("#")[0]
        short = name.removeprefix(project + ".")
        paths[name] = uri
        if item["dispname"] == "-":
            paths[short] = uri
            for alias, module in ALIASES.items():
                if name.startswith(module + "."):
                    paths[alias + name[len(module) :]] = uri
        if page.parent.name == "guides" and short.startswith(page.stem + "."):
            paths[short.removeprefix(page.stem + ".")] = uri
    targets = {}
    for name, uri in (external | paths).items():
        if uri.startswith("https://"):
            targets[name] = uri
        else:
            filename, sep, fragment = uri.partition("#")
            targets[name] = (
                os.path.relpath(output / filename, page.parent) + sep + fragment
            )

    html = page.read_text(encoding="utf-8")
    classes = {
        targets[item["name"]] for item in objects["items"] if item["role"] == "class"
    }
    targets.update(example_methods(html, targets, classes))
    parser = CodeLinks(html, targets)
    parser.feed(html)
    parser.close()
    if parser.edits:
        pieces = []
        start = 0
        for first, last, link in parser.edits:
            pieces.extend((html[start:first], link))
            start = last
        page.write_text("".join(pieces) + html[start:], encoding="utf-8")


if __name__ == "__main__":
    output = Path(os.environ["QUARTO_PROJECT_OUTPUT_DIR"])
    objects = json.loads(Path("objects.json").read_text(encoding="utf-8"))
    external = json.loads(
        Path(__file__).with_name("external-api-links.json").read_text(encoding="utf-8")
    )
    for filename in os.environ["QUARTO_PROJECT_OUTPUT_FILES"].splitlines():
        page = Path(filename)
        if page.suffix == ".html":
            link_page(page, output, objects, external)
