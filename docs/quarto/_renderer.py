"""Render NumPy-style API fields using their documented type notation."""

from __future__ import annotations

import re
from copy import copy

from griffe import Docstring, ParameterKind
from plum import dispatch
from quartodoc import MdRenderer


def _documented_fields(doc: Docstring) -> Docstring:
    """Keep field types as prose and use each parameter's own signature default."""
    # Griffe parses array-like as subtraction and {a, b} as a tuple with default a.
    # Keep its section/description parsing and change only copies of the fields.
    sections = re.split(r"(?m)^([^\s].*)\n-{3,}[ \t]*$", doc.value)
    headers = {
        title.lower(): [
            line for line in body.splitlines() if line and not line[0].isspace()
        ]
        for title, body in zip(sections[1::2], sections[2::2], strict=True)
    }
    result = copy(doc)
    result.parsed = []
    signature = getattr(doc.parent, "parameters", {})
    for section in doc.parsed:
        kind = section.kind.value
        lines = headers.get(kind, [])
        if not lines or kind not in {
            "parameters",
            "other parameters",
            "attributes",
            "returns",
            "yields",
        }:
            result.parsed.append(section)
            continue
        named = {}
        types = []
        for line in lines:
            names, sep, annotation = line.partition(" : ")
            types.append(annotation if sep else line)
            if sep:
                for name in names.split(","):
                    named[name.strip()] = annotation.strip()
        section = copy(section)
        section.value = [copy(item) for item in section.value]
        for index, field in enumerate(section.value):
            annotation = named.get(field.name)
            if kind in {"returns", "yields"} and len(types) == len(section.value):
                annotation = types[index]
            if annotation:
                default = None
                match = re.fullmatch(
                    r"(.+?),\s*default\b\s*[:=]?\s*(.+)", annotation, re.I
                )
                if match:
                    annotation, default = match.groups()
                field.annotation = re.sub(
                    r"(?<![\w.'\"])array_like(?![\w'\"])",
                    "array-like",
                    re.sub(r",\s*optional\.?$", "", annotation),
                )
                if kind in {"parameters", "other parameters"}:
                    field.value = default or None
            if kind in {"parameters", "other parameters"} and field.name in signature:
                parameter = signature[field.name]
                field.value = parameter.default
                if parameter.kind in {
                    ParameterKind.var_positional,
                    ParameterKind.var_keyword,
                }:
                    field.value = None
                    field.name = field.name.replace("*", r"\*")
        result.parsed.append(section)
    return result


class Renderer(MdRenderer):
    """Render readable parameter declarations in API definition lists."""

    style = "reducers-numpy-fields"

    @dispatch
    def render(self, doc: Docstring) -> str:
        rendered = super().render(_documented_fields(doc))
        return rendered.replace(
            " [ = ]{.parameter-default-sep}", "[, default =]{.parameter-default-sep}"
        )
