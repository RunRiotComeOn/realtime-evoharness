"""Prompt templates per language: `get("en")` or `get("zh")` returns a module of template strings."""
import importlib


def get(lang: str):
    try:
        return importlib.import_module(f"{__name__}.{lang}")
    except ModuleNotFoundError:
        raise ValueError(f"no prompt templates for language {lang!r} (available: en, zh)")


def fill(template: str, **values) -> str:
    for k, v in values.items():
        template = template.replace("{" + k + "}", str(v))
    return template
