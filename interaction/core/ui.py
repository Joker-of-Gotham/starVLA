from __future__ import annotations

import re
import sys
import time
from typing import Any


MARKUP_RE = re.compile(
    r"\[/?(?:bold|dim|italic|underline|blink|reverse|red|green|yellow|blue|magenta|cyan|white|black)(?: [a-zA-Z]+)*\]"
)


def console() -> Any | None:
    try:
        from rich.console import Console

        return Console(highlight=False)
    except Exception:
        return None


def remove_markup(message: str) -> str:
    return MARKUP_RE.sub("", message)


def cprint(message: str) -> None:
    rich_console = console()
    if rich_console:
        rich_console.print(message)
    else:
        print(remove_markup(message))


def status_markup(value: Any, ok_text: str = "OK", fail_text: str = "FAIL") -> str:
    if isinstance(value, bool):
        return f"[green]{ok_text}[/green]" if value else f"[red]{fail_text}[/red]"
    text = str(value)
    lower = text.lower()
    if lower in {"ok", "pass", "passed", "alive", "running", "true"}:
        return f"[green]{text}[/green]"
    if lower in {"warn", "warning", "degraded"}:
        return f"[yellow]{text}[/yellow]"
    if lower in {"fail", "failed", "error", "fatal", "false", "stopped"}:
        return f"[red]{text}[/red]"
    return text


def severity_style(severity: str | None) -> str:
    if severity == "fatal":
        return "bold red"
    if severity in {"warning", "warn"}:
        return "bold yellow"
    if severity == "ok":
        return "bold green"
    return "cyan"


def human_age(timestamp: float | None) -> str:
    if not timestamp:
        return "-"
    seconds = max(time.time() - timestamp, 0.0)
    if seconds < 60:
        return f"{seconds:.0f}s ago"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m ago"
    hours = minutes / 60
    if hours < 48:
        return f"{hours:.1f}h ago"
    return f"{hours / 24:.1f}d ago"


def human_bytes(size: int | float | None) -> str:
    if size is None:
        return "-"
    value = float(size)
    units = ["B", "KB", "MB", "GB", "TB"]
    unit = units[0]
    for unit in units:
        if abs(value) < 1024 or unit == units[-1]:
            break
        value /= 1024
    if unit == "B":
        return f"{int(value)} {unit}"
    return f"{value:.1f} {unit}"


def print_table(title: str, columns: list[str], rows: list[list[Any]], styles: list[str] | None = None) -> None:
    rich_console = console()
    if rich_console:
        from rich.table import Table

        table = Table(title=title, expand=False, show_lines=False)
        styles = styles or []
        for index, col in enumerate(columns):
            kwargs = {"overflow": "fold"}
            if index < len(styles) and styles[index]:
                kwargs["style"] = styles[index]
            table.add_column(col, **kwargs)
        for row in rows:
            table.add_row(*(str(item) for item in row))
        rich_console.print(table)
        return

    print(title)
    print("\t".join(columns))
    for row in rows:
        print("\t".join(remove_markup(str(item)) for item in row))


def clear_and_print(text: str) -> None:
    if sys.stdout.isatty():
        sys.stdout.write("\033[H\033[J")
    print(remove_markup(text))
    sys.stdout.flush()
