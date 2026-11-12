"""Mirage: testing tensor representations and training states."""


def main() -> None:
    """Preserve the configured console entry point without importing torch."""
    from mirage.cli import main as cli_main

    raise SystemExit(cli_main())
