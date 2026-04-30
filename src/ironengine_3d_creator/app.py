"""Application entry point."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ironengine-3d-creator")
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=None,
        help="Override the export target directory for this session.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Quiet down PyOpenGL's accelerate-not-loaded informational notice — it's
    # a benign optional optimisation and clutters startup logs.
    logging.getLogger("OpenGL.acceleratesupport").setLevel(logging.WARNING)

    # Late imports keep startup fast and let `--help` work without Qt installed.
    from PySide6.QtWidgets import QApplication

    from .ui.main_window import MainWindow
    from .ui.theme import set_theme, stylesheet, current as theme_current
    from .core.settings import Settings

    settings = Settings.load()
    set_theme(settings.get("ui", "theme", default="cyber_neon"))

    # Pin the chosen acceleration backend up-front so workers and generators
    # don't each re-detect. Auto resolves to the best available GPU backend.
    from .core.resources import set_active_backend
    set_active_backend(
        settings.get("resources", "backend", default="auto"),
        prefer_gpu=bool(settings.get("resources", "prefer_gpu", default=True)),
    )

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("IronEngine 3D Creator")
    app.setOrganizationName("IronEngine")
    app.setStyleSheet(stylesheet(theme_current()))

    window = MainWindow(target_dir_override=args.target_dir)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
