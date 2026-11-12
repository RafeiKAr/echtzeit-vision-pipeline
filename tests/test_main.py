"""Tests – ausführen mit: pytest"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import main  # noqa: E402


def test_main_laeuft(capsys):
    main.main()
    assert "läuft" in capsys.readouterr().out
