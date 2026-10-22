"""Repository paths for the Revision 5 analysis scripts.

The preregistered sources used absolute paths on the machine where the
experiments were run. This module points the runnable copies at this
repository. The byte-identical frozen files are in
preregistration/hashed_files_as_frozen/ and are not imported from here.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = Path(__file__).resolve().parent
PIPELINE = CODE / "pipeline"
DATA_CSV = ROOT / "data" / "emotion_dataset_translated.csv"
EARLIER_OVERLAP = ROOT / "data" / "overlap_matrix_earlier_version.csv"
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
STIMULI = ROOT / "stimuli"
REFERENCE = ROOT / "reference_map"
FIGURES = ROOT / "figures"


def ensure_runtime_dirs() -> None:
    for path in (
        CACHE,
        FIGURES,
        RESULTS,
        RESULTS / "experiments",
        RESULTS / "baseline_zh",
        RESULTS / "baseline_en",
        REFERENCE,
        STIMULI,
    ):
        path.mkdir(parents=True, exist_ok=True)


def text_open(file, mode="r", **kwargs):
    """Open text as UTF-8 with newline '\\n', including on Windows."""
    kwargs.setdefault("encoding", "utf-8")
    kwargs.setdefault("newline", "\n")
    return open(file, mode, **kwargs)


def load_nrc_lexicon() -> dict:
    """NRC Emotion Lexicon as bundled in nrclex 4.1.0.

    Returns a dict of lower-cased word to a list of emotion labels. The
    analysis uses lower-cased exact matching and no lemmatisation, which is
    the scorer named in Supplementary Table S4.
    """
    try:
        import nrclex
    except ImportError as exc:
        raise ImportError(
            "nrclex==4.1.0 is required for the NRC scorer in Families 3 and 4. "
            "Install it with: pip install nrclex==4.1.0"
        ) from exc
    from importlib.resources import files

    path = files("nrclex").joinpath("data", "nrc_en.json")
    return json.loads(path.read_text(encoding="utf-8"))


def apply_figure_style() -> str:
    """Prefer the serif face used for the manuscript figures when it is installed."""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    names = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ("Liberation Serif", "Times New Roman", "Nimbus Roman", "DejaVu Serif"):
        if candidate in names:
            plt.rcParams["font.family"] = candidate
            return candidate
    plt.rcParams["font.family"] = "serif"
    return "serif"
