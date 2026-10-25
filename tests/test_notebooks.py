"""The large-galaxy notebook reuses the walkthrough's code cells: each cell tagged
same-as-walkthrough must match one of the walkthrough's exactly, so an edit to one notebook
can't silently miss the other. The other cells (steps 6-8 and 12) do the same jobs in a
different way: after changing either notebook, check the other one too and carry the change
over where it applies."""
import json
from pathlib import Path

NOTEBOOKS = Path(__file__).parents[1] / "notebooks"


def code_cells(path):
    nb = json.loads(path.read_text())
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def test_shared_cells_match_the_walkthrough():
    walkthrough = {"".join(c["source"]) for c in code_cells(NOTEBOOKS / "walkthrough.ipynb")}
    shared = [c for c in code_cells(NOTEBOOKS / "walkthrough_large_galaxy.ipynb")
              if "same-as-walkthrough" in c.get("metadata", {}).get("tags", [])]
    assert len(shared) >= 10
    for cell in shared:
        source = "".join(cell["source"])
        assert source in walkthrough, f"cell starting {source.splitlines()[0]!r} differs from the walkthrough's"
