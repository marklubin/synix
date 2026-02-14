"""Copy staged intel update into sources to trigger cascade rebuild."""

import shutil
from pathlib import Path

here = Path(__file__).parent
shutil.copy(
    here / "sources" / "staged" / "acme_q1_update.md",
    here / "sources" / "competitors" / "acme_q1_update.md",
)
print("Copied acme_q1_update.md into sources/competitors/")
