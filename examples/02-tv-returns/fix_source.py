"""Scrub PII from source data — fix at the source, not the artifact."""
import json
from pathlib import Path

p = Path("fixtures/vendor_offers.json")
data = json.loads(p.read_text())
for offer in data:
    if "notes" in offer:
        offer["notes"] = offer["notes"].replace("john.smith@gmail.com", "A customer")
p.write_text(json.dumps(data, indent=2) + "\n")
print("Scrubbed PII from vendor_offers.json")
