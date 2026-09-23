"""Talk to a local `gutcheck serve` with the exact request a Jev client sends. Only the base URL changes.

    gutcheck serve &            # http://127.0.0.1:8765
    python examples/jev_compatible_client.py
"""
import json
import urllib.request

BASE = "http://127.0.0.1:8765"   # instead of https://api.typesafe.ai

req = {
    "model": "jev-latest",        # mapped to the server's default local model
    "state": {"subject": "Invoice INV-2291", "body": "The total is $4,120 but the PO says $3,900. Please fix."},
    "questions": {
        "matches_po": {"type": "noul", "instructions": "Does the invoice amount match the purchase order?"},
        "action": {"type": "choice", "instructions": "What should accounts payable do?",
                   "criteria": {"approve": "pay as is", "dispute": "ask the vendor to correct it", "escalate": "needs a human"}},
    },
}
r = urllib.request.Request(BASE + "/v1/systemone", data=json.dumps(req).encode(), headers={"content-type": "application/json"})
with urllib.request.urlopen(r) as resp:
    print(json.dumps(json.loads(resp.read()), indent=2))
