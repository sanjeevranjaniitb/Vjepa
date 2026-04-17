"""
Launch both VJepa2 demos simultaneously on different ports.

  - POC Incident Detection : http://localhost:7860
  - Feature Explorer       : http://localhost:7861
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
APPS = [
    ("POC Incident Detection", "poc_incident_detection.py", 7860),
    ("Feature Explorer", "vjepa2_feature_explorer.py", 7861),
]


def main():
    procs = []
    for name, script, port in APPS:
        print(f"Starting {name} on port {port}...")
        p = subprocess.Popen(
            [sys.executable, str(ROOT / script)],
            env={**__import__("os").environ, "GRADIO_SERVER_PORT": str(port)},
        )
        procs.append((name, p))

    print(f"\n  POC Incident Detection : http://localhost:7860")
    print(f"  Feature Explorer       : http://localhost:7861\n")

    try:
        for name, p in procs:
            p.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        for name, p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
