from __future__ import annotations

"""Simple entrypoint to run the full pipeline inside containers.

Delegates to main.main() which orchestrates preprocessing → training → evaluation
with logging and writes outputs to models/ and results/.
"""

from main import main  # noqa: E402

if __name__ == "__main__":
    main()
