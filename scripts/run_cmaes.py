#!/usr/bin/env python3
"""Run the CMA-ES optimizer on the physical-modulation study."""

from __future__ import annotations

from scs_search.search.optimizer_cli import main


if __name__ == "__main__":
    main("cmaes")
