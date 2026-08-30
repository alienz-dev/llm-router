"""Re-verify which models support tool calling and strict JSON-schema output.

Talks directly to each provider's endpoint — never through the router — so the
result is ground truth about the model, not about our own plumbing. Writes both
the DB columns and a dated inventory file, which is what seeds a fresh database.

    uv run python scripts/probe_capabilities.py                    # models due for a re-check
    uv run python scripts/probe_capabilities.py --all              # ignore freshness
    uv run python scripts/probe_capabilities.py --out docs/x.json  # inventory path

Two calls per model. Against OpenRouter's 50 requests/day free cap that is real
money, so the default only touches models whose reading is missing or stale.
"""
import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from llm_router.capabilities import (  # noqa: E402
    CAPABILITY_TTL_DAYS, PROBE_BUDGET_PER_PROVIDER, run_capability_probe,
    seed_from_inventory,
)
from llm_router.db import close_db, get_db  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true",
                        help="probe every active model, ignoring freshness")
    parser.add_argument("--budget", type=int, default=PROBE_BUDGET_PER_PROVIDER,
                        help="max models per provider this run")
    parser.add_argument("--out", default=None,
                        help="write an inventory file (default: docs/model-capabilities-<today>.json)")
    parser.add_argument("--seed", action="store_true",
                        help="apply the checked-in inventory first, then stop")
    args = parser.parse_args()

    await get_db()
    seeded = await seed_from_inventory()
    if seeded:
        print(f"seeded {seeded} models from the checked-in inventory")
    if args.seed:
        await close_db()
        return 0

    stats = await run_capability_probe(
        ttl_days=0 if args.all else CAPABILITY_TTL_DAYS,
        budget_per_provider=args.budget,
    )
    for r in stats["results"]:
        print(f"{r['provider']:12} {r['model'][:46]:46} "
              f"tools={r['tools']:14} json={r['json']}", flush=True)
    print(f"\nprobed {stats['probed']} models, updated {stats['updated']}")

    if stats["results"]:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out = Path(args.out or ROOT / "docs" / f"model-capabilities-{today}.json")
        out.write_text(json.dumps(stats["results"], indent=2))
        print(f"wrote {out}")

    await close_db()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
