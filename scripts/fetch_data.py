"""Download race data from Jolpica-F1 and write data/processed/weekends.csv."""
import argparse
import logging
from pathlib import Path

from f1opt.dataset import build_weekend_table
from f1opt.jolpica import JolpicaClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--start", type=int, default=2018)
parser.add_argument("--end", type=int, default=2026)
parser.add_argument("--refresh", type=int, nargs="*", default=[],
                    help="seasons to re-download even if cached (use for the in-progress season)")
parser.add_argument("--out", default="data/processed/weekends.csv")
args = parser.parse_args()

df = build_weekend_table(JolpicaClient(), range(args.start, args.end + 1), args.refresh)
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"wrote {out}: {len(df)} rows")
