"""Load Fantasy prices for a race from a CSV (columns: entity,id,price) or generate demo prices."""
import argparse

from sqlalchemy import select

from f1opt import prices, serving
from f1opt.config import get_settings
from f1opt.db import Race, make_engine, make_session_factory

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--csv", help="CSV file with entity,id,price rows")
parser.add_argument("--demo", action="store_true", help="store illustrative demo prices")
parser.add_argument("--source", default="csv")
parser.add_argument("--season", type=int)
parser.add_argument("--round", type=int)
args = parser.parse_args()
if bool(args.csv) == args.demo:
    parser.error("give exactly one of --csv or --demo")

with make_session_factory(make_engine(get_settings().database_url))() as session:
    if args.season and args.round:
        race = session.scalar(select(Race).where(Race.season == args.season, Race.round == args.round))
    else:
        race = serving.next_race(session)
    sheet = prices.demo_sheet(session, race) if args.demo else prices.CsvPriceSource(args.csv).fetch()
    n = prices.store_prices(session, race, sheet, prices.DEMO_SOURCE if args.demo else args.source)
    print(f"stored {n} prices for {race.name} {race.season} (round {race.round})")
