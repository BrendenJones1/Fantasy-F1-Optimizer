"""Run one ingestion cycle (ingest, retrain if new races, refresh prices) and exit."""
import logging

from f1opt.config import get_settings
from f1opt.scheduler import run_cycle

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
run_cycle(get_settings())
