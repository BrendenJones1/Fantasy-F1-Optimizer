"""Scheduled job: ingest new results, retrain when data changed, make sure prices exist."""
import logging
import os
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import select

from f1opt import prices, prospective, serving
from f1opt.config import Settings, get_settings
from f1opt.db import Constructor, Driver, make_engine, make_session_factory
from f1opt.ingestion import run_ingestion
from f1opt.jolpica import JolpicaClient

logger = logging.getLogger(__name__)


def refresh_prices(session, settings: Settings) -> None:
    """Best-effort price collection for the next race. Never raises: prices must not block results."""
    try:
        race = serving.next_race(session)
    except serving.NoUpcomingRace:
        return
    try:
        if os.environ.get("F1_FANTASY_COOKIE"):
            names = dict(session.execute(select(Driver.driver_id, Driver.name)).all())
            cnames = dict(session.execute(select(Constructor.constructor_id, Constructor.name)).all())
            source = prices.FantasyApiPriceSource(race.season, names, cnames)
            n = prices.store_prices(session, race, source.fetch(), source.name)
            logger.info("stored %d prices from %s", n, source.name)
            return
        drv, _ = prices.latest_prices(session, race)
        if not drv and settings.demo_prices:
            n = prices.store_prices(session, race, prices.demo_sheet(session, race), prices.DEMO_SOURCE)
            logger.info("no real prices for %s %d; stored %d demo prices", race.name, race.season, n)
    except Exception:  # noqa: BLE001
        logger.exception("price refresh failed; continuing without prices")


def run_cycle(settings: Settings, session_factory=None, client=None) -> None:
    session_factory = session_factory or make_session_factory(make_engine(settings.database_url))
    client = client or JolpicaClient(cache_dir=settings.cache_dir)
    run = run_ingestion(session_factory, client, settings.first_season)
    model_missing = not (Path(settings.model_dir) / serving.MODEL_FILE).exists()
    with session_factory() as session:
        if run.new_races > 0 or model_missing:
            serving.train_and_save(session, settings.model_dir)
        refresh_prices(session, settings)
        try:
            prospective.log_next_race(session, serving.ModelStore(settings.model_dir), settings.budget)
        except Exception:  # noqa: BLE001 - logging must never fail a cycle
            logger.exception("prediction logging failed")


def _safe_cycle(settings: Settings) -> None:
    try:
        run_cycle(settings)
    except Exception:  # noqa: BLE001 - the schedule must survive a failed run
        logger.exception("scheduled cycle failed; will retry at the next trigger")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = get_settings()
    logger.info("running initial cycle")
    _safe_cycle(settings)
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(_safe_cycle, CronTrigger.from_crontab(settings.ingest_cron, timezone="UTC"),
                      args=[settings], max_instances=1, coalesce=True)
    logger.info("scheduler started (cron %r UTC)", settings.ingest_cron)
    scheduler.start()


if __name__ == "__main__":
    main()
