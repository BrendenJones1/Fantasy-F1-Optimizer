"""Score logged pre-race predictions against realised results (run after ingestion)."""
from f1opt.config import get_settings
from f1opt.db import make_engine, make_session_factory
from f1opt.prospective import score_logged_races

with make_session_factory(make_engine(get_settings().database_url))() as session:
    df = score_logged_races(session)
if df.empty:
    print("no completed races with logged predictions yet")
else:
    print(df.round(3).to_string(index=False))
    print(f"\nraces: {len(df)}  mean MAE: {df['mae'].mean():.2f}  mean Spearman: {df['spearman'].mean():.3f}")
