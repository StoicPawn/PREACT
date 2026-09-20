"""Run implemented Wave-1 source ingestion into the historical warehouse."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os

from preact.data_hub.gateway import SharedProviderGateway
from preact.history.ingestion_state import IngestionStateStore
from preact.history.warehouse import HistoricalWarehouse
from preact.history.wave1_pipeline import Wave1IngestionPipeline


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=["world_bank", "unhcr", "ucdp"])
    parser.add_argument("--hub-root", default=os.getenv("SHARED_DATA_HUB_ROOT", "data/shared_hub"))
    parser.add_argument("--history-db", default=os.getenv("PREACT_HISTORY_DB", "data/history/preact_history.duckdb"))
    parser.add_argument("--state-db", default=os.getenv("PREACT_INGESTION_STATE", "data/history/ingestion_state.sqlite3"))

    parser.add_argument("--country")
    parser.add_argument("--indicator")
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)

    parser.add_argument("--dataset", default="population")
    parser.add_argument("--country-origin")
    parser.add_argument("--country-asylum")
    parser.add_argument("--all-origins", action="store_true")
    parser.add_argument("--all-asylum", action="store_true")

    parser.add_argument("--resource", default="gedevents")
    parser.add_argument("--version")
    parser.add_argument("--version-release-at")
    parser.add_argument("--max-pages", type=int)
    args = parser.parse_args()

    pipeline = Wave1IngestionPipeline(
        gateway=SharedProviderGateway(args.hub_root),
        warehouse=HistoricalWarehouse(args.history_db),
        state=IngestionStateStore(args.state_db),
    )

    if args.source == "world_bank":
        if not all([args.country, args.indicator, args.start_year, args.end_year]):
            parser.error("world_bank requires --country --indicator --start-year --end-year")
        result = pipeline.ingest_world_bank(
            country=args.country,
            indicator=args.indicator,
            start_year=args.start_year,
            end_year=args.end_year,
        )
    elif args.source == "unhcr":
        result = pipeline.ingest_unhcr(
            dataset=args.dataset,
            year_from=args.start_year,
            year_to=args.end_year,
            country_origin=args.country_origin,
            country_asylum=args.country_asylum,
            include_all_origins=args.all_origins,
            include_all_asylum=args.all_asylum,
            max_pages=args.max_pages,
        )
    else:
        if not args.version:
            parser.error("ucdp requires --version")
        result = pipeline.ingest_ucdp(
            resource=args.resource,
            version=args.version,
            version_release_at=parse_datetime(args.version_release_at),
            max_pages=args.max_pages,
        )

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
