from datetime import datetime, timezone

from preact.feature_store.news_context import build_news_context_snapshot
from preact.history.warehouse import HistoricalWarehouse
from preact.projections.gdelt_history import gdelt_event_records


UTC = timezone.utc


def test_news_context_uses_global_and_focal_events_without_future_knowledge(tmp_path):
    warehouse = HistoricalWarehouse(tmp_path / "history.duckdb")
    acquired = datetime(2020, 1, 10, tzinfo=UTC)
    rows = [
        {
            "GLOBALEVENTID": "1",
            "SQLDATE": "20200105",
            "DATEADDED": "20200105120000",
            "ActionGeo_CountryCode": "IT",
            "QuadClass": "4",
            "GoldsteinScale": "-8",
            "AvgTone": "-5",
            "NumArticles": "10",
            "EventRootCode": "19",
        },
        {
            "GLOBALEVENTID": "2",
            "SQLDATE": "20200106",
            "DATEADDED": "20200106120000",
            "ActionGeo_CountryCode": "FR",
            "QuadClass": "1",
            "GoldsteinScale": "6",
            "AvgTone": "3",
            "NumArticles": "4",
            "EventRootCode": "04",
        },
        {
            "GLOBALEVENTID": "3",
            "SQLDATE": "20200107",
            "DATEADDED": "20210101120000",
            "ActionGeo_CountryCode": "IT",
            "QuadClass": "4",
            "GoldsteinScale": "-10",
            "AvgTone": "-9",
            "NumArticles": "100",
            "EventRootCode": "20",
        },
    ]
    warehouse.insert_records(
        gdelt_event_records(
            rows,
            acquired_at=acquired,
            fips_to_iso3={"IT": "ITA", "FR": "FRA"},
        )
    )

    snapshot = build_news_context_snapshot(
        warehouse,
        cutoff=datetime(2020, 1, 8, tzinfo=UTC),
        windows_days=(30,),
    )
    italy = snapshot.features_for("iso3:ITA")
    france = snapshot.features_for("iso3:FRA")

    assert italy["news_context:system_30d:events"] == 2.0
    assert italy["news_context:focal_30d:events"] == 1.0
    assert france["news_context:focal_30d:events"] == 1.0
    assert italy["news_context:system_30d:conflict_pressure"] > 0
    assert italy["news_context:system_30d:cooperation_pressure"] > 0
    assert italy["news_context:focal_30d:conflict_pressure"] > 0
    assert "news_context:system_30d:root_20" not in italy
    assert len(snapshot.evidence_fingerprint) == 64


def test_news_context_retrospective_mode_can_see_later_known_event(tmp_path):
    from preact.history.schema import KnowledgeMode

    warehouse = HistoricalWarehouse(tmp_path / "history.duckdb")
    warehouse.insert_records(
        gdelt_event_records(
            [
                {
                    "GLOBALEVENTID": "late",
                    "SQLDATE": "20200102",
                    "DATEADDED": "20210101120000",
                    "ActionGeo_CountryCode": "IT",
                    "GoldsteinScale": "-5",
                    "NumArticles": "3",
                }
            ],
            acquired_at=datetime(2021, 1, 1, tzinfo=UTC),
            fips_to_iso3={"IT": "ITA"},
        )
    )

    strict = build_news_context_snapshot(
        warehouse,
        cutoff=datetime(2020, 1, 3, tzinfo=UTC),
        windows_days=(30,),
    )
    retrospective = build_news_context_snapshot(
        warehouse,
        cutoff=datetime(2020, 1, 3, tzinfo=UTC),
        windows_days=(30,),
        knowledge_mode=KnowledgeMode.RETROSPECTIVE,
    )

    assert strict.features_for("iso3:ITA")["news_context:system_30d:events"] == 0.0
    assert (
        retrospective.features_for("iso3:ITA")["news_context:system_30d:events"]
        == 1.0
    )
