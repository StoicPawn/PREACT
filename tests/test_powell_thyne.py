from datetime import datetime, timezone

from preact.history.connectors.powell_thyne import PowellThyneCoupConnector, PowellThyneRelease
from preact.projections.powell_thyne import powell_thyne_records

UTC=timezone.utc


def test_powell_thyne_csv_and_vintage_semantics():
    payload=(
        "country,ccode,year,month,day,coup\n"
        "Example,999,2020,5,2,1\n"
        "Example,999,2021,6,3,2\n"
    ).encode()
    rows=PowellThyneCoupConnector.parse(payload,format="csv")
    release=PowellThyneRelease(
        "2022-01-01","https://example.test/coups.csv",
        datetime(2022,1,1,tzinfo=UTC),"csv"
    )
    records=powell_thyne_records(
        rows,release=release,retrieved_at=datetime(2026,1,1,tzinfo=UTC)
    )
    attempts=[x for x in records if x.variable=="event:coup_attempt"]
    successes=[x for x in records if x.variable=="event:coup_success"]
    assert len(attempts)==2
    assert len(successes)==1
    assert attempts[0].entity_id=="cow_ccode:999"
    assert all(x.known_at==release.published_at for x in records)
