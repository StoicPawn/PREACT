from datetime import datetime,timezone

from preact.history.coverage import build_coverage_report
from preact.history.document_store import HistoricalDocumentStore
from preact.history.graph_store import HistoricalGraphStore
from preact.history.schema import Provenance,TemporalRecord
from preact.history.warehouse import HistoricalWarehouse

UTC=timezone.utc

def test_coverage_reports_real_ingestion(tmp_path):
    w=HistoricalWarehouse(tmp_path/"h.duckdb")
    g=HistoricalGraphStore(tmp_path/"g.duckdb")
    d=HistoricalDocumentStore(tmp_path/"d.duckdb")
    now=datetime(2026,1,1,tzinfo=UTC)
    w.insert_records([TemporalRecord(
        record_id="x",entity_id="iso3:ITA",variable="x",value=1,
        valid_from=now,known_at=now,
        provenance=Provenance(source="world_bank",source_ref="x",retrieved_at=now)
    )])
    rows={r.source_id:r for r in build_coverage_report(w,g,d)}
    assert rows["world_bank"].status=="ingested"
    assert rows["world_bank"].records==1
    assert rows["gdelt"].status=="not_ingested"
