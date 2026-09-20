"""Coverage/quality reporting across PREACT evidence stores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .document_store import HistoricalDocumentStore
from .graph_store import HistoricalGraphStore
from .source_catalog import SOURCE_BY_ID,SOURCES
from .warehouse import HistoricalWarehouse


@dataclass(frozen=True)
class SourceCoverage:
    source_id:str
    name:str
    access:str
    replay_policy:str
    records:int=0
    relations:int=0
    documents:int=0
    earliest_valid:object|None=None
    latest_valid:object|None=None
    latest_retrieved:object|None=None

    @property
    def status(self)->str:
        total=self.records+self.relations+self.documents
        return "ingested" if total>0 else "not_ingested"


def build_coverage_report(
    warehouse:HistoricalWarehouse,
    graph:HistoricalGraphStore,
    documents:HistoricalDocumentStore,
)->list[SourceCoverage]:
    records={}
    with warehouse.connect() as conn:
        for row in conn.execute("""
            SELECT source,COUNT(*),MIN(valid_from),MAX(valid_from),MAX(retrieved_at)
            FROM temporal_records GROUP BY source
        """).fetchall():
            records[row[0]]=row[1:]
    relations={}
    with graph.connect() as conn:
        for row in conn.execute("""
            SELECT source,COUNT(*),MIN(valid_from),MAX(valid_from),MAX(retrieved_at)
            FROM historical_relations GROUP BY source
        """).fetchall():
            relations[row[0]]=row[1:]
    docs={}
    with documents.connect() as conn:
        for row in conn.execute("""
            SELECT source_id,COUNT(*),MIN(published_at),MAX(published_at),MAX(acquired_at)
            FROM historical_documents GROUP BY source_id
        """).fetchall():
            docs[row[0]]=row[1:]

    source_ids=sorted(set(SOURCE_BY_ID)|set(records)|set(relations)|set(docs))
    output=[]
    for source_id in source_ids:
        spec=SOURCE_BY_ID.get(source_id)
        r=records.get(source_id,(0,None,None,None))
        g=relations.get(source_id,(0,None,None,None))
        d=docs.get(source_id,(0,None,None,None))
        dates=[x for x in (r[1],g[1],d[1]) if x is not None]
        latest=[x for x in (r[2],g[2],d[2]) if x is not None]
        retrieved=[x for x in (r[3],g[3],d[3]) if x is not None]
        output.append(SourceCoverage(
            source_id=source_id,
            name=spec.name if spec else source_id,
            access=spec.access if spec else "unknown",
            replay_policy=spec.replay_policy if spec else "unknown",
            records=int(r[0]),
            relations=int(g[0]),
            documents=int(d[0]),
            earliest_valid=min(dates) if dates else None,
            latest_valid=max(latest) if latest else None,
            latest_retrieved=max(retrieved) if retrieved else None,
        ))
    return output
