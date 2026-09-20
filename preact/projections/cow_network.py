"""Normalize COW network/capability releases into PREACT evidence."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from typing import Iterable, Mapping

from preact.history.relations import HistoricalRelation
from preact.history.schema import EvidenceClass, Provenance, TemporalRecord


def _lower(row: Mapping[str, str]) -> dict[str, str]:
    return {str(k).strip().lower(): str(v).strip() for k, v in row.items()}


def _first(row: Mapping[str, str], *keys: str) -> str:
    lowered=_lower(row)
    for key in keys:
        value=lowered.get(key.lower(),"")
        if value:
            return value
    return ""


def _year(row: Mapping[str, str]) -> int | None:
    raw=_first(row,"year","yr")
    try:
        return int(float(raw))
    except (TypeError,ValueError):
        return None


def _cow_entity(code: str) -> str:
    return f"cow_ccode:{str(code).strip()}"


def _relation_id(kind: str, row: Mapping[str,str]) -> str:
    digest=sha256(repr(sorted((str(k),str(v)) for k,v in row.items())).encode()).hexdigest()
    return f"cow:{kind}:sha256:{digest}"


def cow_alliance_relations(
    rows: Iterable[Mapping[str,str]],
    *,
    known_at: datetime,
    retrieved_at: datetime,
) -> list[HistoricalRelation]:
    output=[]
    for row in rows:
        year=_year(row)
        c1=_first(row,"ccode1","state1no","state1")
        c2=_first(row,"ccode2","state2no","state2")
        if year is None or not c1 or not c2:
            continue
        output.append(HistoricalRelation(
            relation_id=_relation_id("alliance",row),
            relation_type="formal_alliance",
            subject_entity_id=_cow_entity(c1),
            object_entity_id=_cow_entity(c2),
            valid_from=datetime(year,1,1,tzinfo=timezone.utc),
            valid_to=datetime(year+1,1,1,tzinfo=timezone.utc),
            known_at=known_at,
            directed=True,
            source="cow",
            source_ref=_first(row,"alliance","allianceid","dyad") or _relation_id("alliance-ref",row),
            retrieved_at=retrieved_at,
            dataset_version="Formal Alliances v4.1",
            attributes=_lower(row),
        ))
    return output


def cow_contiguity_relations(
    rows: Iterable[Mapping[str,str]],
    *,
    known_at: datetime,
    retrieved_at: datetime,
) -> list[HistoricalRelation]:
    output=[]
    for row in rows:
        year=_year(row)
        c1=_first(row,"state1no","ccode1","state1")
        c2=_first(row,"state2no","ccode2","state2")
        if year is None or not c1 or not c2:
            continue
        output.append(HistoricalRelation(
            relation_id=_relation_id("contiguity",row),
            relation_type="direct_contiguity",
            subject_entity_id=_cow_entity(c1),
            object_entity_id=_cow_entity(c2),
            valid_from=datetime(year,1,1,tzinfo=timezone.utc),
            valid_to=datetime(year+1,1,1,tzinfo=timezone.utc),
            known_at=known_at,
            directed=True,
            source="cow",
            source_ref=f"{c1}:{c2}:{year}",
            retrieved_at=retrieved_at,
            dataset_version="Direct Contiguity v3.2",
            attributes=_lower(row),
        ))
    return output


def cow_mid_relations(
    rows: Iterable[Mapping[str,str]],
    *,
    known_at: datetime,
    retrieved_at: datetime,
) -> list[HistoricalRelation]:
    output=[]
    for row in rows:
        lowered=_lower(row)
        c1=_first(row,"ccode1","statea","state1")
        c2=_first(row,"ccode2","stateb","state2")
        start_year=None
        for key in ("strtyr","styear","year","begyear"):
            try:
                start_year=int(float(lowered.get(key,"")))
                break
            except (TypeError,ValueError):
                continue
        if start_year is None or not c1 or not c2:
            continue
        end_year=start_year
        for key in ("endyear","endyr"):
            try:
                end_year=int(float(lowered.get(key,"")))
                break
            except (TypeError,ValueError):
                continue
        output.append(HistoricalRelation(
            relation_id=_relation_id("mid",row),
            relation_type="militarized_interstate_dispute",
            subject_entity_id=_cow_entity(c1),
            object_entity_id=_cow_entity(c2),
            valid_from=datetime(start_year,1,1,tzinfo=timezone.utc),
            valid_to=datetime(end_year+1,1,1,tzinfo=timezone.utc),
            known_at=known_at,
            directed=False,
            source="cow",
            source_ref=_first(row,"dispnum","dyadnum","mid") or _relation_id("mid-ref",row),
            retrieved_at=retrieved_at,
            dataset_version="Dyadic MID v4.03",
            attributes=lowered,
        ))
    return output


def cow_nmc_records(
    rows: Iterable[Mapping[str,str]],
    *,
    known_at: datetime,
    retrieved_at: datetime,
) -> list[TemporalRecord]:
    metrics=("irst","milex","milper","pec","energy","tpop","upop","cinc")
    output=[]
    for row in rows:
        lowered=_lower(row)
        year=_year(row)
        ccode=_first(row,"ccode")
        if year is None or not ccode:
            continue
        for metric in metrics:
            raw=lowered.get(metric)
            if raw in (None,"","-9","-9.0"):
                continue
            try:
                value=float(raw)
            except ValueError:
                continue
            output.append(TemporalRecord(
                record_id=f"cow:nmc:{ccode}:{year}:{metric}:v7",
                entity_id=_cow_entity(ccode),
                variable=f"cow_nmc:{metric}",
                value=value,
                valid_from=datetime(year,1,1,tzinfo=timezone.utc),
                valid_to=datetime(year+1,1,1,tzinfo=timezone.utc),
                known_at=known_at,
                evidence_class=EvidenceClass.OBSERVATION,
                provenance=Provenance(
                    source="cow",
                    source_ref=f"NMCv7:{ccode}:{year}:{metric}",
                    retrieved_at=retrieved_at,
                    dataset_version="NMC v7.0",
                    notes="Retrospective COW release; strict replay before known_at is blocked.",
                ),
                attributes={"stateabb":lowered.get("stateabb"),"ccode":ccode},
            ))
    return output
