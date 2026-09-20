"""Normalize Maddison and SIPRI historical workbooks."""

from __future__ import annotations

from datetime import datetime,timezone
from io import BytesIO
import re
from typing import Any

import pandas as pd

from preact.history.schema import EvidenceClass,Provenance,TemporalRecord


def maddison_records(
    payload:bytes,
    *,
    known_at:datetime,
    retrieved_at:datetime,
    dataset_version:str="MPD 2023",
)->list[TemporalRecord]:
    frame=pd.read_excel(BytesIO(payload),sheet_name="Full data")
    frame.columns=[str(c).strip().lower().replace(" ","_") for c in frame.columns]
    code_col=next((c for c in ("countrycode","country_code","code") if c in frame.columns),None)
    year_col=next((c for c in ("year","yr") if c in frame.columns),None)
    if code_col is None or year_col is None:
        raise ValueError("Maddison Full data sheet lacks country code/year columns")
    variables={
        "gdppc":"maddison:gdppc",
        "pop":"maddison:population_thousands",
    }
    output=[]
    for _,row in frame.iterrows():
        code=str(row.get(code_col) or "").strip()
        try:
            year=int(row.get(year_col))
        except (TypeError,ValueError):
            continue
        if not code or year<1:
            continue
        for column,variable in variables.items():
            if column not in frame.columns:
                continue
            try:
                value=float(row[column])
            except (TypeError,ValueError):
                continue
            if pd.isna(value):
                continue
            output.append(TemporalRecord(
                record_id=f"maddison:{code}:{year}:{column}:{dataset_version}",
                entity_id=f"maddison_code:{code}",
                variable=variable,
                value=value,
                valid_from=datetime(year,1,1,tzinfo=timezone.utc),
                valid_to=datetime(year+1,1,1,tzinfo=timezone.utc),
                known_at=known_at,
                evidence_class=EvidenceClass.ESTIMATE,
                provenance=Provenance(
                    source="maddison",
                    source_ref=f"{code}:{year}:{column}",
                    retrieved_at=retrieved_at,
                    dataset_version=dataset_version,
                    transform="MPD Full data workbook -> annual estimate",
                    notes="Retrospective estimate; strict replay before known_at is blocked.",
                ),
                attributes={"country":str(row.get("country") or "").strip() or None},
            ))
    return output


_MISSING={"",". .","..","...","xxx","nan","na","n/a","—","-"}


def _numeric(value:Any)->float|None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value,(int,float)):
        return float(value)
    raw=str(value).strip()
    if raw.lower() in _MISSING:
        return None
    raw=raw.replace(",","").replace("%","").strip()
    match=re.search(r"[-+]?\d+(?:\.\d+)?",raw)
    return float(match.group(0)) if match else None


def _sipri_variable(sheet:str)->str|None:
    name=sheet.lower()
    if "share" in name and "gdp" in name:
        return "sipri:milex_share_gdp_pct"
    if "per capita" in name:
        return "sipri:milex_per_capita_usd"
    if "constant" in name:
        return "sipri:milex_constant_usd_m"
    if "current" in name:
        return "sipri:milex_current_usd_m"
    return None


def _wide_sheet_records(frame:pd.DataFrame)->tuple[int,dict[int,int]]:
    for idx,row in frame.iterrows():
        years={}
        for col,value in enumerate(row.tolist()):
            try:
                year=int(float(value))
            except (TypeError,ValueError):
                continue
            if 1900<=year<=2200:
                years[col]=year
        if len(years)>=5:
            return int(idx),years
    raise ValueError("could not find year header row")


def sipri_milex_records(
    payload:bytes,
    *,
    known_at:datetime,
    retrieved_at:datetime,
    dataset_version:str,
)->list[TemporalRecord]:
    workbook=pd.read_excel(BytesIO(payload),sheet_name=None,header=None)
    output=[]
    for sheet,frame in workbook.items():
        variable=_sipri_variable(str(sheet))
        if variable is None:
            continue
        try:
            header_idx,year_columns=_wide_sheet_records(frame)
        except ValueError:
            continue
        country_col=min(year_columns)-1
        if country_col<0:
            continue
        for row_idx in range(header_idx+1,len(frame)):
            country=str(frame.iat[row_idx,country_col] or "").strip()
            if not country or country.lower() in {"nan","total","world"}:
                continue
            slug=re.sub(r"[^a-z0-9]+","-",country.lower()).strip("-")
            if not slug:
                continue
            for col,year in year_columns.items():
                value=_numeric(frame.iat[row_idx,col])
                if value is None:
                    continue
                output.append(TemporalRecord(
                    record_id=f"sipri:{slug}:{year}:{variable}:{dataset_version}",
                    entity_id=f"sipri_country:{slug}",
                    variable=variable,
                    value=value,
                    valid_from=datetime(year,1,1,tzinfo=timezone.utc),
                    valid_to=datetime(year+1,1,1,tzinfo=timezone.utc),
                    known_at=known_at,
                    evidence_class=EvidenceClass.ESTIMATE,
                    provenance=Provenance(
                        source="sipri",
                        source_ref=f"{sheet}:{country}:{year}",
                        retrieved_at=retrieved_at,
                        dataset_version=dataset_version,
                        transform="SIPRI wide workbook -> annual estimate",
                        notes="SIPRI revises historical values; exact workbook edition is preserved.",
                    ),
                    attributes={"country":country,"sheet":str(sheet)},
                ))
    return output
