from datetime import datetime,timezone
from io import BytesIO

import pandas as pd

from preact.projections.historical_workbooks import maddison_records,sipri_milex_records

UTC=timezone.utc
NOW=datetime(2026,1,1,tzinfo=UTC)

def test_maddison_full_data_projection():
    buf=BytesIO()
    with pd.ExcelWriter(buf,engine="openpyxl") as writer:
        pd.DataFrame({
            "countrycode":["ITA"],"country":["Italy"],"year":[1950],
            "gdppc":[5000.0],"pop":[47000.0],
        }).to_excel(writer,sheet_name="Full data",index=False)
    rows=maddison_records(buf.getvalue(),known_at=NOW,retrieved_at=NOW)
    assert {r.variable for r in rows}=={"maddison:gdppc","maddison:population_thousands"}
    assert all(r.known_at==NOW for r in rows)

def test_sipri_wide_projection_skips_missing_markers():
    raw=[
        ["Military expenditure",None,None,None,None,None,None],
        ["Country",2020,2021,2022,2023,2024,2025],
        ["Italy","30","31",". .","33","34","35"],
    ]
    buf=BytesIO()
    with pd.ExcelWriter(buf,engine="openpyxl") as writer:
        pd.DataFrame(raw).to_excel(writer,sheet_name="Current US$",index=False,header=False)
    rows=sipri_milex_records(
        buf.getvalue(),known_at=NOW,retrieved_at=NOW,dataset_version="test"
    )
    assert len(rows)==5
    assert all(r.variable=="sipri:milex_current_usd_m" for r in rows)
