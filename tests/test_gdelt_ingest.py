import io
import zipfile

from preact.intelligence.gdelt_ingest import (
    _EVENT_COLUMNS,
    _GKG_COLUMNS,
    _MENTION_COLUMNS,
    normalize_gkg_documents,
    parse_event_zip,
    parse_gkg_zip,
    parse_mentions_zip,
    summarize_mentions,
)


def _zip_row(values):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("data.csv", "\t".join(values) + "\n")
    return buffer.getvalue()


def test_parse_event_zip_names_core_fields():
    values = [""] * len(_EVENT_COLUMNS)
    values[0] = "123"
    values[7] = "ITA"
    values[17] = "FRA"
    values[26] = "040"
    values[60] = "https://example.test/story"
    row = parse_event_zip(_zip_row(values))[0]
    assert row["GLOBALEVENTID"] == "123"
    assert row["Actor1CountryCode"] == "ITA"
    assert row["Actor2CountryCode"] == "FRA"
def test_mentions_are_aggregated_as_evidence_not_facts():
    a = [""] * len(_MENTION_COLUMNS)
    a[0] = "123"
    a[2] = "20260927111500"
    a[4] = "wire-a.example"
    a[11] = "80"
    a[13] = "-1.5"

    b = a.copy()
    b[4] = "wire-b.example"
    b[11] = "90"
    b[13] = "0.5"

    rows = parse_mentions_zip(_zip_row(a) + b"")
    rows.extend(parse_mentions_zip(_zip_row(b)))
    summary = summarize_mentions(rows)[0]
    assert summary["provider_event_id"] == "123"
    assert summary["mention_count"] == 2
    assert summary["distinct_source_count"] == 2
    assert summary["max_confidence"] == 90.0
    assert summary["mean_document_tone"] == -0.5


def test_gkg_normalizes_country_theme_and_entities():
    values = [""] * len(_GKG_COLUMNS)
    values[0] = "20260927111500-0"
    values[1] = "20260927111500"
    values[3] = "example.com"
    values[4] = "https://example.com/article"
    values[7] = "DIPLOMACY;SANCTIONS;"
    values[10] = "1#Italy#IT#IT##42.8#12.8#IT#10;1#France#FR#FR##46#2#FR#20"
    values[11] = "Person A;Person B"
    values[13] = "European Union;Government"
    values[15] = "-2.4,1,2,3,4,5,100"
    values[23] = "Person A,10;European Union,20"

    rows = parse_gkg_zip(_zip_row(values))
    docs = normalize_gkg_documents(rows)
    assert len(docs) == 1
    item = docs[0]
    assert item["country_codes"] == ["FR", "IT"]
    assert "DIPLOMACY" in item["themes"]
    assert item["persons"] == ["Person A", "Person B"]
    assert item["organizations"] == ["European Union", "Government"]
    assert item["overall_tone"] == -2.4
