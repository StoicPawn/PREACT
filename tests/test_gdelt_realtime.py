import io
import zipfile

from preact.data_hub.gdelt_realtime import parse_event_zip, parse_lastupdate


def test_parse_lastupdate_classifies_core_files() -> None:
    text = "\n".join(
        [
            "123 abc https://data.gdeltproject.org/gdeltv2/20200101000000.export.CSV.zip",
            "456 def https://data.gdeltproject.org/gdeltv2/20200101000000.mentions.CSV.zip",
            "789 ghi https://data.gdeltproject.org/gdeltv2/20200101000000.gkg.csv.zip",
        ]
    )
    refs = parse_lastupdate(text)
    assert [ref.kind for ref in refs] == ["events", "mentions", "gkg"]


def test_parse_event_zip_names_minimal_fields() -> None:
    values = [""] * 61
    values[0] = "123"
    values[1] = "20200101"
    values[6] = "Actor A"
    values[16] = "Actor B"
    values[26] = "190"
    values[60] = "https://example.test/story"

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("events.csv", "\t".join(values) + "\n")

    rows = parse_event_zip(buffer.getvalue())
    assert rows[0]["GLOBALEVENTID"] == "123"
    assert rows[0]["Actor1Name"] == "Actor A"
    assert rows[0]["SOURCEURL"] == "https://example.test/story"
