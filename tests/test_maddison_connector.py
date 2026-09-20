from datetime import datetime, timezone

from preact.data_hub.gateway import ProviderResponse
from preact.history.connectors.maddison import Maddison2023Connector


class FakeGateway:
    def get_json(self, **kwargs):
        return ProviderResponse(
            source_id="maddison",
            operation="metadata",
            payload={
                "data": {
                    "latestVersion": {
                        "versionNumber": 1,
                        "files": [
                            {
                                "dataFile": {
                                    "id": 421302,
                                    "filename": "mpd2023_web.xlsx",
                                }
                            }
                        ],
                    }
                }
            },
            retrieved_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            cached=False,
            request_fingerprint="x",
        )


class FakeBulk:
    source_id = "maddison"


def test_maddison_resolves_file_by_doi_metadata_not_hardcoded_file_id() -> None:
    connector = Maddison2023Connector(FakeGateway(), FakeBulk())
    resolved = connector.resolve_file()
    assert resolved.datafile_id == 421302
    assert resolved.label == "mpd2023_web.xlsx"
