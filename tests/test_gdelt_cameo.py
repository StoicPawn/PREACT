from preact.history.connectors.gdelt_cameo import build_cameo_country_map


def test_cameo_country_map_normalizes_legacy_codes_and_rejects_regions():
    payload = (
        b"CODE\tLABEL\n"
        b"DEU\tGermany\n"
        b"ROM\tRomania\n"
        b"TMP\tEast Timor\n"
        b"MTN\tMontenegro\n"
        b"EUR\tEurope\n"
    )

    mapping = build_cameo_country_map(payload)

    assert mapping.resolve("DEU") == "DEU"
    assert mapping.resolve("rom") == "ROU"
    assert mapping.resolve("TMP") == "TLS"
    assert mapping.resolve("MTN") == "MNE"
    assert mapping.resolve("EUR") is None
    assert mapping.resolve("UNKNOWN") is None
