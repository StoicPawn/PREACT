from preact.history.connectors.geonames import (
    CountryCodeMap,
    GeoNamesCountryInfoConnector,
)


def test_country_info_maps_fips_to_iso() -> None:
    payload = (
        "#ISO\tISO3\tISO-Numeric\tfips\tCountry\n"
        "IT\tITA\t380\tIT\tItaly\tRome\t301230\t60000000\tEU\t.it\tEUR\tEuro"
        "\t39\t#####\t\tIT\t3175395\tCH,AT,SI,SM,FR,VA\t\n"
        "SY\tSYR\t760\tSY\tSyria\tDamascus\t185180\t20000000\tAS\t.sy\tSYP"
        "\tPound\t963\t\t\tar-SY\t163843\tIQ,JO,IL,TR,LB\t\n"
    ).encode("utf-8")
    rows = GeoNamesCountryInfoConnector.parse(payload)
    mapping = CountryCodeMap(rows)
    assert mapping.by_fips["IT"].iso3 == "ITA"
    assert mapping.fips_to_iso3()["SY"] == "SYR"
