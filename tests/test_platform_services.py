from datetime import datetime, timedelta, timezone

from preact.history.schema import EvidenceClass, Provenance, TemporalRecord
from preact.platform.services import ScenarioIntervention, ScenarioLabService


UTC = timezone.utc


def test_scenario_lab_labels_modified_values_as_counterfactual() -> None:
    baseline = TemporalRecord(
        record_id="gdp-1",
        entity_id="country:x",
        variable="gdp_growth",
        value=2.0,
        valid_from=datetime(2020, 1, 1, tzinfo=UTC),
        known_at=datetime(2020, 2, 1, tzinfo=UTC),
        provenance=Provenance(
            source="official",
            source_ref="x",
            retrieved_at=datetime(2020, 2, 1, tzinfo=UTC),
        ),
    )
    result = ScenarioLabService().apply(
        scenario_id="shock",
        baseline=[baseline],
        interventions=[
            ScenarioIntervention(
                variable="gdp_growth",
                operation="add",
                value=-3.0,
                description="explicit recession shock",
            )
        ],
        baseline_cutoff=datetime(2020, 2, 1, tzinfo=UTC),
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    modified = result.records[0]
    assert modified.value == -1.0
    assert modified.evidence_class is EvidenceClass.COUNTERFACTUAL
    assert modified.attributes["baseline_record_id"] == "gdp-1"
    assert modified.provenance.source == "scenario_lab"
