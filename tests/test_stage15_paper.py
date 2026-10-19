import csv
import json

from experiments.build_stage15_paper import (
    REQUIRED_SECTIONS,
    STAGE14,
    ablation_results_table,
    audit_claims,
    build,
    build_claim_registry,
    main_results_table,
    read_csv,
    references_bib,
    section_texts,
)


def test_stage14_csv_conversion_produces_latex_tables():
    main = read_csv(STAGE14 / "main_table.csv")
    ablations = read_csv(STAGE14 / "ablation_table.csv")
    main_tex = main_results_table(main)
    ablation_tex = ablation_results_table(ablations)
    assert "Stage13 Learned" in main_tex
    assert "100.0 & 0.0 & 0.201" in main_tex
    assert r"\begin{table*}" in main_tex
    assert "Input identifiability" in ablation_tex
    assert r"stage10\_1" in ablation_tex


def test_claim_audit_covers_manuscript_and_blocks_overclaiming():
    main = read_csv(STAGE14 / "main_table.csv")
    sections = section_texts(main)
    registry = build_claim_registry(STAGE14)
    audit = audit_claims(registry, sections, references_bib())
    assert audit["passed"]
    assert audit["manuscript_claims"] == audit["supported_manuscript_claims"]
    assert audit["rejected_candidate_claims"] == 3
    assert {row["claim_id"] for row in registry if row["status"] == "rejected"} == {
        "X01",
        "X02",
        "X03",
    }


def test_claim_audit_fails_if_rejected_claim_enters_manuscript():
    main = read_csv(STAGE14 / "main_table.csv")
    sections = section_texts(main)
    sections["introduction"] += r"\claim{X01}{Universal robustness.}"
    audit = audit_claims(build_claim_registry(STAGE14), sections, references_bib())
    assert not audit["passed"]
    assert audit["rejected_claims_in_manuscript"] == ["X01"]


def test_stage15_build_writes_complete_skeleton_and_manifest(tmp_path):
    destination = tmp_path / "paper"
    manifest = build(destination, STAGE14, compile_pdf=False)
    assert set(manifest["required_sections"]) == set(REQUIRED_SECTIONS)
    assert manifest["claim_audit"]["passed"]
    assert manifest["algorithms_modified"] is False
    for section in REQUIRED_SECTIONS:
        assert (destination / "sections" / f"{section}.tex").is_file()
    for table in ("main_results", "ablation_results", "failure_cases"):
        assert (destination / "tables" / f"{table}.tex").is_file()
    assert len(list((destination / "figures" / "main").glob("*.pdf"))) == 3
    assert len(list((destination / "figures" / "appendix").glob("*.pdf"))) == 4
    with (destination / "claim_registry.csv").open(newline="") as handle:
        claims = list(csv.DictReader(handle))
    assert len(claims) == manifest["claim_audit"]["registered_claims"]
    assert json.loads((destination / "claim_audit.json").read_text())["passed"]

