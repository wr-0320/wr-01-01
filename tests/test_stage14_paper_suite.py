from experiments.build_stage14_paper_suite import (
    ablation_table,
    failure_table,
    main_table,
    theoretical_appendix,
)


def test_main_table_uses_single_balanced_protocol_and_all_methods():
    rows = main_table()
    assert len(rows) == 18
    assert {row["episodes"] for row in rows} == {34}
    assert len({row["method"] for row in rows}) == 6
    assert {row["attack"] for row in rows} == {
        "Honest",
        "Scripted-3",
        "Adaptive attacker",
    }


def test_ablation_and_failure_tables_retain_provenance():
    ablations = ablation_table()
    failures = failure_table()
    assert {row["source"] for row in ablations} >= {
        "Stage7.3",
        "Stage9.2",
        "Stage10.1",
        "Stage10.2",
        "Stage13",
    }
    assert len(failures) >= 7
    assert all(row["source"] and row["evidence"] for row in failures)


def test_theory_draft_states_identifiability_and_scope_limits():
    draft = theoretical_appendix()
    assert "Proposition 1" in draft
    assert "Bayes error" in draft
    assert "does not establish minimax optimality" in draft
    assert "Forked Routes" in draft

