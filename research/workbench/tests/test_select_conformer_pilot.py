from pathlib import Path

import torch

from ghxtox.select_conformer_pilot import select_pilot


def test_select_pilot_preserves_source_ids_and_strata(tmp_path: Path) -> None:
    fasta = tmp_path / "train.fasta"
    fasta.write_text(
        ">pep|1\nACCA\n>pep|0\nAAAA\n>pep|1\nCCCC\n>pep|0\nGGGG\n",
        encoding="utf-8",
    )
    folds = tmp_path / "folds.csv"
    folds.write_text(
        "source_index,sample_id,label,sequence,group_id,fold\n"
        "0,pep_1,1,ACCA,g0,0\n1,pep_2,0,AAAA,g1,0\n"
        "2,pep_3,1,CCCC,g2,1\n3,pep_4,0,GGGG,g3,1\n",
        encoding="utf-8",
    )
    rows = select_pilot(fasta, folds, tmp_path / "pilot.fasta", tmp_path / "pilot.csv", target_size=4)
    assert {row["sample_id"] for row in rows} == {"pep_1", "pep_2", "pep_3", "pep_4"}
    text = (tmp_path / "pilot.fasta").read_text(encoding="utf-8")
    assert ">pep_1|1" in text


def test_select_pilot_excludes_already_generated_source_indices(tmp_path: Path) -> None:
    fasta = tmp_path / "train.fasta"
    fasta.write_text(">a|1\nAAAA\n>b|0\nBBBB\n>c|1\nCCCC\n", encoding="utf-8")
    folds = tmp_path / "folds.csv"
    folds.write_text(
        "source_index,sample_id,label,sequence,group_id,fold\n"
        "0,pep_1,1,AAAA,g0,0\n1,pep_2,0,BBBB,g1,1\n2,pep_3,1,CCCC,g2,2\n",
        encoding="utf-8",
    )
    excluded = tmp_path / "generated.csv"
    excluded.write_text("source_index,sample_id\n1,pep_2\n", encoding="utf-8")
    rows = select_pilot(
        fasta,
        folds,
        tmp_path / "remaining.fasta",
        tmp_path / "remaining.csv",
        target_size=10,
        max_length=15,
        exclude_manifest=excluded,
    )
    assert [row["source_index"] for row in rows] == [0, 2]


def test_balanced_length_bins_are_exact_across_fold_and_label(tmp_path: Path) -> None:
    entries = [
        (0, 1, "ACCA"),
        (0, 0, "AAAA"),
        (1, 1, "CCCC"),
        (1, 0, "GGGG"),
        (0, 1, "ACCAC"),
        (0, 0, "AAAAA"),
        (1, 1, "CCCCC"),
        (1, 0, "GGGGG"),
    ]
    fasta = tmp_path / "train.fasta"
    fasta.write_text(
        "".join(f">pep|{label}\n{sequence}\n" for _, label, sequence in entries),
        encoding="utf-8",
    )
    folds = tmp_path / "folds.csv"
    folds.write_text(
        "source_index,sample_id,label,sequence,group_id,fold\n"
        + "".join(
            f"{index},pep_{index},{label},{sequence},g{index},{fold}\n"
            for index, (fold, label, sequence) in enumerate(entries)
        ),
        encoding="utf-8",
    )
    processed = tmp_path / "processed.pt"
    torch.save(
        {
            "records": [
                {
                    "sequence": sequence,
                    "plddt": torch.full((len(sequence),), 0.6 if index % 2 else 0.9),
                }
                for index, (_, _, sequence) in enumerate(entries)
            ]
        },
        processed,
    )
    rows = select_pilot(
        fasta,
        folds,
        tmp_path / "pilot.fasta",
        tmp_path / "pilot.csv",
        target_size=8,
        min_length=4,
        max_length=5,
        processed=processed,
        length_bins=[(4, 4), (5, 5)],
    )
    assert len(rows) == 8
    assert {
        (row["length_bin"], row["fold"], row["label"])
        for row in rows
    } == {
        (length_bin, fold, label)
        for length_bin in ("4-4", "5-5")
        for fold in (0, 1)
        for label in (0, 1)
    }
