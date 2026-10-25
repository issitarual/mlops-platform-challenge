"""Rules for the data directory. mlship does not read the data; train.py does."""

from pathlib import Path

from mlship.domain.validation import Severity, ValidationIssue

KNOWN_FORMATS = frozenset(
    {
        ".csv",
        ".tsv",
        ".txt",
        ".json",
        ".jsonl",
        ".parquet",
        ".feather",
        ".arrow",
        ".npy",
        ".npz",
        ".gz",
        ".zip",
    }
)
_EXAMPLES = 3


def check_data_dir(data_dir: Path) -> list[ValidationIssue]:
    if not data_dir.is_dir():
        return [
            ValidationIssue(
                "DATA_DIR_MISSING",
                Severity.ERROR,
                f"{data_dir} is not a directory.",
                str(data_dir),
                "Pass the folder that holds the training data with --data.",
            )
        ]
    files = sorted(
        p
        for p in data_dir.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(data_dir).parts)
    )
    if not files:
        return [
            ValidationIssue(
                "DATA_DIR_EMPTY",
                Severity.ERROR,
                f"{data_dir} has no files.",
                str(data_dir),
                "Download or copy the data first (see the example's README).",
            )
        ]
    unknown = [p for p in files if p.suffix.lower() not in KNOWN_FORMATS]
    if not unknown:
        return []
    examples = ", ".join(p.relative_to(data_dir).as_posix() for p in unknown[:_EXAMPLES])
    more = f" and {len(unknown) - _EXAMPLES} more" if len(unknown) > _EXAMPLES else ""
    return [
        ValidationIssue(
            "DATA_FORMAT_UNKNOWN",
            Severity.WARNING,
            f"{len(unknown)} file(s) with an unusual format: {examples}{more}.",
            str(data_dir),
            "Fine if train.py reads them. Otherwise keep only training data in the folder: "
            "every file is hashed and mounted into the container.",
        )
    ]
