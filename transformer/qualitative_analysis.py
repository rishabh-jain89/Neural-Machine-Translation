import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path


SEED = 19

ERROR_CATEGORIES = (
    "mistranslation;omission;repetition;named_entity_or_number;"
    "negation;hallucination;morphology_or_agreement;word_order;"
    "untranslated_word;premature_eos"
)


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def resolve_project_root(value):
    if value:
        return Path(value).resolve()
    return Path(__file__).resolve().parent.parent


def length_bucket(n):
    n = int(n)
    if n <= 20:
        return "short_le_20"
    if n <= 50:
        return "medium_21_50"
    return "long_gt_50"


def proportional_targets(n):
    short = round(n * 0.30)
    medium = round(n * 0.50)
    long_ = n - short - medium
    return {
        "short_le_20": short,
        "medium_21_50": medium,
        "long_gt_50": long_,
    }


def sample_mode(args):
    root = resolve_project_root(args.project_root)

    baseline_path = root / "output/evaluation/test_predictions.csv"
    refined_path = (root / "output" / "evaluation" / "refined_transformer" / "test_predictions.csv")    

    baseline = read_csv(baseline_path)
    refined = read_csv(refined_path)

    baseline_by_id = {int(r["id"]): r for r in baseline}
    refined_by_id = {int(r["id"]): r for r in refined}

    common_ids = sorted(
        set(baseline_by_id) & set(refined_by_id)
    )

    if len(common_ids) != 2507:
        print(
            f"Warning: found {len(common_ids)} common examples "
            f"instead of 2507.",
            flush=True,
        )

    merged = []

    for idx in common_ids:
        b = baseline_by_id[idx]
        e = refined_by_id[idx]

        if b["source_en"].strip() != e["source_en"].strip():
            raise RuntimeError(f"Source mismatch at id={idx}")

        if b["reference_hi"].strip() != e["reference_hi"].strip():
            raise RuntimeError(f"Reference mismatch at id={idx}")

        src_tokens = int(b["src_sp_tokens"])

        merged.append({
            "id": idx,
            "source_en": b["source_en"],
            "reference_hi": b["reference_hi"],
            "src_sp_tokens": src_tokens,
            "length_bucket": length_bucket(src_tokens),
            "bilstm_greedy": b.get("bilstm_greedy", ""),
            "baseline_transformer_greedy": b.get(
                "transformer_greedy", ""
            ),
            "refined_transformer_greedy": e.get(
                "greedy_hi", ""
            ),
            "refined_transformer_beam2": e.get(
                "beam_hi", ""
            ),
        })

    buckets = {
        "short_le_20": [],
        "medium_21_50": [],
        "long_gt_50": [],
    }

    for row in merged:
        buckets[row["length_bucket"]].append(row)

    rng = random.Random(args.seed)
    targets = proportional_targets(args.n)

    selected = []

    for bucket_name, count in targets.items():
        candidates = buckets[bucket_name]

        if len(candidates) < count:
            raise RuntimeError(
                f"Not enough examples in {bucket_name}: "
                f"need {count}, have {len(candidates)}"
            )

        selected.extend(
            rng.sample(candidates, count)
        )

    rng.shuffle(selected)

    output_path = (
        Path(args.output).resolve()
        if args.output
        else (
            root
            / "output/evaluation/"
              f"qualitative_sample_{args.n}.csv"
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "sample_no",
        "id",
        "length_bucket",
        "src_sp_tokens",
        "source_en",
        "reference_hi",
        "bilstm_greedy",
        "standard_transformer_greedy",
        "enhanced_transformer_greedy",
        "enhanced_transformer_beam2",

        "bilstm_adequacy_1_5",
        "bilstm_fluency_1_5",
        "bilstm_error_types",

        "standard_adequacy_1_5",
        "standard_fluency_1_5",
        "standard_error_types",

        "enhanced_adequacy_1_5",
        "enhanced_fluency_1_5",
        "enhanced_error_types",

        "preferred_model",
        "comments",
    ]

    with open(
        output_path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for sample_no, row in enumerate(selected, start=1):
            out = {key: "" for key in fieldnames}
            out.update(row)
            out["sample_no"] = sample_no
            writer.writerow(out)

    print(f"Created: {output_path}", flush=True)
    print("Sampling:", targets, flush=True)


def parse_float(value):
    value = (value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def summarize_mode(args):
    if not args.input:
        raise SystemExit(
            "--input is required for --mode summarize"
        )

    rows = read_csv(Path(args.input))

    systems = {
        "bilstm": (
            "bilstm_adequacy_1_5",
            "bilstm_fluency_1_5",
            "bilstm_error_types",
        ),
        "standard_transformer": (
            "standard_adequacy_1_5",
            "standard_fluency_1_5",
            "standard_error_types",
        ),
        "enhanced_transformer": (
            "enhanced_adequacy_1_5",
            "enhanced_fluency_1_5",
            "enhanced_error_types",
        ),
    }

    summary = {
        "examples": len(rows),
        "systems": {},
        "preferred_model_counts": dict(
            Counter(
                r.get("preferred_model", "").strip()
                for r in rows
                if r.get("preferred_model", "").strip()
            )
        ),
    }

    for system, (adequacy_col, fluency_col, errors_col) in systems.items():
        adequacy = [
            x
            for x in (
                parse_float(r.get(adequacy_col))
                for r in rows
            )
            if x is not None
        ]

        fluency = [
            x
            for x in (
                parse_float(r.get(fluency_col))
                for r in rows
            )
            if x is not None
        ]

        errors = Counter()

        for row in rows:
            raw = (row.get(errors_col) or "").strip()
            if not raw:
                continue

            for error in raw.split(";"):
                error = error.strip()
                if error:
                    errors[error] += 1

        summary["systems"][system] = {
            "annotated_adequacy_n": len(adequacy),
            "mean_adequacy": (
                sum(adequacy) / len(adequacy)
                if adequacy else None
            ),
            "annotated_fluency_n": len(fluency),
            "mean_fluency": (
                sum(fluency) / len(fluency)
                if fluency else None
            ),
            "error_counts": dict(errors.most_common()),
        }

    output_path = (
        Path(args.summary_output).resolve()
        if args.summary_output
        else Path(args.input).with_suffix(".summary.json")
    )

    output_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved: {output_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["sample", "summarize"],
        required=True,
    )

    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--n",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--summary-output",
        type=str,
        default=None,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "sample":
        sample_mode(args)
    else:
        summarize_mode(args)
