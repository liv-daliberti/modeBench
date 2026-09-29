"""Render README level-result tables from retained evidence; --check detects drift."""

import argparse
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
BEGIN = "<!-- modebench-level-results:start -->"
END = "<!-- modebench-level-results:end -->"
DOMAINS = ["graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan"]
NAMES = dict(zip(DOMAINS, ["Graph", "Countdown", "Python", "MathIR", "PantryPlan"]))
METHODS = {
    "drgrpo": "Dr.GRPO",
    "replay_drgrpo": "Re:Dr",
    "maxrl": "MaxRL",
    "replay_maxrl": "Re:Max",
}


def read(path):
    return json.loads((ROOT / path).read_text())


def table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def render():
    construction = read("provenance/level-construction.json")
    training = read("evidence/level-training.json")
    grid = read("evidence/mode_diversity_base_grid.json")["cells"]
    # Keep the documentation bound to the published dataset and admission identity.
    manifest = read("data/manifest.json")
    splits = {
        (r["level"], r["domain"]): r
        for r in manifest["splits"]
        if r["split"] == "eval" and not r["config_name"].endswith("_unique_answer")
    }
    for record in construction["frozen_eval_structure"]:
        assert (
            record["source"]["sha256"]
            == splits[(record["level"], record["domain"])]["parquet_sha256"]
        )
    assert (
        construction["published_admission"]
        == read("provenance/levels45_manifest.json")["admission_status"]
    )
    lines = [
        BEGIN,
        "### Recorded tuning outcomes",
        "",
        "These are **construction/confirmation measurements**, not the later native evaluation grid. All rates below are fractions; `pass@1` is per-response correctness. Displayed values are rounded; linked JSON retains full precision.",
        "",
        "<details>",
        "<summary>Level-2 development admission and Levels 3–5 held-out matching</summary>",
        "",
        "Level 2: frozen models evaluated paired Level-1 construction references and Level-2 development tasks, with eight responses per prompt. Every listed Level-2 value is in [0.10, 0.90] and below its paired reference. The final source also records a Falcon3-1B check.",
        "",
    ]
    rows = []
    for domain in DOMAINS:
        records = [r for r in construction["level2"]["measurements"] if r["domain"] == domain]
        values = []
        for model in ("qwen-0.5b", "falcon-1b"):
            r = next(r for r in records if r["model"] == model)
            assert (
                0.1 <= r["level2_pass_at_8"] <= 0.9
                and r["level2_pass_at_8"] < r["level1_pass_at_8"]
            )
            values.append(f"{r['level1_pass_at_8']:.3f} → {r['level2_pass_at_8']:.3f}")
        rows.append([NAMES[domain], *values])
    lines += [
        table(["Domain", "Qwen2.5-0.5B L1 → L2 pass@8", "Falcon3-1B L1 → L2 pass@8"], rows),
        "",
        "Levels 3–5: each entry is **pass@1 / pass@8**. The reference is a fixed Qwen2.5-0.5B measurement on historical Level-1 confirmation tasks. Candidates use 3B, 7B, and 14B respectively, with 128 prompts × four groups × eight responses per domain. Both absolute differences must be within **0.04 / 0.08**.",
        "",
    ]
    rows = []
    for domain in DOMAINS:
        records = [
            r for r in construction["scale_calibration"]["measurements"] if r["domain"] == domain
        ]
        target = records[0]["target"]
        values = [f"{target['pass1']:.4f} / {target['pass8']:.4f}"]
        for level in (3, 4, 5):
            r = next(r for r in records if r["level"] == level)
            assert r["target"] == target
            matched = all(
                abs(r["metrics"][m] - target[m]) <= tolerance
                for m, tolerance in construction["scale_calibration"]["tolerances"].items()
            )
            assert matched == r["difficulty_matched"]
            values.append(
                f"{r['metrics']['pass1']:.4f} / {r['metrics']['pass8']:.4f}"
                + (" **†**" if not matched else "")
            )
        rows.append([NAMES[domain], *values])
    lines += [
        table(["Domain", "Fixed L1 reference", "L3, Qwen 3B", "L4, Qwen 7B", "L5, Qwen 14B"], rows),
        "",
        "**† Level-4 MathIR fails the pass@1 gate:** 0.0864258 − 0.0437012 = 0.0427246, exceeding 0.04. Its pass@8 gate passes. All other displayed candidate cells pass both gates. The public package retains Level 4 as not admitted overall; MathIR results are descriptive and must not be labeled difficulty-matched.",
        "",
        "The Level-3 release is adaptive round 2: Graph and Python have new confirmation measurements; Countdown, MathIR, and PantryPlan retain their earlier confirmation evidence. The Level-1 reference was not resampled. These tolerances establish empirical matching, not a statistical equivalence test. [Calibration values, source hashes, and information boundaries](provenance/level-construction.json) retain the details.",
        "",
        "</details>",
        "",
        "### Untrained-model results across levels",
        "",
        "The frozen grid contains **375 model/domain/level cells**. The table gives mean **pass@8** over all five domains for four Qwen2.5-Instruct scales measured at every level. Each cell uses 128 native held-out prompts and four independent groups of eight responses. Scores are averaged over groups within prompts, then prompts, then domains; this is not pass@32.",
        "",
    ]
    rows = []
    for level in range(1, 6):
        values = []
        for model in ("05b", "3b", "7b", "14b"):
            cells = [r for r in grid if r["level"] == f"level{level}" and r["model_label"] == model]
            assert len(cells) == 5
            values.append(f"{mean(r['pass8'] for r in cells):.3f}")
        rows.append([f"{level}" + (" †" if level == 4 else ""), *values])
    lines += [
        table(["Level", "Qwen 0.5B", "Qwen 3B", "Qwen 7B", "Qwen 14B"], rows),
        "",
        "† Includes the unmatched MathIR condition. This grid uses native chat, temperature 1, top-p 1, a 192-token limit, boxed-direct Graph prompts, and guided, syntax-constrained decoding for the other domains. It is a separate measurement from tuning and from training evaluation; the saved-response CLI does not generate those constrained samples.",
        "",
        "<details>",
        "<summary>Qwen2.5-7B by level and domain: correctness, distinct modes, and PCMD</summary>",
        "",
        "The same 7B model is shown across all 25 cells. PCMD pools the 32 responses **within each prompt** and then averages prompts with at least two verified responses. `Eligible` gives that count out of 128. A dash suppresses PCMD when fewer than 30 prompts qualify; it is not zero.",
        "",
    ]
    rows = []
    for level in range(1, 6):
        for domain in DOMAINS:
            r = next(
                r
                for r in grid
                if r["level"] == f"level{level}"
                and r["model_label"] == "7b"
                and r["domain"] in (domain, "pantry" if domain == "pantry_plan" else domain)
            )
            rows.append(
                [
                    level,
                    NAMES[domain],
                    f"{r['pass8']:.3f}",
                    f"{r['distinct8']:.3f}",
                    f"{r['pmd']:.3f}" if r["reportable"] else "—",
                    f"{r['defined_prompts']}/128",
                ]
            )
    lines += [
        table(["Level", "Domain", "pass@8", "distinct@8", "PCMD", "Eligible"], rows),
        "",
        "All measured models, domains, support counts, standard errors, and receipt identities are in [the frozen grid](evidence/mode_diversity_base_grid.json). Model coverage is 17 models at Levels 1–4 and seven Qwen scales at Level 5, so an all-model average would change its population. The table above uses the same four models and five domains throughout.",
        "",
        "</details>",
        "",
        "### Training experiments on Levels 1–3",
        "",
        "**Qwen2.5-0.5B-Instruct**, final recorded step **3072**, seeds **43–47**, four groups of eight responses on 128 held-out prompts per domain. Re:Dr adds verified-mode replay to Dr.GRPO; Re:Max adds it to MaxRL. Training implementations and broader cohorts live in [Re:Max](https://github.com/liv-daliberti/remax). These are recorded experiments, not new training runs.",
        "",
        "For a complete common correctness population, pass@8 weights **Graph, Countdown, Python, and PantryPlan** equally within each seed and then averages the five seeds. MathIR is excluded consistently because one Level-3 MaxRL terminal evaluation is conflicted. PCMD averages eligible seeds within each of **Graph, MathIR, and PantryPlan**, then weights those three domains equally. The seed counts are shown in that order; every contributing seed needs at least **30 eligible prompts**. Countdown and Python are excluded from this overview because some method/level combinations lack eligible seeds; all five domains appear below. These PCMD means are descriptive: their contributing seed populations differ, so differences are not paired treatment-effect estimates or confidence intervals.",
        "",
    ]
    summaries = {}
    for level in range(1, 4):
        for method in METHODS:
            for domain in DOMAINS:
                rs = [
                    r
                    for r in training["records"]
                    if (r["level"], r["method"], r["domain"]) == (level, method, domain)
                ]
                assert len(rs) == 5
                complete = [r for r in rs if r["admitted_terminal"]]
                for r in complete:
                    assert len(r["groups"]) == 4
                pmds = [
                    r["pcmd"]["pmd"]
                    for r in rs
                    if r["pcmd"] is not None and r["pcmd"]["reportable"]
                ]
                summaries[(level, method, domain)] = {
                    "pass8": mean(mean(g["pass8"] for g in r["groups"]) for r in complete)
                    if complete
                    else None,
                    "terminal_seeds": len(complete),
                    "pcmd": mean(pmds) if pmds else None,
                    "pcmd_seeds": len(pmds),
                }
    rows = []
    for level in range(1, 4):
        for method, name in METHODS.items():
            accuracy = [
                summaries[(level, method, dom)]
                for dom in training["summary_policy"]["pass8_domains"]
            ]
            assert all(r["terminal_seeds"] == 5 for r in accuracy)
            diversity = [
                summaries[(level, method, dom)]
                for dom in training["summary_policy"]["pcmd_domains"]
            ]
            rows.append(
                [
                    level,
                    name,
                    f"{mean(r['pass8'] for r in accuracy):.3f}",
                    f"{mean(r['pcmd'] for r in diversity):.3f}"
                    if all(r["pcmd"] is not None for r in diversity)
                    else "—",
                    " / ".join(str(r["pcmd_seeds"]) for r in diversity),
                ]
            )
    lines += [
        table(
            [
                "Level",
                "Method",
                "pass@8 (4 domains)",
                "PCMD (3 domains)",
                "PCMD seeds: Graph / MathIR / Pantry",
            ],
            rows,
        ),
        "",
        "<details>",
        "<summary>Training results for every domain and level</summary>",
        "",
        "Each entry is **pass@8 / PCMD [eligible PCMD seeds]**. Pass@8 uses all five terminal seeds unless marked `*`. PCMD uses only the eligible seeds shown; `— [0]` means insufficient support. These are absolute endpoint means, not paired replay effects.",
        "",
    ]
    rows = []
    for level in range(1, 4):
        for domain in DOMAINS:
            values = []
            for method in METHODS:
                r = summaries[(level, method, domain)]
                accuracy = f"{r['pass8']:.3f}" + ("*" if r["terminal_seeds"] != 5 else "")
                diversity = f"{r['pcmd']:.3f}" if r["pcmd"] is not None else "—"
                values.append(f"{accuracy} / {diversity} [{r['pcmd_seeds']}]")
            rows.append([level, NAMES[domain], *values])
    lines += [
        table(["Level", "Domain", *METHODS.values()], rows),
        "",
        "*Level-3 MaxRL MathIR correctness uses four seeds; seed 45 has conflicted terminal draws and is omitted. The retained Level-2 Re:Max PantryPlan diversity archive lacks seed 46, while its correctness endpoint is available; it is not imputed. [Per-seed values, draw metadata, gaps, and source hashes](evidence/level-training.json) support these summaries. Some later manuscript analyses use a 20-prompt PCMD threshold; these tables consistently use the retained 30-prompt rule and therefore can differ from those figures.",
        "",
        "</details>",
        "",
        "**Coverage limits:** the complete grid supplies untrained-model measurements on Levels 1–5; this training summary covers Levels 1–3 only. It contains no Level-4/5 Re:Max or Re:Dr training endpoints. Levels differ in task populations, mode-count distributions, and sometimes prompt guidance, so cross-level differences do not isolate a causal effect of difficulty. Comparisons of a trained policy to an untrained model need the same evaluation protocol.",
        "",
        "Regenerate these tables with `python ops/summarize_levels.py --write`; verify them with `python ops/summarize_levels.py --check`. This reproduces the tabulated summaries from retained evidence without accessing the research checkout. The extract does not bundle training checkpoints or raw generations, so it does not independently regrade or rerun those experiments.",
        END,
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--write", action="store_true")
    args = parser.parse_args()
    generated = render()
    path = ROOT / "README.md"
    text = path.read_text()
    if args.check or args.write:
        left, rest = text.split(BEGIN, 1)
        current, right = rest.split(END, 1)
        if args.check:
            if BEGIN + current + END != generated:
                raise SystemExit("README level results differ; review evidence then regenerate")
            print(
                "Verified level tables: 15 scale-calibration cells, 375 base cells, 300 training seed slots."
            )
        else:
            path.write_text(left + generated + right)
    else:
        print(generated)


if __name__ == "__main__":
    main()
