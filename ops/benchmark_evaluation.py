"""Measure CLI throughput and sampled Linux process-tree RSS; no model calls."""

import argparse
import json
import platform
import subprocess
import sys
import tempfile
import time
from importlib.resources import files
from pathlib import Path


def peak_rss(process):
    pending, total, parent = [process.pid], 0, 0
    for pid in pending:
        try:
            status = Path(f"/proc/{pid}/status").read_text()
            rss = next(
                int(line.split()[1]) * 1024
                for line in status.splitlines()
                if line.startswith("VmRSS:")
            )
            total += rss
            if pid == process.pid:
                parent = rss
            pending.extend(map(int, Path(f"/proc/{pid}/task/{pid}/children").read_text().split()))
        except (OSError, StopIteration):
            continue
    return parent, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000])
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or any(n < 1 for n in args.sizes):
        parser.error("use positive sizes and a fresh output path")
    demo = json.loads(files("modebench").joinpath("walkthrough.json").read_text())
    results = []
    with tempfile.TemporaryDirectory(prefix="modebench-throughput-") as temporary:
        work = Path(temporary)
        (work / "run.json").write_text(json.dumps(demo["run"]))
        for count in args.sizes:
            source = work / f"input-{count}.jsonl"
            with source.open("w") as handle:
                for i in range(count):
                    handle.write(
                        json.dumps({**demo["records"][i % 5], "id": f"benchmark-{i}"}) + "\n"
                    )
            for workers in args.workers:
                destination = work / f"result-{count}-{workers}.json"
                command = [
                    sys.executable,
                    "-m",
                    "modebench.cli",
                    "evaluate",
                    str(source),
                    "--run",
                    str(work / "run.json"),
                    "--output",
                    str(destination),
                    "--workers",
                    str(workers),
                    "--progress-every",
                    "0",
                ]
                start, parent_peak, tree_peak = time.perf_counter(), 0, 0
                with (
                    (work / "stdout.log").open("w") as stdout,
                    (work / "stderr.log").open("w") as stderr,
                ):
                    process = subprocess.Popen(command, stdout=stdout, stderr=stderr, cwd=work)
                    try:
                        while process.poll() is None:
                            parent, tree = peak_rss(process)
                            parent_peak, tree_peak = max(parent_peak, parent), max(tree_peak, tree)
                            time.sleep(0.025)
                    finally:
                        if process.poll() is None:
                            process.terminate()
                            process.wait(timeout=10)
                if process.returncode:
                    raise RuntimeError((work / "stderr.log").read_text())
                elapsed = time.perf_counter() - start
                report = json.loads(destination.read_text())
                directory = destination.with_name(destination.name + ".work")
                result = {
                    "prompts": count,
                    "responses": count * 4,
                    "workers": workers,
                    "seconds": elapsed,
                    "responses_per_second": count * 4 / elapsed,
                    "peak_parent_rss_bytes": parent_peak,
                    "peak_process_tree_rss_bytes": tree_peak,
                    "input_bytes": source.stat().st_size,
                    "receipt_bytes": destination.stat().st_size,
                    "journal_bytes": (directory / "results.jsonl").stat().st_size,
                    "checkpoint_bytes": sum(
                        p.stat().st_size for p in directory.iterdir() if p.is_file()
                    ),
                    "package_source_sha256": report["software"]["package_source_sha256"],
                }
                results.append(result)
                print(json.dumps(result), flush=True)
    payload = {
        "schema": "modebench-throughput-v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "workload": "Repeated frozen Level 1 walkthrough, five domains, four saved responses per prompt.",
        "method": "25 ms /proc RSS sampling; sum of live evaluator and descendant RSS, including shared pages counted per process; local /tmp SQLite checkpoint; warm OS cache; no model calls; single measurement per condition.",
        "limitations": "RSS sampling can miss short peaks; shared-page double counting; repeated tasks favor parser caches; results are machine/workload specific, not a performance guarantee.",
        "runs": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
