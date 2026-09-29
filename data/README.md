# Frozen ModeBench datasets

Browse first by level, then by domain. Each domain directory contains its available `train.parquet`, `dev.parquet`, and `eval.parquet` splits.

| Level | Countdown | Graph coloring | MathIR | PantryPlan | Python factors |
| --- | --- | --- | --- | --- | --- |
| [Level 1](level1/) | [Countdown](level1/countdown/) | [Graph coloring](level1/graph_coloring/) | [MathIR](level1/mathir/) | [PantryPlan](level1/pantry_plan/) | [Python factors](level1/python_factors/) |
| [Level 2](level2/) | [Countdown](level2/countdown/) | [Graph coloring](level2/graph_coloring/) | [MathIR](level2/mathir/) | [PantryPlan](level2/pantry_plan/) | [Python factors](level2/python_factors/) |
| [Level 3](level3/) | [Countdown](level3/countdown/) | [Graph coloring](level3/graph_coloring/) | [MathIR](level3/mathir/) | [PantryPlan](level3/pantry_plan/) | [Python factors](level3/python_factors/) |
| [Level 4](level4/) | [Countdown](level4/countdown/) | [Graph coloring](level4/graph_coloring/) | [MathIR](level4/mathir/) | [PantryPlan](level4/pantry_plan/) | [Python factors](level4/python_factors/) |
| [Level 5](level5/) | [Countdown](level5/countdown/) | [Graph coloring](level5/graph_coloring/) | [MathIR](level5/mathir/) | [PantryPlan](level5/pantry_plan/) | [Python factors](level5/python_factors/) |

The [Level 1 graph-coloring single-answer diagnostic](level1/graph_coloring/unique_answer/) is separate from the main benchmark.

Configuration names (for example, `level1_countdown`) are unchanged. [manifest.json](manifest.json) maps each configuration and split to its file, row count, and frozen SHA-256. All 72 splits retain their original bytes and 15,552 rows.

See the [dataset guide](../docs/datasets.md) for loading, split availability, level-admission qualifications, and source terms.
