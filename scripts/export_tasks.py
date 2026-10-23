"""Write tasks (without answers) and answer keys as JSONL.

    python scripts/export_tasks.py --per-difficulty 50 --out data/
"""

import argparse
import json
from pathlib import Path

from gst_env import DIFFICULTIES, generate_task, solve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-difficulty", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with open(args.out / "tasks.jsonl", "w") as tasks, open(args.out / "answers.jsonl", "w") as answers:
        for difficulty in DIFFICULTIES:
            for seed in range(args.per_difficulty):
                task = generate_task(seed, difficulty)
                tasks.write(json.dumps(task.to_dict()) + "\n")
                answers.write(json.dumps({"task_id": task.task_id, **solve(task)}) + "\n")
    print(f"wrote {args.per_difficulty * len(DIFFICULTIES)} tasks to {args.out}/")


if __name__ == "__main__":
    main()
