import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class ScriptSmokeTest(unittest.TestCase):
    def test_build_proposal_sweep_writes_joblist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            joblist = Path(tmpdir) / "proposal_jobs.txt"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/build_proposal_sweep.py",
                    "--experiments",
                    "popgym_repeat_previous",
                    "--backbones",
                    "gru",
                    "transformer",
                    "--seeds",
                    "0",
                    "1",
                    "--write-joblist",
                    str(joblist),
                ],
                cwd=REPO_ROOT,
                check=True,
            )
            lines = [line for line in joblist.read_text().splitlines() if line.strip()]
            self.assertEqual(len(lines), 12)
            self.assertTrue(all("train.py" in line for line in lines))

    def test_analyze_results_generates_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "logdir"
            out = Path(tmpdir) / "analysis"
            self._write_fake_run(
                root / "popgym_repeat_previous" / "task1" / "gru" / "seed0",
                task="popgym_RepeatPreviousEasy-v0",
                backbone="gru",
                seed=0,
                scores=[0.1, 0.2, 0.3],
            )
            self._write_fake_run(
                root / "popgym_repeat_previous" / "task1" / "transformer" / "seed0",
                task="popgym_RepeatPreviousEasy-v0",
                backbone="transformer",
                seed=0,
                scores=[0.2, 0.4, 0.5],
            )
            subprocess.run(
                [
                    sys.executable,
                    "scripts/analyze_proposal_results.py",
                    "--logdir-root",
                    str(root),
                    "--output-dir",
                    str(out),
                    "--bootstrap-samples",
                    "10",
                ],
                cwd=REPO_ROOT,
                check=True,
            )
            self.assertTrue((out / "runs.csv").exists())
            self.assertTrue((out / "suite_summary.csv").exists())
            self.assertTrue((out / "performance_profiles.csv").exists())
            self.assertTrue((out / "summary.md").exists())

    def _write_fake_run(self, run_dir: Path, task: str, backbone: str, seed: int, scores):
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "resolved_config.json").open("w") as f:
            json.dump(
                {
                    "seed": seed,
                    "env": {"task": task},
                    "model": {"backbone": backbone, "rep_loss": "r2dreamer"},
                    "trainer": {"steps": 1000},
                },
                f,
            )
        with (run_dir / "run_metadata.json").open("w") as f:
            json.dump(
                {
                    "status": "completed",
                    "seed": seed,
                    "env_name": task,
                    "model_backbone": backbone,
                    "elapsed_seconds": 10.0,
                    "trainer_steps": 1000,
                },
                f,
            )
        with (run_dir / "metrics.jsonl").open("w") as f:
            for idx, score in enumerate(scores, start=1):
                f.write(json.dumps({"step": idx * 100, "episode/eval_score": score}) + "\n")


if __name__ == "__main__":
    unittest.main()
