"""Repeat the Python and Markdown checks for the cleaned stag-hunt modules."""
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COOPERATION = Path("predpreygrass/non_evolutionary/discrete_space/project_cooperation")
FORWARD = COOPERATION / "stag_hunt_forward_view"


def main():
    commands = [
        ["pyright", "--project", "pyrightconfig.json"],
        ["ruff", "check", str(FORWARD), str(COOPERATION / "stag_hunt_defection"), *[
            str(p.relative_to(ROOT))
            for p in (ROOT / COOPERATION / "stag_hunt_reputation").glob("evaluate_ppo_from_checkpoint*.py")
        ]],
        ["markdownlint-cli2", str(FORWARD / "README.md"), str(COOPERATION / "stag_hunt_defection/README.md"),
         str(COOPERATION / "stag_hunt_defection/COMPARISON_STAG_HUNT.md")],
    ]
    failed = False
    for command in commands:
        executable = shutil.which(command[0])
        if executable is None:
            candidates = [ROOT / "node_modules/.bin" / command[0], Path(sys.executable).parent / command[0]]
            executable = next((str(p) for p in candidates if p.exists()), None)
        if executable is None:
            print(f"Missing {command[0]}; run npm ci and install the Python dev dependencies.", flush=True)
            failed = True
            continue
        print("Running " + " ".join(command), flush=True)
        failed |= subprocess.run([executable, *command[1:]], cwd=ROOT).returncode != 0

    # Check Markdown and HTML local links, using tracked docs only. The active
    # continuous module is deliberately excluded from this maintenance scope.
    files = subprocess.check_output(["git", "ls-files", "*.md"], cwd=ROOT, text=True).splitlines()
    for filename in files:
        if "base_environment_continuous" in Path(filename).parts:
            continue
        path = ROOT / filename
        for number, line in enumerate(path.read_text().splitlines(), 1):
            targets = re.findall(r'\]\(([^\s)]+)\)', line)
            targets += re.findall(r'''(?:src|href)=["']([^"']+)["']''', line)
            for target in targets:
                target = target.split("#", 1)[0]
                if target and ":" not in target and not target.startswith("/") and not (path.parent / target).exists():
                    print(f"{filename}:{number}: missing local target {target}")
                    failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
