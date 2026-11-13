"""Check locked accelerator sources and conflicts without downloading GPU wheels."""

import argparse
import hashlib
import itertools
import json
import subprocess
import tempfile
import tomllib
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
PROFILES = ("cpu", "cu126", "cu128")


def uv(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", *args], cwd=ROOT, capture_output=True, text=True, check=check
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="Write the JSON evidence to this path."
    )
    args = parser.parse_args()
    uv("lock", "--check", "--offline")
    profiles = {}
    versions = {}

    with tempfile.TemporaryDirectory(prefix="llm-profiles-") as directory:
        for profile in PROFILES:
            path = Path(directory) / f"pylock.{profile}.toml"
            uv(
                "export",
                "--locked",
                "--offline",
                "--extra",
                profile,
                "--no-dev",
                "--no-emit-project",
                "--no-emit-workspace",
                "--format",
                "pylock.toml",
                "--output-file",
                str(path),
            )
            packages = tomllib.loads(path.read_text())["packages"]
            versions[profile] = {
                package["name"]: package["version"] for package in packages
            }
            torch = next(package for package in packages if package["name"] == "torch")
            expected_index = f"https://download.pytorch.org/whl/{profile}"

            if (
                torch["version"] != f"2.8.0+{profile}"
                or torch["index"] != expected_index
            ):
                raise RuntimeError(
                    f"Incorrect Torch version/source for {profile}: {torch}"
                )

            wheels = torch["wheels"]

            for wheel in wheels:
                url = urlsplit(wheel["url"])

                if (
                    url.scheme != "https"
                    or url.hostname
                    not in {"download.pytorch.org", "download-r2.pytorch.org"}
                    or not url.path.startswith(f"/whl/{profile}/")
                ):
                    raise RuntimeError(f"Unexpected Torch wheel source for {profile}.")

            if not any("cp312-cp312" in wheel["url"] for wheel in wheels):
                raise RuntimeError(f"No CPython 3.12 Torch wheel for {profile}.")

            if not all(wheel["hashes"].get("sha256") for wheel in wheels):
                raise RuntimeError(f"Missing Torch wheel hash for {profile}.")

            cuda = {
                package["name"]: package["version"]
                for package in packages
                if package["name"].startswith("nvidia-") or package["name"] == "triton"
            }

            if (profile == "cpu" and cuda) or (profile != "cpu" and not cuda):
                raise RuntimeError(
                    f"Unexpected CUDA dependency selection for {profile}: {cuda}"
                )

            dry_run = uv(
                "sync",
                "--locked",
                "--offline",
                "--dry-run",
                "--extra",
                profile,
                "--no-dev",
            )
            profiles[profile] = {
                "torch": torch["version"],
                "index": torch["index"],
                "package_count": len(packages),
                "torch_wheel_hashes_present": True,
                "cuda_dependencies": cuda,
                "sync_dry_run_exit_code": dry_run.returncode,
            }

    common = set.intersection(*(set(packages) for packages in versions.values())) - {
        "torch"
    }

    for package in common:
        if len({packages[package] for packages in versions.values()}) != 1:
            raise RuntimeError(f"Shared dependency {package} differs between profiles.")

    conflicts = []

    for first, second in itertools.combinations(PROFILES, 2):
        result = uv(
            "sync",
            "--locked",
            "--offline",
            "--dry-run",
            "--extra",
            first,
            "--extra",
            second,
            check=False,
        )

        if result.returncode == 0 or "incompatible" not in result.stderr.lower():
            raise RuntimeError(
                f"Profiles {first}/{second} did not fail as an incompatible pair."
            )

        conflicts.append({"profiles": [first, second], "exit_code": result.returncode})

    all_extras = uv(
        "sync", "--locked", "--offline", "--dry-run", "--all-extras", check=False
    )

    if all_extras.returncode == 0 or "incompatible" not in all_extras.stderr.lower():
        raise RuntimeError("--all-extras did not fail as incompatible.")

    report = {
        "status": "passed",
        "uv": uv("--version").stdout.strip(),
        "lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        "profiles": profiles,
        "shared_dependency_versions": {
            name: versions["cpu"][name] for name in sorted(common)
        },
        "rejected_conflicts": conflicts,
        "all_extras_exit_code": all_extras.returncode,
        "cuda_runtime_tested": False,
    }
    output = json.dumps(report, indent=2) + "\n"

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)

    print(output, end="")


if __name__ == "__main__":
    main()
