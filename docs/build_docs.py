import os
import shutil
import subprocess
from pathlib import Path


EXCLUDED_API_PATHS = [
    "dashboard",
    "edl",
    "ground_segment",
    "interfaces/gmat",
]


def run_command(cmd: list[str], cwd: Path) -> None:
    print(f"Running command in {cwd}: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def build() -> None:
    docs_dir = Path.cwd()
    src_dir = (docs_dir.parent / "src" / "opengnc").resolve()
    api_dir = docs_dir / "api"

    print("Detected tutorials directory in docs, skipping copy step...")
    print(f"Docs dir: {docs_dir}")
    print(f"Source dir: {src_dir}")
    print(f"Excluded API paths: {', '.join(EXCLUDED_API_PATHS)}")

    if api_dir.exists():
        print(f"Removing existing API dir: {api_dir}")
        shutil.rmtree(api_dir)

    apidoc_cmd = ["sphinx-apidoc", "-f", "-o", str(api_dir), str(src_dir)]
    apidoc_cmd.extend(str(src_dir / rel_path) for rel_path in EXCLUDED_API_PATHS)
    run_command(apidoc_cmd, docs_dir)

    sphinx_cmd = ["sphinx-build", "-W", "--keep-going", ".", "_build/html"]
    run_command(sphinx_cmd, docs_dir)
    print("Build completed successfully!")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    build()
