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


def build() -> None:
    docs_dir = Path.cwd()
    src_dir = (docs_dir.parent / "src" / "opengnc").resolve()
    api_dir = docs_dir / "api"

    print("Detected tutorials directory in docs, skipping copy step...")

    if api_dir.exists():
        shutil.rmtree(api_dir)

    print("Running sphinx-apidoc...")
    cmd = ["sphinx-apidoc", "-f", "-o", str(api_dir), str(src_dir)]
    cmd.extend(str(src_dir / rel_path) for rel_path in EXCLUDED_API_PATHS)
    subprocess.run(cmd, check=True)

    print("Running sphinx-build...")
    subprocess.run(["sphinx-build", "-W", "--keep-going", ".", "_build/html"], check=True)
    print("Build completed successfully!")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    build()
