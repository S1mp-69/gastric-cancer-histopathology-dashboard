"""Package research files only; exclude Git metadata and Python bytecode."""
import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new archive name")
    root = Path(__file__).resolve().parent
    if args.output.resolve().is_relative_to(root):
        parser.error("Place the archive outside the source folder")
    files = [path for path in root.rglob("*") if path.is_file()
             and not any(part in {".git", "__pycache__"} for part in path.relative_to(root).parts)]
    with ZipFile(args.output, "x", compression=ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, Path(root.name) / path.relative_to(root))
    with ZipFile(args.output) as archive:
        assert archive.testzip() is None
    print(f"Verified {len(files)} files in {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
