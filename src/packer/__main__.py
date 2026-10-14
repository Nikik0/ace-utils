"""Command-line interface for .kspkg packages (``python -m src.packer``)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.packer.install import default_mods_dir, install_kspkg
from src.packer.kspkg import KspkgReader, KspkgWriter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="src.packer", description="KSPackage (.kspkg) utility")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list files in a package")
    list_parser.add_argument("kspkg", type=Path)

    unpack_parser = subparsers.add_parser("unpack", help="extract a package")
    unpack_parser.add_argument("kspkg", type=Path)
    unpack_parser.add_argument("-o", "--out", type=Path, required=True)

    pack_parser = subparsers.add_parser("pack", help="pack a folder into a .kspkg")
    pack_parser.add_argument("folder", type=Path)
    pack_parser.add_argument("-o", "--out", type=Path, required=True)
    pack_parser.add_argument("--prefix", default="", help="internal path prefix (forward slashes)")

    install_parser = subparsers.add_parser("install", help="copy a package to the mods folder")
    install_parser.add_argument("kspkg", type=Path)
    install_parser.add_argument("--mods", type=Path, default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "list":
        with KspkgReader(args.kspkg) as reader:
            for entry in reader.entries:
                kind = "dir" if entry.is_directory else "file"
                print(f"{kind} {entry.size:>12} {entry.path}")
        return 0

    if args.command == "unpack":
        with KspkgReader(args.kspkg) as reader:
            count = reader.extract_all(args.out)
        print(f"extracted {count} files to {args.out}")
        return 0

    if args.command == "pack":
        writer = KspkgWriter()
        count = writer.add_tree(args.folder, args.prefix)
        writer.write(args.out)
        print(f"packed {count} files into {args.out}")
        return 0

    if args.command == "install":
        target = args.mods if args.mods is not None else default_mods_dir()
        dest = install_kspkg(args.kspkg, target)
        print(f"installed to {dest}")
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
