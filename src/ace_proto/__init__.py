"""Generated ACE protobuf schemas recovered from AssettoCorsaEVO.exe.

The modules here are produced from ``refs/ace-protos/raw`` by:

    python -m grpc_tools.protoc -I refs/ace-protos/raw --python_out=src/ace_proto <files>

protoc emits absolute imports (e.g. ``import Scene_pb2``), so this package
places its own directory on ``sys.path`` before any generated module loads.
Import ``src.ace_proto`` first, then the generated modules resolve.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DIR = str(Path(__file__).resolve().parent)
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)
