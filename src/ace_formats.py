"""Mapping between ACE asset file extensions and recovered protobuf schemas.

The schemas live in ``refs/ace-protos/raw`` (recovered from AssettoCorsaEVO.exe
with protodump) and are compiled into ``src.ace_proto`` by::

    python -m grpc_tools.protoc -I refs/ace-protos/raw \
        --python_out=src/ace_proto <raw/*.proto>

Only file types that are protobuf are listed here. Others (``.texturemips``
raw mip blobs, ``.track_layout`` custom binary) are handled elsewhere.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from google.protobuf.message import Message

import src.ace_proto  # noqa: F401  (registers the generated modules on sys.path)


@dataclass(frozen=True, slots=True)
class AceFormat:
    """An ACE asset file type and the protobuf message that encodes it."""

    extension: str
    module: str
    message: str

    def message_class(self) -> type[Message]:
        module = importlib.import_module(self.module)
        cls = cast("type[Message]", getattr(module, self.message))
        if not issubclass(cls, Message):
            raise TypeError(f"{self.module}.{self.message} is not a protobuf message")
        return cls


FORMATS: tuple[AceFormat, ...] = (
    AceFormat(".track", "TrackData_pb2", "TrackData"),
    AceFormat(".mesh", "Mesh_pb2", "MeshData"),
    AceFormat(".material", "Renderer_pb2", "MaterialData"),
    AceFormat(".texture", "Renderer_pb2", "TextureMetadata"),
    AceFormat(".scene", "Scene_pb2", "SceneData"),
    AceFormat(".trackcontrolpoints", "AIData_pb2", "TrackControlPoints"),
    AceFormat(".aisplinedata", "AIData_pb2", "AISplineData"),
    AceFormat(".reference", "AIData_pb2", "AISplineReferenceData"),
)

_BY_EXTENSION: dict[str, AceFormat] = {fmt.extension: fmt for fmt in FORMATS}


def format_for(path: str | Path) -> AceFormat | None:
    """Return the protobuf format for a file path, or None if not protobuf."""
    return _BY_EXTENSION.get(Path(path).suffix)


def new_message(extension: str) -> Message:
    """Create an empty message for a file extension (e.g. ``".track"``)."""
    fmt = _BY_EXTENSION.get(extension)
    if fmt is None:
        raise ValueError(f"no protobuf schema for extension: {extension}")
    return fmt.message_class()()


def decode(path: str | Path) -> Message:
    """Read and parse a protobuf asset file into its message."""
    fmt = format_for(path)
    if fmt is None:
        raise ValueError(f"no protobuf schema for: {path}")
    message = fmt.message_class()()
    message.ParseFromString(Path(path).read_bytes())
    return message


def encode(message: Message, path: str | Path) -> None:
    """Serialize a message to a file path."""
    Path(path).write_bytes(message.SerializeToString())
