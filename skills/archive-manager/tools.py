"""Archive Manager Skill - create/extract ZIP, TAR, GZIP archives."""

import os
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("archive-manager")

FORMAT_MAP = {
    "zip": "zip",
    "tar": "tar",
    "tar.gz": "tar.gz",
    "tgz": "tar.gz",
    "tar.bz2": "tar.bz2",
    "tbz2": "tar.bz2",
}


@tool_wrapper(required_params=["files", "output"])
def create_archive_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Create an archive from files."""
    status.set_callback(params.pop("_status_callback", None))
    files = params["files"]
    output = params["output"]
    fmt = FORMAT_MAP.get(params.get("format", "zip").lower(), "zip")

    existing = []
    for f in files:
        p = Path(f)
        if p.exists():
            existing.append(p)
        else:
            return tool_error(f"File not found: {f}")

    if not existing:
        return tool_error("No valid files to archive")

    try:
        if fmt == "zip":
            with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
                for fp in existing:
                    zf.write(fp, fp.name)
        else:
            mode_map = {"tar": "w", "tar.gz": "w:gz", "tar.bz2": "w:bz2"}
            with tarfile.open(output, mode_map[fmt]) as tf:
                for fp in existing:
                    tf.add(str(fp), arcname=fp.name)

        out_path = Path(output)
        return tool_response(
            output_path=str(out_path.resolve()),
            file_count=len(existing),
            size_bytes=out_path.stat().st_size,
            format=fmt,
        )
    except Exception as e:
        return tool_error(f"Archive creation failed: {e}")


@tool_wrapper(required_params=["archive_path"])
def extract_archive_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Extract an archive to a directory."""
    status.set_callback(params.pop("_status_callback", None))
    archive_path = Path(params["archive_path"])
    output_dir = params.get("output_dir", ".")

    if not archive_path.exists():
        return tool_error(f"Archive not found: {archive_path}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        extracted = []
        if zipfile.is_zipfile(str(archive_path)):
            with zipfile.ZipFile(str(archive_path), "r") as zf:
                zf.extractall(str(out))
                extracted = zf.namelist()
        elif tarfile.is_tarfile(str(archive_path)):
            with tarfile.open(str(archive_path), "r:*") as tf:
                tf.extractall(str(out))
                extracted = tf.getnames()
        else:
            return tool_error("Unsupported archive format")

        return tool_response(
            extracted_files=extracted,
            file_count=len(extracted),
            output_dir=str(out.resolve()),
        )
    except Exception as e:
        return tool_error(f"Extraction failed: {e}")


@tool_wrapper(required_params=["archive_path"])
def list_archive_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """List contents of an archive."""
    status.set_callback(params.pop("_status_callback", None))
    archive_path = Path(params["archive_path"])

    if not archive_path.exists():
        return tool_error(f"Archive not found: {archive_path}")

    try:
        files = []
        total_size = 0
        if zipfile.is_zipfile(str(archive_path)):
            with zipfile.ZipFile(str(archive_path), "r") as zf:
                for info in zf.infolist():
                    files.append(
                        {
                            "name": info.filename,
                            "size": info.file_size,
                            "compressed": info.compress_size,
                        }
                    )
                    total_size += info.file_size
        elif tarfile.is_tarfile(str(archive_path)):
            with tarfile.open(str(archive_path), "r:*") as tf:
                for member in tf.getmembers():
                    files.append(
                        {"name": member.name, "size": member.size, "is_dir": member.isdir()}
                    )
                    total_size += member.size
        else:
            return tool_error("Unsupported archive format")

        return tool_response(
            files=files,
            file_count=len(files),
            total_size=total_size,
            archive_size=archive_path.stat().st_size,
        )
    except Exception as e:
        return tool_error(f"Failed to list archive: {e}")


__all__ = ["create_archive_tool", "extract_archive_tool", "list_archive_tool"]
