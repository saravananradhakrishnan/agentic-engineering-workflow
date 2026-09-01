"""Deterministic filesystem tools with strict workspace boundary enforcement."""

import os
from pathlib import Path
from typing import List, Optional, Union
from pydantic import BaseModel, Field

# Default workspace directory relative to root project
DEFAULT_WORKSPACE_DIR = Path("workspace/generated_app")


class WriteFileResult(BaseModel):
    """Structured result model for write_file tool operations."""

    success: bool = Field(description="Indicates if the file write operation succeeded")
    path: str = Field(description="Target relative path of the file")
    message: str = Field(description="Human-readable success or status message")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")


class ReadFileResult(BaseModel):
    """Structured result model for read_file tool operations."""

    success: bool = Field(description="Indicates if the file read operation succeeded")
    path: str = Field(description="Target relative path of the file")
    content: Optional[str] = Field(default=None, description="UTF-8 text content of the file")
    message: str = Field(description="Human-readable success or status message")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")


class ListFilesResult(BaseModel):
    """Structured result model for list_files tool operations."""

    success: bool = Field(description="Indicates if the listing operation succeeded")
    files: List[str] = Field(default_factory=list, description="List of relative file paths in workspace")
    message: str = Field(description="Human-readable summary message")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")


def get_safe_path(
    relative_path: str, base_dir: Optional[Union[str, Path]] = None
) -> Path:
    """Enforces workspace boundary security policy.

    Rejects:
    1. Absolute paths (e.g., '/etc/passwd', 'C:\\Windows')
    2. Path traversal attempts containing '..' components
    3. Paths resolving outside the designated workspace directory

    Args:
        relative_path: User/agent supplied relative path.
        base_dir: Optional workspace root override (defaults to workspace/generated_app).

    Returns:
        Resolved absolute Path within workspace.

    Raises:
        ValueError: If any path security rule is violated.
    """
    if base_dir is None:
        base_workspace = DEFAULT_WORKSPACE_DIR.resolve()
    else:
        base_workspace = Path(base_dir).resolve()

    raw_path = Path(relative_path)

    # 1. Reject absolute paths
    if raw_path.is_absolute():
        raise ValueError(
            f"Absolute paths are rejected for security reasons: '{relative_path}'"
        )

    # 2. Reject explicit path traversal elements ('..')
    if ".." in raw_path.parts:
        raise ValueError(
            f"Path traversal ('..') is strictly prohibited: '{relative_path}'"
        )

    # 3. Resolve full target path
    target_path = (base_workspace / raw_path).resolve()

    # 4. Verify boundary constraint (target must be inside base_workspace)
    try:
        target_path.relative_to(base_workspace)
    except ValueError:
        raise ValueError(
            f"Path escape attempt detected. Target '{relative_path}' resolves outside workspace boundary '{base_workspace}'."
        )

    return target_path


def write_file(
    path: str, content: str, base_dir: Optional[Union[str, Path]] = None
) -> WriteFileResult:
    """Write text content to a file strictly within workspace/generated_app/.

    Args:
        path: Relative file path within workspace (e.g. 'app/calculator.py').
        content: Text content to write into file.
        base_dir: Optional workspace directory override.

    Returns:
        WriteFileResult containing status and error details.
    """
    try:
        target_path = get_safe_path(path, base_dir=base_dir)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content, encoding="utf-8")
        return WriteFileResult(
            success=True,
            path=path,
            message=f"Successfully wrote file: '{path}'",
        )
    except Exception as e:
        return WriteFileResult(
            success=False,
            path=path,
            error=str(e),
            message=f"Failed to write file '{path}': {e}",
        )


def read_file(
    path: str, base_dir: Optional[Union[str, Path]] = None
) -> ReadFileResult:
    """Read text content from a file strictly within workspace/generated_app/.

    Args:
        path: Relative file path within workspace.
        base_dir: Optional workspace directory override.

    Returns:
        ReadFileResult containing file content or error details.
    """
    try:
        target_path = get_safe_path(path, base_dir=base_dir)
        if not target_path.exists() or not target_path.is_file():
            return ReadFileResult(
                success=False,
                path=path,
                error=f"File does not exist: '{path}'",
                message=f"File non-existent: '{path}'",
            )

        content = target_path.read_text(encoding="utf-8")
        return ReadFileResult(
            success=True,
            path=path,
            content=content,
            message=f"Successfully read file: '{path}'",
        )
    except Exception as e:
        return ReadFileResult(
            success=False,
            path=path,
            error=str(e),
            message=f"Failed to read file '{path}': {e}",
        )


def list_files(
    base_dir: Optional[Union[str, Path]] = None
) -> ListFilesResult:
    """Recursively list all relative file paths within workspace/generated_app/.

    Args:
        base_dir: Optional workspace directory override.

    Returns:
        ListFilesResult containing list of relative file path strings.
    """
    try:
        if base_dir is None:
            base_workspace = DEFAULT_WORKSPACE_DIR.resolve()
        else:
            base_workspace = Path(base_dir).resolve()

        base_workspace.mkdir(parents=True, exist_ok=True)

        relative_files = []
        for file_path in base_workspace.rglob("*"):
            if file_path.is_file():
                rel_p = str(file_path.relative_to(base_workspace))
                relative_files.append(rel_p)

        relative_files.sort()
        return ListFilesResult(
            success=True,
            files=relative_files,
            message=f"Successfully listed {len(relative_files)} files in workspace.",
        )
    except Exception as e:
        return ListFilesResult(
            success=False,
            files=[],
            error=str(e),
            message=f"Failed to list files in workspace: {e}",
        )
