from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel


class APIResponse(BaseModel):
    """Generic API response schema"""
    success: bool
    message: Optional[str] = None
    data: Optional[Any] = None
    error: Optional[dict] = None


class ActionItemExtractResponse(BaseModel):
    """Response schema for action item extraction"""
    note_id: Optional[int] = None
    items: List[dict]  # Contains id and text of each action item