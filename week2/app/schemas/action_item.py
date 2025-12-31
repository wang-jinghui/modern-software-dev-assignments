from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ActionItemBase(BaseModel):
    text: str = Field(..., min_length=1, description="Action item text")
    note_id: Optional[int] = Field(default=None, description="Associated note ID")


class ActionItemCreate(ActionItemBase):
    pass


class ActionItem(ActionItemBase):
    id: int
    done: bool
    created_at: datetime

    class Config:
        from_attributes = True


class ActionItemMarkDoneRequest(BaseModel):
    done: bool = Field(default=True, description="Whether the action item is done")


class ActionItemExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to extract action items from")
    save_note: bool = Field(default=False, description="Whether to save the note to database")