from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class NoteBase(BaseModel):
    content: str = Field(..., min_length=1, description="Note content")


class NoteCreate(NoteBase):
    pass


class Note(NoteBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class NoteExtractRequest(BaseModel):
    content: str = Field(..., min_length=1, description="Note content to extract action items from")
    save_note: bool = Field(default=False, description="Whether to save the note to database")