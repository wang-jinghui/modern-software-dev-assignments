from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from ..exceptions import NoteNotFoundError, DatabaseOperationError
from ..repositories import NoteRepository
from ..schemas.note import Note, NoteCreate, NoteExtractRequest
from ..schemas.response import APIResponse


router = APIRouter(prefix="/notes", tags=["notes"])


@router.post("", response_model=APIResponse)
def create_note(note_create: NoteCreate) -> APIResponse:
    try:
        content = note_create.content.strip()
        if not content:
            raise HTTPException(status_code=400, detail="content is required")
        
        note_id = NoteRepository.create_note(content)
        note_row = NoteRepository.get_note(note_id)
        
        if not note_row:
            raise HTTPException(status_code=500, detail="Failed to retrieve created note")
        
        note = Note(
            id=note_row["id"],
            content=note_row["content"],
            created_at=note_row["created_at"]
        )
        
        return APIResponse(success=True, data=note)
    except HTTPException:
        raise
    except DatabaseOperationError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e.message}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.get("/{note_id}", response_model=APIResponse)
def get_single_note(note_id: int) -> APIResponse:
    try:
        row = NoteRepository.get_note(note_id)
        if row is None:
            raise NoteNotFoundError(note_id)
        
        note = Note(
            id=row["id"],
            content=row["content"],
            created_at=row["created_at"]
        )
        
        return APIResponse(success=True, data=note)
    except NoteNotFoundError:
        raise HTTPException(status_code=404, detail="note not found")
    except DatabaseOperationError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e.message}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.get("", response_model=APIResponse)
def list_all_notes() -> APIResponse:
    """New endpoint to list all notes as per assignment requirements"""
    try:
        rows = NoteRepository.list_notes()
        notes = [
            Note(
                id=row["id"],
                content=row["content"],
                created_at=row["created_at"]
            )
            for row in rows
        ]
        
        return APIResponse(success=True, data=notes)
    except DatabaseOperationError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e.message}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")