from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from .. import db
from ..exceptions import ActionItemNotFoundError, DatabaseOperationError
from ..repositories import ActionItemRepository, NoteRepository
from ..schemas.action_item import ActionItemExtractRequest, ActionItemMarkDoneRequest
from ..schemas.response import ActionItemExtractResponse, APIResponse
from ..services.extract import extract_action_items, extract_action_items_llm


router = APIRouter(prefix="/action-items", tags=["action-items"])


@router.post("/extract", response_model=APIResponse)
def extract(payload: ActionItemExtractRequest) -> APIResponse:
    try:
        text = payload.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="text is required")

        note_id: Optional[int] = None
        if payload.save_note:
            note_id = NoteRepository.create_note(text)

        items = extract_action_items(text)
        ids = ActionItemRepository.create_action_items(items, note_id=note_id)
        
        response_data = ActionItemExtractResponse(
            note_id=note_id,
            items=[{"id": i, "text": t} for i, t in zip(ids, items)]
        )
        
        return APIResponse(success=True, data=response_data)
    except DatabaseOperationError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e.message}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.post("/extract-llm", response_model=APIResponse)
def extract_llm(payload: ActionItemExtractRequest) -> APIResponse:
    """New endpoint for LLM-powered action item extraction"""
    try:
        text = payload.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="text is required")

        note_id: Optional[int] = None
        if payload.save_note:
            note_id = NoteRepository.create_note(text)

        items = extract_action_items_llm(text)
        ids = ActionItemRepository.create_action_items(items, note_id=note_id)
        
        response_data = ActionItemExtractResponse(
            note_id=note_id,
            items=[{"id": i, "text": t} for i, t in zip(ids, items)]
        )
        
        return APIResponse(success=True, data=response_data)
    except DatabaseOperationError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e.message}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.get("", response_model=APIResponse)
def list_all(note_id: Optional[int] = None) -> APIResponse:
    try:
        rows = ActionItemRepository.list_action_items(note_id=note_id)
        action_items = [
            {
                "id": r["id"],
                "note_id": r["note_id"],
                "text": r["text"],
                "done": bool(r["done"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]
        
        return APIResponse(success=True, data=action_items)
    except DatabaseOperationError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e.message}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.post("/{action_item_id}/done", response_model=APIResponse)
def mark_done(action_item_id: int, payload: ActionItemMarkDoneRequest) -> APIResponse:
    try:
        done = payload.done
        ActionItemRepository.mark_action_item_done(action_item_id, done)
        
        result = {"id": action_item_id, "done": done}
        return APIResponse(success=True, data=result)
    except ActionItemNotFoundError:
        raise HTTPException(status_code=404, detail="action item not found")
    except DatabaseOperationError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e.message}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")