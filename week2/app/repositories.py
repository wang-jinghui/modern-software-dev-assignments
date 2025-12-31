from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, List

from . import db
from .exceptions import NoteNotFoundError, ActionItemNotFoundError, DatabaseOperationError


class NoteRepository:
    @staticmethod
    def create_note(content: str) -> int:
        try:
            return db.insert_note(content)
        except Exception as e:
            raise DatabaseOperationError("insert_note", str(e))

    @staticmethod
    def get_note(note_id: int) -> Optional[sqlite3.Row]:
        try:
            return db.get_note(note_id)
        except Exception as e:
            raise DatabaseOperationError("get_note", str(e))

    @staticmethod
    def list_notes() -> List[sqlite3.Row]:
        try:
            return db.list_notes()
        except Exception as e:
            raise DatabaseOperationError("list_notes", str(e))


class ActionItemRepository:
    @staticmethod
    def create_action_items(items: List[str], note_id: Optional[int] = None) -> List[int]:
        try:
            return db.insert_action_items(items, note_id)
        except Exception as e:
            raise DatabaseOperationError("insert_action_items", str(e))

    @staticmethod
    def list_action_items(note_id: Optional[int] = None) -> List[sqlite3.Row]:
        try:
            return db.list_action_items(note_id)
        except Exception as e:
            raise DatabaseOperationError("list_action_items", str(e))

    @staticmethod
    def mark_action_item_done(action_item_id: int, done: bool) -> None:
        try:
            # Check if action item exists
            action_item = ActionItemRepository.get_action_item(action_item_id)
            if not action_item:
                raise ActionItemNotFoundError(action_item_id)
            
            db.mark_action_item_done(action_item_id, done)
        except ActionItemNotFoundError:
            raise
        except Exception as e:
            raise DatabaseOperationError("mark_action_item_done", str(e))

    @staticmethod
    def get_action_item(action_item_id: int) -> Optional[sqlite3.Row]:
        try:
            # We need to implement this method since it's used above
            with db.get_connection() as connection:
                cursor = connection.cursor()
                cursor.execute(
                    "SELECT id, note_id, text, done, created_at FROM action_items WHERE id = ?",
                    (action_item_id,)
                )
                return cursor.fetchone()
        except Exception as e:
            raise DatabaseOperationError("get_action_item", str(e))