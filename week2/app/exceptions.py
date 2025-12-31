from __future__ import annotations

from typing import Optional


class ActionItemExtractionError(Exception):
    """Raised when action item extraction fails"""
    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        self.message = message
        self.original_exception = original_exception
        super().__init__(self.message)


class NoteNotFoundError(Exception):
    """Raised when a note is not found"""
    def __init__(self, note_id: int):
        self.note_id = note_id
        super().__init__(f"Note with id {note_id} not found")


class ActionItemNotFoundError(Exception):
    """Raised when an action item is not found"""
    def __init__(self, action_item_id: int):
        self.action_item_id = action_item_id
        super().__init__(f"Action item with id {action_item_id} not found")


class DatabaseOperationError(Exception):
    """Raised when a database operation fails"""
    def __init__(self, operation: str, message: str):
        self.operation = operation
        self.message = message
        super().__init__(f"Database operation '{operation}' failed: {message}")