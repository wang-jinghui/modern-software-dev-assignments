from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional

import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .db import init_db
from .exceptions import ActionItemExtractionError, NoteNotFoundError, ActionItemNotFoundError, DatabaseOperationError
from .routers import action_items, notes


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan to handle startup and shutdown events"""
    # Startup
    logging.info("Initializing database...")
    init_db()
    logging.info("Database initialized successfully")
    
    yield  # Application runs here
    
    # Shutdown
    logging.info("Application shutdown")


app = FastAPI(title=settings.app_title, debug=settings.debug, lifespan=lifespan)


@app.exception_handler(NoteNotFoundError)
async def handle_note_not_found(request, exc: NoteNotFoundError):
    return HTTPException(status_code=404, detail=f"Note with id {exc.note_id} not found")


@app.exception_handler(ActionItemNotFoundError)
async def handle_action_item_not_found(request, exc: ActionItemNotFoundError):
    return HTTPException(status_code=404, detail=f"Action item with id {exc.action_item_id} not found")


@app.exception_handler(DatabaseOperationError)
async def handle_database_error(request, exc: DatabaseOperationError):
    logging.error(f"Database operation '{exc.operation}' failed: {exc.message}")
    return HTTPException(status_code=500, detail="Database operation failed")


@app.exception_handler(ActionItemExtractionError)
async def handle_extraction_error(request, exc: ActionItemExtractionError):
    logging.error(f"Action item extraction failed: {exc.message}")
    if exc.original_exception:
        logging.error(f"Original exception: {exc.original_exception}")
    return HTTPException(status_code=500, detail="Action item extraction failed")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    html_path = Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    return html_path.read_text(encoding="utf-8")


app.include_router(notes.router)
app.include_router(action_items.router)


static_dir = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")