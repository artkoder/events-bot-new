"""Universal Data Structure (UDS) schema for festival parsing.

Defines the Pydantic models for structured festival data output.
All fields are optional to follow "null if unknown" strategy.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, HttpUrl


class UDSVenue(BaseModel):
    """Venue/location information."""
    title: Optional[str] = None
    city: Optional[str] = None
    address: Optional[str] = None
    coordinates: Optional[tuple[float, float]] = None
    indoor: Optional[bool] = None


class UDSActivity(BaseModel):
    """Single activity/event in the festival program."""
    title: Optional[str] = None
    type: Optional[str] = None  # "film" | "concert" | "workshop" | "lecture" | etc.
    date: Optional[str] = None  # ISO 8601
    time_start: Optional[str] = None  # HH:MM
    time_end: Optional[str] = None
    duration_minutes: Optional[int] = None
    venue: Optional[str] = None  # Venue title reference
    description: Optional[str] = None
    price: Optional[str] = None
    is_free: Optional[bool] = None
    registration_url: Optional[str] = None
    speakers: Optional[list[str]] = None
    
    # Film-specific fields
    director: Optional[str] = None
    year: Optional[int] = None
    country: Optional[str] = None
    language: Optional[str] = None
    subtitles: Optional[str] = None


class UDSDocument(BaseModel):
    """Linked document (PDF program, etc.)."""
    type: Optional[str] = None  # "pdf" | "doc" | "image"
    url: Optional[str] = None
    title: Optional[str] = None


class UDSFestival(BaseModel):
    """Core festival information."""
    title_full: Optional[str] = None
    title_short: Optional[str] = None
    description_short: Optional[str] = None
    description_full: Optional[str] = None
    dates: Optional[dict[str, str]] = Field(
        default=None,
        description="{'start': 'YYYY-MM-DD', 'end': 'YYYY-MM-DD'}"
    )
    is_annual: Optional[bool] = None
    edition: Optional[int] = None  # Which edition (e.g., 5th annual)
    audience: Optional[str] = None  # Target audience description
    links: Optional[dict[str, Any]] = Field(
        default=None,
        description="{'website': 'url', 'socials': ['url1', 'url2']}"
    )
    registration: Optional[dict[str, Any]] = Field(
        default=None,
        description="{'is_free': bool, 'common_url': 'url', 'price_info': 'text'}"
    )
    contacts: Optional[dict[str, str]] = Field(
        default=None,
        description="{'phone': '+7...', 'email': 'fest@...'}"
    )
    documents: Optional[list[UDSDocument]] = None
    past_editions: Optional[list[dict]] = None  # Previous years info


class UDSOutput(BaseModel):
    """Complete UDS output from parser."""
    
    # Metadata
    uds_version: str = "1.0.0"
    source_url: str
    extracted_at: str  # ISO 8601 timestamp
    parser_version: str
    run_id: str
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Overall extraction confidence (0.0-1.0)"
    )
    
    # Content
    festival: UDSFestival
    program: list[UDSActivity] = Field(default_factory=list)
    venues: list[UDSVenue] = Field(default_factory=list)
    images_festival: list[str] = Field(
        default_factory=list,
        description="URLs of festival images"
    )
    
    # Processing metadata
    llm_model: Optional[str] = None
    processing_time_ms: Optional[float] = None
    warnings: list[str] = Field(default_factory=list)
    
    class Config:
        extra = "allow"  # Allow additional fields for future expansion


def validate_uds(data: dict) -> tuple[UDSOutput | None, list[str]]:
    """Validate UDS data against schema.
    
    Args:
        data: Raw dict from LLM or JSON file
        
    Returns:
        Tuple of (validated UDSOutput or None, list of validation errors)
    """
    errors = []
    
    try:
        uds = UDSOutput(**data)
        return uds, []
    except Exception as e:
        errors.append(f"Validation error: {e}")
        return None, errors


def create_empty_uds(
    source_url: str,
    run_id: str,
    parser_version: str,
) -> UDSOutput:
    """Create empty UDS structure with required metadata."""
    return UDSOutput(
        source_url=source_url,
        extracted_at=datetime.utcnow().isoformat() + "Z",
        parser_version=parser_version,
        run_id=run_id,
        festival=UDSFestival(),
    )
