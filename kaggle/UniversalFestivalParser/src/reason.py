"""Reason module for Universal Festival Parser (RDR Architecture).

Uses Gemma 3-27B to:
1. Analyze distilled content
2. Extract structured festival information
3. Output UDS-compliant JSON
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# System prompt for Gemma - GREEDY extraction strategy
SYSTEM_PROMPT = """You are a festival information extraction expert with GREEDY extraction strategy.
Your task is to MAXIMIZE data extraction from website content. Extract EVERYTHING available.

LANGUAGE: All text output (descriptions, titles, audience info) MUST be in RUSSIAN language!
Используй русский язык для всех текстовых полей!

CRITICAL: Pay attention to the "Today's date" at the start - use it for correct year inference!
If dates mention "декабрь" and we are in late December/early January, use the PREVIOUS year for December dates.
The festival is happening NOW or very recently.

Output ONLY valid JSON matching this schema:
{
  "festival": {
    "title_full": "Full festival name - REQUIRED",
    "title_short": "Short name or null",
    "description_short": "1-2 sentence engaging description - REQUIRED",
    "dates": {"start": "YYYY-MM-DD - REQUIRED", "end": "YYYY-MM-DD - REQUIRED"},
    "is_annual": true/false/null,
    "audience": "Target audience (e.g. 'families with children', 'music lovers')",
    "links": {
      "website": "Official URL - look for main domain",
      "socials": ["VK/TG/other social URLs"],
      "tickets": "Main ticket purchase URL if exists"
    },
    "registration": {
      "is_free": true/false/null,
      "common_url": "Main ticket/registration URL",
      "price_info": "Price range like '500-2000 руб' or 'бесплатно'"
    },
    "contacts": {
      "phone": "Phone number or null",
      "email": "Email or null"
    },
    "documents": [{"type": "pdf", "url": "URL", "title": "Document name"}]
  },
  "program": [
    {
      "title": "Event title - BE SPECIFIC",
      "type": "theater/concert/workshop/lecture/film/show/other",
      "date": "YYYY-MM-DD - USE CORRECT YEAR!",
      "time_start": "HH:MM or null",
      "time_end": "HH:MM or null",
      "venue": "Full venue name with address if available",
      "price": "Exact price like '500 руб' or 'бесплатно' or null",
      "is_free": true/false,
      "ticket_url": "Direct URL to buy tickets for THIS event",
      "description": "DETAILED description - include all available info: what happens, who performs, program highlights, age restrictions, etc.",
      "performers": ["Names of performers/actors if mentioned"]
    }
  ],
  "venues": [
    {"title": "Full venue name", "city": "City", "address": "Full street address"}
  ],
  "images_festival": ["Only real photo URLs, NOT .svg icons"]
}

EXTRACTION RULES (GREEDY):
1. Extract EVERY event from the program - don't skip any
2. For EACH event, look for its specific ticket URL (pyramida.info, etc)
3. Use CORRECT YEAR based on today's date context
4. Dates MUST be in ISO 8601 format (YYYY-MM-DD)
5. Extract performer names when mentioned
6. Include price information for each event
7. DO NOT include .svg icon URLs in images_festival
8. If an event repeats on multiple dates, create SEPARATE entries
9. Output ONLY the JSON, no explanations"""


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences from a response if present."""
    if "```json" in text:
        return text.split("```json")[1].split("```")[0].strip()
    if "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text.strip()


async def reason_with_gemma(
    distilled_content: str,
    api_key: str,
    model: str = "gemma-3-27b-it",
    llm_logger = None,
) -> tuple[dict | None, str | None]:
    """Call LLM to extract festival information.
    
    Args:
        distilled_content: Prepared content for analysis
        api_key: Google API key
        model: Model name
        llm_logger: Optional LLMLogger for tracking
        
    Returns:
        Tuple of (extracted_data dict or None, error message or None)
    """
    import google.generativeai as genai
    
    genai.configure(api_key=api_key)
    
    # For Gemma models, include system prompt in the main prompt
    # (system_instruction not supported for gemma-3-*)
    full_prompt = f"""{SYSTEM_PROMPT}

---

Analyze this festival website content and extract structured information.

{distilled_content}

Extract all festival details, program/schedule, venues, and images.
Output ONLY valid JSON matching the schema."""

    try:
        model_instance = genai.GenerativeModel(
            model_name=f"models/{model}",
        )
        
        if llm_logger:
            with llm_logger.track(
                phase="reason",
                model=model,
                prompt=full_prompt,
                content_length=len(distilled_content),
            ) as tracker:
                response = await model_instance.generate_content_async(
                    full_prompt,
                    generation_config={
                        "temperature": 0.1,
                        "max_output_tokens": 8192,
                    },
                )
                response_text = response.text or ""
                tracker.set_response(response_text)
        else:
            response = await model_instance.generate_content_async(
                full_prompt,
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 8192,
                },
            )
            response_text = response.text or ""
        
        response_text = _strip_code_fences(response_text)
        data = json.loads(response_text)
        logger.info("Successfully extracted festival data")
        return data, None
        
    except json.JSONDecodeError as e:
        error = f"Failed to parse JSON response: {e}"
        logger.error(error)
        return None, error
        
    except Exception as e:
        error = f"Gemma API error: {e}"
        logger.error(error)
        return None, error


async def validate_and_enhance(
    extracted_data: dict,
    distilled: dict,
    api_key: str,
    llm_logger = None,
) -> dict:
    """Validate and enhance extracted data with a second LLM pass.
    
    Args:
        extracted_data: Initial extraction from reason_with_gemma
        distilled: Original distilled content
        api_key: Google API key
        llm_logger: Optional LLMLogger
        
    Returns:
        Enhanced and validated data
    """
    import google.generativeai as genai
    
    genai.configure(api_key=api_key)
    
    # Check for missing critical fields
    festival = extracted_data.get("festival", {})
    missing = []
    
    if not festival.get("title_full") and not festival.get("title_short"):
        missing.append("festival name")
    if not festival.get("dates", {}).get("start"):
        missing.append("start date")
    
    if not missing:
        return extracted_data  # No validation needed
    
    validation_prompt = f"""The following festival data was extracted but is missing: {', '.join(missing)}

Please search the original content more carefully and fill in the missing fields.

Current extraction:
{json.dumps(extracted_data, ensure_ascii=False, indent=2)}

Original content:
{distilled.get('main_text', '')[:4000]}

Output the complete corrected JSON."""

    try:
        model = genai.GenerativeModel(
            model_name="models/gemma-3-27b-it",
        )
        
        if llm_logger:
            with llm_logger.track(
                phase="validate",
                model="gemma-3-27b",
                prompt=validation_prompt,
            ) as tracker:
                response = await model.generate_content_async(
                    validation_prompt,
                    generation_config={"temperature": 0.1},
                )
                tracker.set_response(response.text or "")
        else:
            response = await model.generate_content_async(
                validation_prompt,
                generation_config={"temperature": 0.1},
            )
        
        response_text = response.text or ""
        response_text = _strip_code_fences(response_text)
        
        enhanced = json.loads(response_text)
        logger.info("Enhanced data with validation pass")
        return enhanced
        
    except Exception as e:
        logger.warning("Validation pass failed: %s", e)
        return extracted_data  # Return original on failure
