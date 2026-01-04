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

# System prompt for Gemma
SYSTEM_PROMPT = """You are a festival information extraction expert. 
Your task is to analyze website content and extract structured information about a festival.

Output ONLY valid JSON matching this schema:
{
  "festival": {
    "title_full": "Full festival name or null",
    "title_short": "Short name or null",
    "description_short": "1-2 sentence description or null",
    "dates": {"start": "YYYY-MM-DD or null", "end": "YYYY-MM-DD or null"},
    "is_annual": true/false/null,
    "audience": "Target audience description or null",
    "links": {
      "website": "Official URL or null",
      "socials": ["VK/TG/other social URLs"]
    },
    "registration": {
      "is_free": true/false/null,
      "common_url": "Ticket/registration URL or null",
      "price_info": "Price description or null"
    },
    "contacts": {
      "phone": "Phone number or null",
      "email": "Email or null"
    },
    "documents": [{"type": "pdf", "url": "URL", "title": "Document name"}]
  },
  "program": [
    {
      "title": "Event title",
      "type": "film/concert/workshop/lecture/other",
      "date": "YYYY-MM-DD or null",
      "time_start": "HH:MM or null",
      "venue": "Location name or null",
      "price": "Price or 'free' or null",
      "is_free": true/false/null,
      "description": "Short description or null"
    }
  ],
  "venues": [
    {"title": "Venue name", "city": "City", "address": "Address or null"}
  ],
  "images_festival": ["URL1", "URL2"]
}

Rules:
- Use null for unknown/missing data, never invent information
- Dates must be ISO 8601 (YYYY-MM-DD)
- Extract ALL activities/events found in the program
- For films: include director, year, country if available
- Output ONLY the JSON, no explanations
"""


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
