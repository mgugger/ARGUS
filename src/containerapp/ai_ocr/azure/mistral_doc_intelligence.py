"""
Mistral Document AI integration for OCR processing.
Provides an alternative to Azure Document Intelligence using Mistral's Document AI API.
"""
import base64
import json
import logging
import httpx
from typing import Optional, Union
from ai_ocr.azure.config import get_config

logger = logging.getLogger(__name__)


def encode_file_to_base64(file_path: str) -> tuple[str, str]:
    """
    Encode a file to base64 string and determine its type.
    
    Args:
        file_path: Path to the file to encode
        
    Returns:
        Tuple of (base64_string, file_type) where file_type is 'document_url' or 'image_url'
    """
    with open(file_path, "rb") as f:
        file_bytes = f.read()
        base64_encoded = base64.b64encode(file_bytes).decode('utf-8')
    
    # Determine file type and construct data URL
    if file_path.lower().endswith('.pdf'):
        data_url = f"data:application/pdf;base64,{base64_encoded}"
        url_type = "document_url"
    elif file_path.lower().endswith(('.jpg', '.jpeg')):
        data_url = f"data:image/jpeg;base64,{base64_encoded}"
        url_type = "image_url"
    elif file_path.lower().endswith('.png'):
        data_url = f"data:image/png;base64,{base64_encoded}"
        url_type = "image_url"
    else:
        # Default to document
        data_url = f"data:application/pdf;base64,{base64_encoded}"
        url_type = "document_url"
    
    return data_url, url_type


def get_mistral_doc_ai_client(cosmos_config_container=None):
    """
    Get Mistral Document AI configuration from environment.
    
    Args:
        cosmos_config_container: Optional Cosmos config container (kept for compatibility)
        
    Returns:
        Dictionary with endpoint, API key, and model name
    """
    config = get_config(cosmos_config_container)
    
    mistral_endpoint = config.get("mistral_doc_ai_endpoint")
    mistral_api_key = config.get("mistral_doc_ai_key")
    mistral_model = config.get("mistral_doc_ai_model", "mistral-document-ai-2505")
    
    if not mistral_endpoint or not mistral_api_key:
        raise ValueError(
            "Mistral Document AI endpoint and key must be configured. "
            "Set MISTRAL_DOC_AI_ENDPOINT and MISTRAL_DOC_AI_KEY environment variables."
        )
    
    return {
        "endpoint": mistral_endpoint,
        "api_key": mistral_api_key,
        "model": mistral_model
    }


def extract_bboxes_from_mistral_response(result: dict) -> dict:
    """
    Extract bounding box data from Mistral Document AI response.
    
    Mistral returns bboxes in a different format than Azure Document Intelligence.
    This function normalizes them to match the ARGUS polygon format.
    
    Args:
        result: The raw JSON response from Mistral Document AI
        
    Returns:
        Dictionary with normalized polygon data
    """
    words = []
    lines = []
    
    if "pages" not in result or not isinstance(result["pages"], list):
        return {"words": words, "lines": lines, "keyValuePairs": [], "paragraphs": []}
    
    for page_idx, page in enumerate(result["pages"]):
        page_number = page_idx + 1
        
        # Extract words with bboxes if available
        if "words" in page and isinstance(page["words"], list):
            for word in page["words"]:
                bbox = word.get("bbox", word.get("bounding_box", []))
                words.append({
                    "content": word.get("text", word.get("content", "")),
                    "confidence": word.get("confidence"),
                    "pageNumber": page_number,
                    "points": normalize_mistral_bbox(bbox)
                })
        
        # Extract lines with bboxes if available
        if "lines" in page and isinstance(page["lines"], list):
            for line in page["lines"]:
                bbox = line.get("bbox", line.get("bounding_box", []))
                lines.append({
                    "content": line.get("text", line.get("content", "")),
                    "pageNumber": page_number,
                    "points": normalize_mistral_bbox(bbox)
                })
        
        # Some Mistral responses have blocks instead of lines
        if "blocks" in page and isinstance(page["blocks"], list):
            for block in page["blocks"]:
                bbox = block.get("bbox", block.get("bounding_box", []))
                if bbox:
                    lines.append({
                        "content": block.get("text", block.get("content", "")),
                        "pageNumber": page_number,
                        "points": normalize_mistral_bbox(bbox)
                    })
    
    return {
        "words": words,
        "lines": lines,
        "keyValuePairs": [],  # Mistral doesn't provide KV pairs like Azure Doc Intel
        "paragraphs": []
    }


def normalize_mistral_bbox(bbox: list) -> list:
    """
    Normalize Mistral bbox format to ARGUS polygon format.
    
    Mistral may return bbox as [x, y, width, height] or [x1, y1, x2, y2, x3, y3, x4, y4].
    We normalize to the 8-point polygon format.
    
    Args:
        bbox: Bounding box from Mistral
        
    Returns:
        List of 8 points [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    if not bbox:
        return []
    
    if len(bbox) == 4:
        # Assume [x, y, width, height] format - convert to 4 corner points
        x, y, w, h = bbox
        return [x, y, x + w, y, x + w, y + h, x, y + h]
    elif len(bbox) == 8:
        # Already in polygon format
        return bbox
    else:
        # Unknown format, return as-is
        return bbox


def get_ocr_results(file_path: str, cosmos_config_container=None, json_schema: Optional[dict] = None, include_polygons: bool = False) -> Union[str, dict]:
    """
    Extract text from document using Mistral Document AI.
    
    This function provides the same interface as Azure Document Intelligence
    to allow seamless switching between OCR providers.
    
    Args:
        file_path: Path to the file to process
        cosmos_config_container: Optional Cosmos config container (kept for compatibility)
        json_schema: Optional JSON schema for structured extraction with bbox annotation
        include_polygons: If True, return full result with polygon data; if False, return only text content
        
    Returns:
        If include_polygons is False: str - OCR text content
        If include_polygons is True: dict - Full result with content and polygon data
    """
    import threading
    thread_id = threading.current_thread().ident
    
    logger.info(f"[Thread-{thread_id}] Starting Mistral Document AI OCR for: {file_path}")
    logger.info(f"[Thread-{thread_id}] Include polygons: {include_polygons}")
    
    # Get Mistral configuration
    mistral_config = get_mistral_doc_ai_client(cosmos_config_container)
    endpoint = mistral_config["endpoint"]
    api_key = mistral_config["api_key"]
    model_name = mistral_config["model"]
    
    # Encode file to base64
    logger.info(f"[Thread-{thread_id}] Encoding file to base64")
    data_url, url_type = encode_file_to_base64(file_path)
    
    # Prepare request payload
    payload = {
        "model": model_name,
        "document": {
            "type": url_type,
            url_type: data_url
        },
        "include_image_base64": False  # We don't need images back, just text
    }
    
    # If JSON schema is provided or include_polygons is True, add bbox annotation format
    if json_schema or include_polygons:
        logger.info(f"[Thread-{thread_id}] Adding bbox annotation format" + (" with schema" if json_schema else ""))
        if json_schema:
            payload["bbox_annotation_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "schema": json_schema,
                    "name": "document_annotation",
                    "strict": True
                }
            }
        else:
            # Request bboxes without schema constraint
            payload["include_bboxes"] = True
    
    # Make request to Mistral API
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    logger.info(f"[Thread-{thread_id}] Submitting document to Mistral Document AI API")
    
    try:
        with httpx.Client(timeout=300.0) as client:  # 5 minute timeout for large documents
            response = client.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"[Thread-{thread_id}] Mistral Document AI response received")
            
            # Extract markdown content from response
            # Mistral Document AI returns pages with markdown content
            ocr_text = ""
            
            if "pages" in result and isinstance(result["pages"], list):
                # Concatenate markdown from all pages
                markdown_parts = []
                for page in result["pages"]:
                    if isinstance(page, dict) and "markdown" in page:
                        markdown_parts.append(page["markdown"])
                ocr_text = "\n\n".join(markdown_parts)
                logger.info(f"[Thread-{thread_id}] Extracted markdown from {len(result['pages'])} page(s)")
            elif "content" in result:
                ocr_text = result["content"]
            elif "text" in result:
                ocr_text = result["text"]
            elif "choices" in result and len(result["choices"]) > 0:
                # OpenAI-style response format
                ocr_text = result["choices"][0].get("message", {}).get("content", "")
            else:
                # Fallback: log warning
                logger.warning(f"[Thread-{thread_id}] Unexpected response format, no markdown content found")
                ocr_text = ""
            
            logger.info(f"[Thread-{thread_id}] Mistral Document AI OCR completed, {len(ocr_text)} characters")
            
            if not include_polygons:
                return ocr_text
            
            # Extract polygon data from Mistral response
            logger.info(f"[Thread-{thread_id}] Extracting polygon data from Mistral response")
            polygon_data = extract_bboxes_from_mistral_response(result)
            
            return {
                "content": ocr_text,
                "words": polygon_data["words"],
                "lines": polygon_data["lines"],
                "keyValuePairs": polygon_data["keyValuePairs"],
                "paragraphs": polygon_data["paragraphs"]
            }
            
    except httpx.HTTPStatusError as e:
        logger.error(f"[Thread-{thread_id}] Mistral API HTTP error: {e.response.status_code}")
        logger.error(f"[Thread-{thread_id}] Response: {e.response.text}")
        raise Exception(f"Mistral Document AI API error: {e.response.status_code} - {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"[Thread-{thread_id}] Mistral API request error: {str(e)}")
        raise Exception(f"Mistral Document AI request failed: {str(e)}")
    except Exception as e:
        logger.error(f"[Thread-{thread_id}] Unexpected error during Mistral Document AI processing: {str(e)}")
        raise
