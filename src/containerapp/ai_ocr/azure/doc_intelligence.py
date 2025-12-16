import json
import pandas as pd
from typing import Optional, List, Dict, Any, Union
from azure.identity import DefaultAzureCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentAnalysisFeature
from ai_ocr.azure.config import get_config


def get_document_intelligence_client(cosmos_config_container=None):
    """Create a new Document Intelligence client instance for each request to avoid connection pooling issues"""
    config = get_config(cosmos_config_container)
    return DocumentIntelligenceClient(
        endpoint=config["doc_intelligence_endpoint"],
        credential=DefaultAzureCredential(),
        headers={"solution":"ARGUS-1.0"}
    )


def extract_polygon_from_bounding_regions(bounding_regions: List[Any]) -> List[Dict[str, Any]]:
    """
    Extract polygon data from Document Intelligence bounding regions.
    
    Args:
        bounding_regions: List of bounding region objects from Document Intelligence
        
    Returns:
        List of polygon dictionaries with points and page number
    """
    polygons = []
    if not bounding_regions:
        return polygons
    
    for region in bounding_regions:
        polygon_data = {
            "pageNumber": getattr(region, 'page_number', 1),
            "points": list(getattr(region, 'polygon', [])) if hasattr(region, 'polygon') else []
        }
        polygons.append(polygon_data)
    
    return polygons


def extract_words_with_polygons(pages: List[Any]) -> List[Dict[str, Any]]:
    """
    Extract all words with their bounding polygons from Document Intelligence pages.
    
    Args:
        pages: List of page objects from Document Intelligence result
        
    Returns:
        List of word dictionaries with content, polygon, and page number
    """
    words_data = []
    if not pages:
        return words_data
    
    for page in pages:
        page_number = getattr(page, 'page_number', 1)
        words = getattr(page, 'words', []) or []
        
        for word in words:
            word_data = {
                "content": getattr(word, 'content', ''),
                "confidence": getattr(word, 'confidence', None),
                "pageNumber": page_number,
                "points": list(getattr(word, 'polygon', [])) if hasattr(word, 'polygon') else []
            }
            words_data.append(word_data)
    
    return words_data


def extract_lines_with_polygons(pages: List[Any]) -> List[Dict[str, Any]]:
    """
    Extract all lines with their bounding polygons from Document Intelligence pages.
    
    Args:
        pages: List of page objects from Document Intelligence result
        
    Returns:
        List of line dictionaries with content, polygon, and page number
    """
    lines_data = []
    if not pages:
        return lines_data
    
    for page in pages:
        page_number = getattr(page, 'page_number', 1)
        lines = getattr(page, 'lines', []) or []
        
        for line in lines:
            line_data = {
                "content": getattr(line, 'content', ''),
                "pageNumber": page_number,
                "points": list(getattr(line, 'polygon', [])) if hasattr(line, 'polygon') else []
            }
            lines_data.append(line_data)
    
    return lines_data


def extract_key_value_pairs(key_value_pairs: List[Any]) -> List[Dict[str, Any]]:
    """
    Extract key-value pairs with their bounding polygons from Document Intelligence result.
    
    Args:
        key_value_pairs: List of key-value pair objects from Document Intelligence
        
    Returns:
        List of key-value dictionaries with key, value, and their polygons
    """
    kv_data = []
    if not key_value_pairs:
        return kv_data
    
    for kv_pair in key_value_pairs:
        key_obj = getattr(kv_pair, 'key', None)
        value_obj = getattr(kv_pair, 'value', None)
        
        kv_entry = {
            "key": {
                "content": getattr(key_obj, 'content', '') if key_obj else '',
                "boundingPolygons": extract_polygon_from_bounding_regions(
                    getattr(key_obj, 'bounding_regions', []) if key_obj else []
                )
            },
            "value": {
                "content": getattr(value_obj, 'content', '') if value_obj else '',
                "boundingPolygons": extract_polygon_from_bounding_regions(
                    getattr(value_obj, 'bounding_regions', []) if value_obj else []
                )
            },
            "confidence": getattr(kv_pair, 'confidence', None)
        }
        kv_data.append(kv_entry)
    
    return kv_data


def extract_paragraphs_with_polygons(paragraphs: List[Any]) -> List[Dict[str, Any]]:
    """
    Extract paragraphs with their bounding polygons from Document Intelligence result.
    
    Args:
        paragraphs: List of paragraph objects from Document Intelligence
        
    Returns:
        List of paragraph dictionaries with content and polygons
    """
    para_data = []
    if not paragraphs:
        return para_data
    
    for paragraph in paragraphs:
        para_entry = {
            "content": getattr(paragraph, 'content', ''),
            "role": getattr(paragraph, 'role', None),
            "boundingPolygons": extract_polygon_from_bounding_regions(
                getattr(paragraph, 'bounding_regions', [])
            )
        }
        para_data.append(para_entry)
    
    return para_data


def get_ocr_results(file_path: str, cosmos_config_container=None, include_polygons: bool = False) -> Union[str, Dict[str, Any]]:
    """
    Get OCR results from Document Intelligence.
    
    Args:
        file_path: Path to the document file
        cosmos_config_container: Optional Cosmos config container
        include_polygons: If True, return full result with polygon data; if False, return only text content
        
    Returns:
        If include_polygons is False: str - OCR text content
        If include_polygons is True: dict - Full result with content and polygon data
    """
    import threading
    import logging
    
    thread_id = threading.current_thread().ident
    logger = logging.getLogger(__name__)
    
    logger.info(f"[Thread-{thread_id}] Starting Document Intelligence OCR for: {file_path}")
    logger.info(f"[Thread-{thread_id}] Include polygons: {include_polygons}")
    
    # Create a new client instance for this request to ensure parallel processing
    client = get_document_intelligence_client(cosmos_config_container)
    
    with open(file_path, "rb") as f:
        logger.info(f"[Thread-{thread_id}] Submitting document to Document Intelligence API")
        poller = client.begin_analyze_document("prebuilt-layout", body=f)

    logger.info(f"[Thread-{thread_id}] Waiting for Document Intelligence results...")
    result = poller.result()
    
    ocr_content = result.content
    logger.info(f"[Thread-{thread_id}] Document Intelligence OCR completed, {len(ocr_content)} characters")
    
    if not include_polygons:
        return ocr_content
    
    # Extract full polygon data for correlation
    logger.info(f"[Thread-{thread_id}] Extracting polygon data from Document Intelligence result")
    
    polygon_data = {
        "content": ocr_content,
        "words": extract_words_with_polygons(getattr(result, 'pages', [])),
        "lines": extract_lines_with_polygons(getattr(result, 'pages', [])),
        "keyValuePairs": extract_key_value_pairs(getattr(result, 'key_value_pairs', [])),
        "paragraphs": extract_paragraphs_with_polygons(getattr(result, 'paragraphs', []))
    }
    
    logger.info(f"[Thread-{thread_id}] Extracted {len(polygon_data['words'])} words, "
                f"{len(polygon_data['lines'])} lines, "
                f"{len(polygon_data['keyValuePairs'])} key-value pairs, "
                f"{len(polygon_data['paragraphs'])} paragraphs with polygons")
    
    return polygon_data

