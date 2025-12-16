"""
Polygon Matcher Module for ARGUS

This module correlates GPT-extracted values with bounding polygons from Document Intelligence.
It uses a hybrid approach:
1. First pass: Match against Document Intelligence key-value pairs by key name similarity
2. Second pass: Fuzzy string match remaining values against words/lines (90% threshold)

All matching occurrences are returned as an array to handle repeated values.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

# Default fuzzy matching threshold (90%)
DEFAULT_FUZZY_THRESHOLD = 90.0


def normalize_key(key: str) -> str:
    """
    Normalize a key string for comparison.
    Removes common variations in key naming.
    """
    return key.lower().replace('_', ' ').replace('-', ' ').strip()


def find_key_value_polygon(
    field_name: str,
    field_value: Any,
    key_value_pairs: List[Dict[str, Any]],
    threshold: float = DEFAULT_FUZZY_THRESHOLD
) -> List[Dict[str, Any]]:
    """
    Find matching polygon(s) from Document Intelligence key-value pairs.
    
    Args:
        field_name: The name of the field being matched
        field_value: The value of the field
        key_value_pairs: List of key-value pairs from Document Intelligence
        threshold: Minimum similarity threshold for key matching
        
    Returns:
        List of matching bounding polygons with metadata
    """
    matches = []
    normalized_field_name = normalize_key(field_name)
    str_field_value = str(field_value).strip() if field_value else ""
    
    for kv_pair in key_value_pairs:
        key_content = kv_pair.get('key', {}).get('content', '')
        value_content = kv_pair.get('value', {}).get('content', '')
        
        # Check if key matches
        key_similarity = fuzz.ratio(normalized_field_name, normalize_key(key_content))
        
        if key_similarity >= threshold:
            # Key matches, check if value also matches (or is close enough)
            value_similarity = fuzz.ratio(str_field_value.lower(), value_content.lower()) if str_field_value and value_content else 0
            
            # If value is similar or we just want the key location
            if value_similarity >= threshold or str_field_value == "":
                value_polygons = kv_pair.get('value', {}).get('boundingPolygons', [])
                for polygon in value_polygons:
                    matches.append({
                        "points": polygon.get('points', []),
                        "pageNumber": polygon.get('pageNumber', 1),
                        "confidence": kv_pair.get('confidence'),
                        "source": "doc_intelligence_kv",
                        "matchedContent": value_content,
                        "keySimilarity": key_similarity,
                        "valueSimilarity": value_similarity
                    })
    
    return matches


def find_fuzzy_match_polygons(
    value: str,
    words: List[Dict[str, Any]],
    lines: List[Dict[str, Any]],
    threshold: float = DEFAULT_FUZZY_THRESHOLD
) -> List[Dict[str, Any]]:
    """
    Find matching polygon(s) using fuzzy string matching against words and lines.
    
    Args:
        value: The value to search for
        words: List of words with polygons from Document Intelligence
        lines: List of lines with polygons from Document Intelligence
        threshold: Minimum similarity threshold (0-100)
        
    Returns:
        List of matching bounding polygons with metadata
    """
    matches = []
    search_value = str(value).strip().lower()
    
    if not search_value:
        return matches
    
    # First try matching against lines (better for multi-word values)
    if len(search_value.split()) > 1:
        for line in lines:
            line_content = line.get('content', '').lower()
            similarity = fuzz.ratio(search_value, line_content)
            
            if similarity >= threshold:
                matches.append({
                    "points": line.get('points', []),
                    "pageNumber": line.get('pageNumber', 1),
                    "confidence": None,
                    "source": "fuzzy_match_line",
                    "matchedContent": line.get('content', ''),
                    "similarity": similarity
                })
            # Also check if the value is contained within the line
            elif search_value in line_content:
                partial_ratio = fuzz.partial_ratio(search_value, line_content)
                if partial_ratio >= threshold:
                    matches.append({
                        "points": line.get('points', []),
                        "pageNumber": line.get('pageNumber', 1),
                        "confidence": None,
                        "source": "fuzzy_match_line_partial",
                        "matchedContent": line.get('content', ''),
                        "similarity": partial_ratio
                    })
    
    # Also try matching against individual words
    for word in words:
        word_content = word.get('content', '').lower()
        
        # For single-word values, use exact or near-exact matching
        if len(search_value.split()) == 1:
            similarity = fuzz.ratio(search_value, word_content)
            if similarity >= threshold:
                matches.append({
                    "points": word.get('points', []),
                    "pageNumber": word.get('pageNumber', 1),
                    "confidence": word.get('confidence'),
                    "source": "fuzzy_match_word",
                    "matchedContent": word.get('content', ''),
                    "similarity": similarity
                })
        else:
            # For multi-word searches, check if this word is part of the search
            for search_word in search_value.split():
                similarity = fuzz.ratio(search_word, word_content)
                if similarity >= threshold:
                    matches.append({
                        "points": word.get('points', []),
                        "pageNumber": word.get('pageNumber', 1),
                        "confidence": word.get('confidence'),
                        "source": "fuzzy_match_word_partial",
                        "matchedContent": word.get('content', ''),
                        "similarity": similarity
                    })
    
    return matches


def deduplicate_polygons(polygons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate polygons based on points and page number.
    Keeps the match with highest confidence/similarity.
    """
    seen = {}
    for polygon in polygons:
        # Create a key from points and page number
        points_key = tuple(polygon.get('points', []))
        page_key = polygon.get('pageNumber', 1)
        key = (points_key, page_key)
        
        if key not in seen:
            seen[key] = polygon
        else:
            # Keep the one with higher confidence/similarity
            existing_score = seen[key].get('confidence') or seen[key].get('similarity') or 0
            new_score = polygon.get('confidence') or polygon.get('similarity') or 0
            if new_score > existing_score:
                seen[key] = polygon
    
    return list(seen.values())


def correlate_field_with_polygons(
    field_name: str,
    field_value: Any,
    polygon_data: Dict[str, Any],
    threshold: float = DEFAULT_FUZZY_THRESHOLD
) -> List[Dict[str, Any]]:
    """
    Correlate a single field with its bounding polygons using hybrid approach.
    
    Args:
        field_name: Name of the extracted field
        field_value: Value of the extracted field
        polygon_data: Full polygon data from Document Intelligence
        threshold: Fuzzy matching threshold (0-100)
        
    Returns:
        List of bounding polygons where this value appears
    """
    all_matches = []
    
    # Skip None or empty values
    if field_value is None or (isinstance(field_value, str) and not field_value.strip()):
        return all_matches
    
    key_value_pairs = polygon_data.get('keyValuePairs', [])
    words = polygon_data.get('words', [])
    lines = polygon_data.get('lines', [])
    
    # First pass: Try to match using Document Intelligence key-value pairs
    kv_matches = find_key_value_polygon(field_name, field_value, key_value_pairs, threshold)
    all_matches.extend(kv_matches)
    
    # Second pass: Fuzzy match against words/lines if no KV match found
    if not kv_matches:
        fuzzy_matches = find_fuzzy_match_polygons(str(field_value), words, lines, threshold)
        all_matches.extend(fuzzy_matches)
    
    # Deduplicate results
    return deduplicate_polygons(all_matches)


def enrich_extraction_with_polygons(
    extracted_data: Dict[str, Any],
    polygon_data: Dict[str, Any],
    threshold: float = DEFAULT_FUZZY_THRESHOLD
) -> Dict[str, Any]:
    """
    Enrich GPT-extracted data with bounding polygon information.
    
    Args:
        extracted_data: The extracted JSON data from GPT
        polygon_data: Full polygon data from Document Intelligence
        threshold: Fuzzy matching threshold (0-100)
        
    Returns:
        Enriched extraction with boundingPolygons for each field
    """
    logger.info(f"Starting polygon correlation with threshold {threshold}%")
    
    def process_value(key: str, value: Any, parent_path: str = "") -> Dict[str, Any]:
        """Recursively process values to add polygon data."""
        full_key = f"{parent_path}.{key}" if parent_path else key
        
        if isinstance(value, dict):
            # Nested object - process each field
            processed = {}
            for nested_key, nested_value in value.items():
                processed[nested_key] = process_value(nested_key, nested_value, full_key)
            return processed
        elif isinstance(value, list):
            # Array - process each item
            processed_items = []
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    # Array of objects
                    processed_item = {}
                    for nested_key, nested_value in item.items():
                        processed_item[nested_key] = process_value(nested_key, nested_value, f"{full_key}[{i}]")
                    processed_items.append(processed_item)
                else:
                    # Array of primitives
                    polygons = correlate_field_with_polygons(key, item, polygon_data, threshold)
                    processed_items.append({
                        "value": item,
                        "boundingPolygons": [
                            {"points": p.get('points', []), "pageNumber": p.get('pageNumber', 1)}
                            for p in polygons
                        ]
                    })
            return processed_items
        else:
            # Primitive value - find polygons
            polygons = correlate_field_with_polygons(key, value, polygon_data, threshold)
            
            result = {
                "value": value,
                "boundingPolygons": [
                    {"points": p.get('points', []), "pageNumber": p.get('pageNumber', 1)}
                    for p in polygons
                ]
            }
            
            # Add source info if available
            if polygons:
                result["source"] = polygons[0].get('source', 'unknown')
                if polygons[0].get('confidence'):
                    result["confidence"] = polygons[0].get('confidence')
            
            return result
    
    # Process all top-level fields
    enriched = {}
    for key, value in extracted_data.items():
        # Skip error/metadata fields
        if key in ['error', 'error_type', 'extraction_failed', 'raw_content', 'parsing_error']:
            enriched[key] = value
            continue
        
        enriched[key] = process_value(key, value)
    
    # Add metadata about polygon correlation
    total_fields = count_fields(enriched)
    fields_with_polygons = count_fields_with_polygons(enriched)
    
    enriched['_polygonMetadata'] = {
        'totalFields': total_fields,
        'fieldsWithPolygons': fields_with_polygons,
        'correlationThreshold': threshold,
        'sourceDataAvailable': {
            'words': len(polygon_data.get('words', [])),
            'lines': len(polygon_data.get('lines', [])),
            'keyValuePairs': len(polygon_data.get('keyValuePairs', [])),
            'paragraphs': len(polygon_data.get('paragraphs', []))
        }
    }
    
    logger.info(f"Polygon correlation complete: {fields_with_polygons}/{total_fields} fields matched")
    
    return enriched


def count_fields(data: Any, count: int = 0) -> int:
    """Count total number of leaf fields in the data structure."""
    if isinstance(data, dict):
        for key, value in data.items():
            if key.startswith('_'):  # Skip metadata fields
                continue
            if isinstance(value, dict):
                if 'value' in value and 'boundingPolygons' in value:
                    count += 1
                else:
                    count = count_fields(value, count)
            elif isinstance(value, list):
                for item in value:
                    count = count_fields(item, count)
            else:
                count += 1
    return count


def count_fields_with_polygons(data: Any, count: int = 0) -> int:
    """Count fields that have at least one bounding polygon."""
    if isinstance(data, dict):
        for key, value in data.items():
            if key.startswith('_'):  # Skip metadata fields
                continue
            if isinstance(value, dict):
                if 'value' in value and 'boundingPolygons' in value:
                    if value['boundingPolygons']:
                        count += 1
                else:
                    count = count_fields_with_polygons(value, count)
            elif isinstance(value, list):
                for item in value:
                    count = count_fields_with_polygons(item, count)
    return count
