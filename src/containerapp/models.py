"""
Data models for the ARGUS Container App
"""
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field


# ============================================================================
# Bounding Polygon Models for Document Intelligence Integration
# ============================================================================

class BoundingPolygon(BaseModel):
    """
    Represents a bounding polygon for a text element on a document page.
    Points are stored as a flat list: [x1, y1, x2, y2, x3, y3, x4, y4]
    representing the four corners of the bounding quadrilateral.
    """
    points: List[float] = Field(default_factory=list, description="Polygon points as [x1, y1, x2, y2, ...]")
    pageNumber: int = Field(default=1, description="1-indexed page number where the element appears")


class ExtractedFieldWithLocation(BaseModel):
    """
    Represents an extracted field value with optional bounding polygon information.
    Used when include_polygons=True in extraction requests.
    """
    value: Any = Field(description="The extracted field value")
    boundingPolygons: List[BoundingPolygon] = Field(
        default_factory=list, 
        description="List of bounding polygons where this value appears (can be multiple for repeated values)"
    )
    confidence: Optional[float] = Field(default=None, description="Confidence score from Document Intelligence")
    source: str = Field(default="gpt", description="Source of the value: 'doc_intelligence_kv', 'fuzzy_match', or 'gpt'")


class OCRPolygonData(BaseModel):
    """
    Stores raw polygon data extracted from Document Intelligence for correlation.
    """
    content: str = Field(default="", description="Full OCR text content")
    words: List[Dict[str, Any]] = Field(default_factory=list, description="Words with polygons")
    lines: List[Dict[str, Any]] = Field(default_factory=list, description="Lines with polygons")
    keyValuePairs: List[Dict[str, Any]] = Field(default_factory=list, description="Key-value pairs with polygons")
    paragraphs: List[Dict[str, Any]] = Field(default_factory=list, description="Paragraphs with polygons")


class ExtractionRequest(BaseModel):
    """
    Request model for document extraction with optional polygon support.
    """
    include_polygons: bool = Field(default=False, description="Whether to include bounding polygons in output")
    fuzzy_match_threshold: float = Field(default=0.90, description="Threshold for fuzzy string matching (0.0-1.0)")


# ============================================================================
# Event Grid and Blob Models
# ============================================================================

class EventGridEvent:
    """Event Grid event model"""
    def __init__(self, event_data: Dict[str, Any]):
        self.id = event_data.get('id')
        self.event_type = event_data.get('eventType')
        self.subject = event_data.get('subject')
        self.event_time = event_data.get('eventTime')
        self.data = event_data.get('data', {})
        self.data_version = event_data.get('dataVersion')
        self.metadata_version = event_data.get('metadataVersion')


class BlobInputStream:
    """Mock BlobInputStream to match the original function interface"""
    def __init__(self, blob_name: str, blob_size: int, blob_client):
        self.name = blob_name
        self.length = blob_size
        self._blob_client = blob_client
        self._content = None
    
    def read(self, size: int = -1):
        """Read blob content"""
        if self._content is None:
            blob_data = self._blob_client.download_blob()
            self._content = blob_data.readall()
        
        if size == -1:
            return self._content
        else:
            return self._content[:size]
