"""Text preprocessing service for memory ingestion."""

import re
import unicodedata
from typing import Dict, Any, List, Optional
from collections import Counter
import structlog

logger = structlog.get_logger()

# PII detection patterns
EMAIL_PATTERN = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
)
PHONE_PATTERN = re.compile(
    r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
)
SSN_PATTERN = re.compile(
    r'\b\d{3}-\d{2}-\d{4}\b|\b\d{3}\s\d{2}\s\d{4}\b'
)
CREDIT_CARD_PATTERN = re.compile(
    r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
)
IP_ADDRESS_PATTERN = re.compile(
    r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
)
URL_PATTERN = re.compile(
    r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?'
)


class Preprocessor:
    """Text preprocessing service for memory ingestion."""

    def __init__(self):
        """Initialize preprocessor."""
        self.pii_patterns = {
            "email": EMAIL_PATTERN,
            "phone": PHONE_PATTERN,
            "ssn": SSN_PATTERN,
            "credit_card": CREDIT_CARD_PATTERN,
            "ip_address": IP_ADDRESS_PATTERN,
            "url": URL_PATTERN,
        }

    def normalize_text(self, text: str) -> str:
        """Normalize text: Unicode, whitespace, encoding.
        
        Args:
            text: Raw text input
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Unicode normalization (NFKC: compatibility decomposition + composition)
        text = unicodedata.normalize("NFKC", text)
        
        # Remove zero-width characters
        text = re.sub(r'[\u200b-\u200f\u202a-\u202e\ufeff]', '', text)
        
        # Normalize whitespace (multiple spaces/tabs/newlines to single space)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text

    def detect_language(self, text: str) -> str:
        """Detect language of text (simple heuristic).
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code (default: 'en')
        """
        # Simple heuristic: check for common patterns
        # For production, use langdetect or polyglot
        if not text:
            return "en"
        
        # Check for common non-English characters
        if re.search(r'[\u4e00-\u9fff]', text):  # Chinese
            return "zh"
        if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):  # Japanese
            return "ja"
        if re.search(r'[\u0400-\u04ff]', text):  # Cyrillic
            return "ru"
        if re.search(r'[\u0600-\u06ff]', text):  # Arabic
            return "ar"
        
        # Default to English
        return "en"

    def redact_pii(self, text: str) -> tuple[str, Dict[str, int], bool]:
        """Redact PII from text using pattern matching.
        
        Args:
            text: Text to redact
            
        Returns:
            Tuple of (redacted_text, pii_counts, was_redacted)
        """
        redacted_text = text
        pii_counts: Dict[str, int] = {}
        was_redacted = False
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = pattern.findall(redacted_text)
            if matches:
                count = len(matches)
                pii_counts[pii_type] = count
                was_redacted = True
                
                # Replace with placeholder
                placeholder = f"[{pii_type.upper()}_REDACTED]"
                redacted_text = pattern.sub(placeholder, redacted_text)
                
                logger.debug(
                    "PII redacted",
                    pii_type=pii_type,
                    count=count,
                    text_preview=text[:100]
                )
        
        return redacted_text, pii_counts, was_redacted

    def chunk_text(
        self,
        text: str,
        max_chunk_size: int = 1000,
        overlap: int = 200,
        min_chunk_size: int = 100
    ) -> List[Dict[str, Any]]:
        """Split text into semantic chunks.
        
        Args:
            text: Text to chunk
            max_chunk_size: Maximum characters per chunk
            overlap: Overlap between chunks in characters
            min_chunk_size: Minimum characters per chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text:
            return []
        
        # If text is shorter than max_chunk_size, return as single chunk
        # (even if below min_chunk_size, we still return it)
        if len(text) <= max_chunk_size:
            return [{"text": text, "start": 0, "end": len(text), "index": 0}]
        
        chunks: List[Dict[str, Any]] = []
        
        # Try to split on sentence boundaries first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for sentence in sentences:
            # Calculate what the chunk would be if we add this sentence
            potential_chunk = current_chunk + (" " + sentence if current_chunk else sentence)
            
            # If adding this sentence would exceed max_chunk_size and we have content
            if len(potential_chunk) > max_chunk_size and current_chunk:
                # Save current chunk if it meets minimum size
                if len(current_chunk.strip()) >= min_chunk_size:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "start": current_start,
                        "end": current_start + len(current_chunk.strip()),
                        "index": chunk_index
                    })
                    chunk_index += 1
                
                # Start new chunk with overlap from previous chunk
                overlap_size = min(overlap, len(current_chunk))
                overlap_text = current_chunk[-overlap_size:] if overlap_size > 0 else ""
                
                # Calculate new start position (end of previous chunk minus overlap)
                if chunks:
                    current_start = chunks[-1]["end"] - len(overlap_text)
                else:
                    current_start = 0
                
                # Start new chunk with overlap
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                    # Update start position for new chunk
                    if chunks:
                        current_start = chunks[-1]["end"]
        
        # Add final chunk if it meets minimum size
        if current_chunk and len(current_chunk.strip()) >= min_chunk_size:
            chunks.append({
                "text": current_chunk.strip(),
                "start": current_start,
                "end": current_start + len(current_chunk.strip()),
                "index": chunk_index
            })
        
        # Fallback: if no chunks created, use simple character-based splitting
        # If still no chunks and text exists, return at least one chunk (even if below min size)
        if not chunks:
            if text.strip():
                # Return the entire text as a single chunk if it's too short
                chunks.append({
                    "text": text.strip(),
                    "start": 0,
                    "end": len(text),
                    "index": 0
                })
            else:
                # Try character-based splitting as last resort
                for i in range(0, len(text), max_chunk_size - overlap):
                    chunk_text = text[i:i + max_chunk_size]
                    if chunk_text.strip():
                        chunks.append({
                            "text": chunk_text.strip(),
                            "start": i,
                            "end": i + len(chunk_text),
                            "index": len(chunks)
                        })
        
        return chunks

    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Metadata dictionary
        """
        metadata: Dict[str, Any] = {
            "char_count": len(text),
            "word_count": len(text.split()) if text else 0,
            "has_code": bool(re.search(r'```|`[^`]+`|def |class |import |from ', text)),
            "has_urls": bool(URL_PATTERN.search(text)),
            "has_numbers": bool(re.search(r'\d', text)),
        }
        
        # Detect content type
        if metadata["has_code"]:
            metadata["content_type"] = "code"
        elif metadata["has_urls"]:
            metadata["content_type"] = "web_content"
        else:
            metadata["content_type"] = "text"
        
        return metadata

    def extract_tags_from_content(self, text: str, max_tags: int = 10) -> List[str]:
        """Extract tags/keywords from content using simple NLP.
        
        Args:
            text: Text content to extract tags from
            max_tags: Maximum number of tags to return
            
        Returns:
            List of extracted tags (normalized, lowercase)
        """
        if not text:
            return []
        
        # Common stop words to filter out
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its',
            'our', 'their', 'what', 'which', 'who', 'whom', 'whose', 'where',
            'when', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'just', 'now', 'then', 'here',
            'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
            'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will',
            'just', 'don', 'should', 'now'
        }
        
        # Normalize text
        normalized = text.lower()
        
        # Remove URLs, emails, and special patterns
        normalized = re.sub(r'https?://\S+|www\.\S+', '', normalized)
        normalized = re.sub(r'\S+@\S+', '', normalized)
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        
        # Extract words (2+ characters, alphanumeric)
        words = re.findall(r'\b[a-z]{2,}\b', normalized)
        
        # Filter stop words and count frequency
        meaningful_words = [w for w in words if w not in stop_words and len(w) >= 2]
        
        if not meaningful_words:
            return []
        
        # Count word frequency
        word_counts = Counter(meaningful_words)
        
        # Get top N most frequent words as tags
        top_words = word_counts.most_common(max_tags)
        
        # Extract tags (normalize: lowercase, remove duplicates)
        tags = [word for word, count in top_words if count >= 1]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tags = []
        for tag in tags:
            if tag not in seen:
                seen.add(tag)
                unique_tags.append(tag)
        
        logger.debug(
            "Tags extracted from content",
            tag_count=len(unique_tags),
            tags=unique_tags[:5]  # Log first 5
        )
        
        return unique_tags[:max_tags]


# Global preprocessor instance
_preprocessor = Preprocessor()


async def preprocess_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    enable_pii_redaction: bool = True
) -> Dict[str, Any]:
    """Preprocess and normalize text.
    
    Args:
        text: Raw text input
        chunk_size: Maximum chunk size for text splitting
        chunk_overlap: Overlap between chunks
        enable_pii_redaction: Whether to redact PII
        
    Returns:
        Preprocessed text with metadata including:
        - normalized_text: Cleaned and normalized text
        - language: Detected language code
        - chunks: List of text chunks
        - pii_redacted: Whether PII was found and redacted
        - pii_counts: Count of each PII type found
        - metadata: Additional metadata about the text
    """
    if not text:
        return {
            "normalized_text": "",
            "language": "en",
            "chunks": [],
            "pii_redacted": False,
            "pii_counts": {},
            "metadata": {},
            "extracted_tags": []
        }
    
    # Step 1: Normalize text
    normalized_text = _preprocessor.normalize_text(text)
    
    # Step 2: Detect language
    language = _preprocessor.detect_language(normalized_text)
    
    # Step 3: Redact PII (if enabled)
    if enable_pii_redaction:
        redacted_text, pii_counts, pii_redacted = _preprocessor.redact_pii(normalized_text)
    else:
        redacted_text = normalized_text
        pii_counts = {}
        pii_redacted = False
    
    # Step 4: Chunk text
    chunks = _preprocessor.chunk_text(
        redacted_text,
        max_chunk_size=chunk_size,
        overlap=chunk_overlap
    )
    
    # Step 5: Extract metadata
    metadata = _preprocessor.extract_metadata(redacted_text)
    metadata["language"] = language
    
    # Step 6: Extract tags from content
    extracted_tags = _preprocessor.extract_tags_from_content(redacted_text)
    
    logger.info(
        "Text preprocessed",
        char_count=len(normalized_text),
        chunk_count=len(chunks),
        pii_redacted=pii_redacted,
        language=language,
        extracted_tags_count=len(extracted_tags)
    )
    
    return {
        "normalized_text": redacted_text,
        "language": language,
        "chunks": chunks,
        "pii_redacted": pii_redacted,
        "pii_counts": pii_counts,
        "metadata": metadata,
        "extracted_tags": extracted_tags,
    }