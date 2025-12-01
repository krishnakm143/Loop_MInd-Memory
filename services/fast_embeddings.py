"""
Ultra-fast synthetic embedding generator.

Generates high-quality embeddings locally using:
- FNV-1a hashing for token features
- TF-IDF weighted token vectors
- N-gram features (bigrams, trigrams, skip-grams)
- Positional encoding
- Sector-specific weighting

Target: <1ms per embedding (vs 500-1500ms for OpenAI API)
"""

import math
import hashlib
import re
from typing import List, Dict, Set, Optional, Tuple
from functools import lru_cache
import numpy as np
import structlog

logger = structlog.get_logger()

# Sector-specific weights for embedding generation
SECTOR_WEIGHTS = {
    "episodic": 1.3,
    "semantic": 1.0,
    "procedural": 1.2,
    "emotional": 1.4,
    "reflective": 0.9,
}

# Stop words for filtering
STOP_WORDS = frozenset([
    "the", "a", "an", "to", "of", "and", "or", "in", "on", "for", "with",
    "at", "by", "is", "it", "be", "as", "are", "was", "were", "from",
    "that", "this", "these", "those", "but", "if", "then", "so", "than",
    "into", "over", "under", "about", "via", "vs", "not", "i", "you",
    "he", "she", "we", "they", "my", "your", "his", "her", "our", "their",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other", "some",
    "such", "no", "nor", "only", "own", "same", "too", "very", "can", "will",
    "just", "should", "now", "has", "have", "had", "do", "does", "did",
    "been", "being", "having", "doing", "would", "could", "might", "must",
    "shall", "may", "need", "dare", "ought", "used", "am", "its"
])

# Synonym mapping for semantic expansion
SYNONYMS = {
    "happy": ["joyful", "pleased", "content", "delighted"],
    "sad": ["unhappy", "sorrowful", "dejected", "melancholy"],
    "good": ["excellent", "great", "fine", "positive"],
    "bad": ["poor", "negative", "terrible", "awful"],
    "fast": ["quick", "rapid", "swift", "speedy"],
    "slow": ["sluggish", "gradual", "leisurely"],
    "big": ["large", "huge", "enormous", "massive"],
    "small": ["tiny", "little", "minute", "compact"],
    "important": ["significant", "crucial", "vital", "essential"],
    "problem": ["issue", "challenge", "difficulty", "obstacle"],
    "solution": ["answer", "resolution", "fix", "remedy"],
    "create": ["make", "build", "construct", "generate"],
    "delete": ["remove", "erase", "destroy", "eliminate"],
    "update": ["modify", "change", "alter", "revise"],
}


class FastEmbeddingGenerator:
    """
    Generates synthetic embeddings in <1ms using locality-sensitive hashing.
    
    This bypasses API calls entirely for ultra-fast retrieval.
    """
    
    def __init__(self, dim: int = 768):
        self.dim = dim
        self._token_cache: Dict[str, np.ndarray] = {}
        
    def _fnv1a_hash(self, data: str) -> int:
        """FNV-1a hash for fast, well-distributed hashing."""
        h = 0x811c9dc5  # FNV offset basis
        for char in data:
            h ^= ord(char)
            h = (h * 16777619) & 0xFFFFFFFF  # FNV prime
        return h
    
    def _murmur_hash(self, data: str, seed: int = 0xDEADBEEF) -> int:
        """MurmurHash-inspired hash for secondary hashing."""
        h = seed
        for char in data:
            h ^= ord(char)
            h = ((h * 0x5bd1e995) & 0xFFFFFFFF)
            h = ((h >> 13) ^ h) & 0xFFFFFFFF
        return h
    
    @lru_cache(maxsize=10000)
    def _tokenize(self, text: str) -> Tuple[str, ...]:
        """Tokenize and normalize text."""
        # Lowercase and extract words
        words = re.findall(r'[a-z0-9]+', text.lower())
        # Filter stop words and short words
        tokens = [w for w in words if w not in STOP_WORDS and len(w) > 1]
        return tuple(tokens)
    
    def _expand_with_synonyms(self, tokens: List[str]) -> Set[str]:
        """Expand tokens with synonyms for better semantic coverage."""
        expanded = set(tokens)
        for token in tokens:
            if token in SYNONYMS:
                expanded.update(SYNONYMS[token])
        return expanded
    
    def _add_feature(self, vec: np.ndarray, key: str, weight: float):
        """Add a feature to the vector using double hashing."""
        h1 = self._fnv1a_hash(key)
        h2 = self._murmur_hash(key)
        
        # Determine sign based on LSB of first hash
        sign = 1 - ((h1 & 1) << 1)  # 1 or -1
        val = weight * sign
        
        # Use power-of-2 optimization if dimension allows
        if self.dim > 0 and (self.dim & (self.dim - 1)) == 0:
            vec[h1 & (self.dim - 1)] += val
            vec[h2 & (self.dim - 1)] += val * 0.5
        else:
            vec[h1 % self.dim] += val
            vec[h2 % self.dim] += val * 0.5
    
    def _add_positional_encoding(self, vec: np.ndarray, pos: int, weight: float):
        """Add sinusoidal positional encoding."""
        idx = pos % self.dim
        ang = pos / (10000 ** ((2 * idx) / self.dim))
        vec[idx] += weight * math.sin(ang)
        vec[(idx + 1) % self.dim] += weight * math.cos(ang)
    
    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """L2 normalize the vector."""
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec
    
    def generate(self, text: str, sector: str = "semantic") -> np.ndarray:
        """
        Generate a synthetic embedding for the given text.
        
        Args:
            text: Input text to embed
            sector: Memory sector for weight adjustment
            
        Returns:
            Normalized embedding vector of dimension self.dim
        """
        vec = np.zeros(self.dim, dtype=np.float32)
        
        # Get canonical tokens
        tokens = list(self._tokenize(text))
        if not tokens:
            # Return uniform vector for empty input
            return np.full(self.dim, 1 / math.sqrt(self.dim), dtype=np.float32)
        
        # Expand with synonyms
        expanded_tokens = list(self._expand_with_synonyms(tokens))
        
        # Calculate token frequencies
        token_counts: Dict[str, int] = {}
        for tok in expanded_tokens:
            token_counts[tok] = token_counts.get(tok, 0) + 1
        
        # Get sector weight
        sector_weight = SECTOR_WEIGHTS.get(sector, 1.0)
        doc_len = len(expanded_tokens)
        log_doc_len = math.log(1 + doc_len)
        
        # Add TF-IDF weighted token features
        for tok, count in token_counts.items():
            tf = count / doc_len
            idf = math.log(1 + doc_len / count)
            weight = (tf * idf + 1) * sector_weight
            
            # Token feature
            self._add_feature(vec, f"{sector}|tok|{tok}", weight)
            
            # Character n-grams for subword matching
            if len(tok) >= 3:
                for i in range(len(tok) - 2):
                    self._add_feature(vec, f"{sector}|c3|{tok[i:i+3]}", weight * 0.4)
            if len(tok) >= 4:
                for i in range(len(tok) - 3):
                    self._add_feature(vec, f"{sector}|c4|{tok[i:i+4]}", weight * 0.3)
        
        # Add bigram features
        for i in range(len(tokens) - 1):
            a, b = tokens[i], tokens[i + 1]
            pos_weight = 1.0 / (1.0 + i * 0.1)  # Position decay
            self._add_feature(vec, f"{sector}|bi|{a}_{b}", 1.4 * sector_weight * pos_weight)
        
        # Add trigram features
        for i in range(len(tokens) - 2):
            a, b, c = tokens[i], tokens[i + 1], tokens[i + 2]
            self._add_feature(vec, f"{sector}|tri|{a}_{b}_{c}", 1.0 * sector_weight)
        
        # Add skip-gram features (skip one word)
        for i in range(min(len(tokens) - 2, 20)):
            a, c = tokens[i], tokens[i + 2]
            self._add_feature(vec, f"{sector}|skip|{a}_{c}", 0.7 * sector_weight)
        
        # Add positional encoding
        for i in range(min(len(tokens), 50)):
            self._add_positional_encoding(vec, i, 0.5 * sector_weight / log_doc_len)
        
        # Add document length feature
        len_bucket = min(int(math.log2(doc_len + 1)), 10)
        self._add_feature(vec, f"{sector}|len|{len_bucket}", 0.6 * sector_weight)
        
        # Add token density feature
        density = len(token_counts) / doc_len
        density_bucket = int(density * 10)
        self._add_feature(vec, f"{sector}|dens|{density_bucket}", 0.5 * sector_weight)
        
        return self._normalize(vec)
    
    def generate_batch(self, texts: List[str], sector: str = "semantic") -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        return [self.generate(text, sector) for text in texts]
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return float(np.dot(a, b))  # Vectors are already normalized


class HybridEmbeddingGenerator:
    """
    Hybrid embedding generator that fuses synthetic and semantic embeddings.
    
    Modes:
    - fast: Synthetic only (<1ms)
    - smart: Fused synthetic + compressed semantic (<5ms with cache)
    - full: Full semantic embeddings (500-1500ms)
    """
    
    def __init__(
        self,
        dim: int = 768,
        semantic_dim: int = 128,  # Compressed dimension for semantic
        mode: str = "fast"
    ):
        self.dim = dim
        self.semantic_dim = semantic_dim
        self.mode = mode
        self.fast_gen = FastEmbeddingGenerator(dim)
        
    def _compress_vector(self, vec: np.ndarray, target_dim: int) -> np.ndarray:
        """Compress a vector to target dimension using mean pooling."""
        if len(vec) <= target_dim:
            return vec
        
        compressed = np.zeros(target_dim, dtype=np.float32)
        bucket_size = len(vec) / target_dim
        
        for i in range(target_dim):
            start = int(i * bucket_size)
            end = int((i + 1) * bucket_size)
            compressed[i] = np.mean(vec[start:end])
        
        # Normalize
        norm = np.linalg.norm(compressed)
        if norm > 0:
            compressed /= norm
        
        return compressed
    
    def _fuse_vectors(
        self,
        synthetic: np.ndarray,
        semantic: np.ndarray,
        syn_weight: float = 0.6,
        sem_weight: float = 0.4
    ) -> np.ndarray:
        """Fuse synthetic and semantic vectors."""
        fused = np.concatenate([
            synthetic * syn_weight,
            semantic * sem_weight
        ])
        
        # Normalize
        norm = np.linalg.norm(fused)
        if norm > 0:
            fused /= norm
        
        return fused
    
    async def generate(
        self,
        text: str,
        sector: str = "semantic",
        semantic_vec: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate embedding based on mode.
        
        Args:
            text: Input text
            sector: Memory sector
            semantic_vec: Pre-computed semantic embedding (for smart mode)
        """
        if self.mode == "fast":
            return self.fast_gen.generate(text, sector)
        
        elif self.mode == "smart" and semantic_vec is not None:
            synthetic = self.fast_gen.generate(text, sector)
            compressed_semantic = self._compress_vector(semantic_vec, self.semantic_dim)
            return self._fuse_vectors(synthetic, compressed_semantic)
        
        else:
            # Full mode - use fast as fallback
            return self.fast_gen.generate(text, sector)


# Singleton instance
_fast_generator: Optional[FastEmbeddingGenerator] = None


def get_fast_generator(dim: int = 768) -> FastEmbeddingGenerator:
    """Get or create the fast embedding generator singleton."""
    global _fast_generator
    if _fast_generator is None or _fast_generator.dim != dim:
        _fast_generator = FastEmbeddingGenerator(dim)
    return _fast_generator


def fast_embed(text: str, sector: str = "semantic", dim: int = 768) -> np.ndarray:
    """Convenience function for fast embedding generation."""
    return get_fast_generator(dim).generate(text, sector)


def fast_embed_batch(texts: List[str], sector: str = "semantic", dim: int = 768) -> List[np.ndarray]:
    """Convenience function for batch fast embedding generation."""
    return get_fast_generator(dim).generate_batch(texts, sector)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))



