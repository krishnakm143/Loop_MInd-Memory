"""ANTIVIRAL detection service for prompt injection and poisoning."""

import re
from typing import Dict, Any, List
from collections import Counter
import structlog

logger = structlog.get_logger()

# Threat score thresholds
QUARANTINE_THRESHOLD = 0.85
LOW_TRUST_THRESHOLD = 0.5

# Deterministic rule patterns for prompt injection
INJECTION_PATTERNS = [
    # Direct instruction overrides
    re.compile(r'ignore\s+(previous|all|above)\s+(instructions?|prompts?|rules?)', re.IGNORECASE),
    re.compile(r'forget\s+(previous|all|everything)', re.IGNORECASE),
    re.compile(r'disregard\s+(previous|all|above)', re.IGNORECASE),
    re.compile(r'override\s+(previous|system|all)', re.IGNORECASE),
    
    # System role manipulation
    re.compile(r'you\s+are\s+(now|a|an)\s+', re.IGNORECASE),
    re.compile(r'act\s+as\s+(if\s+you\s+are|a|an)', re.IGNORECASE),
    re.compile(r'pretend\s+(you\s+are|to\s+be)', re.IGNORECASE),
    
    # Instruction injection
    re.compile(r'new\s+(instructions?|prompts?|rules?):', re.IGNORECASE),
    re.compile(r'following\s+(instructions?|prompts?|rules?):', re.IGNORECASE),
    re.compile(r'execute\s+(the\s+following|these)', re.IGNORECASE),
    
    # Token manipulation
    re.compile(r'<\|.*?\|>', re.IGNORECASE),  # Special tokens
    re.compile(r'\[INST\]|\[/INST\]', re.IGNORECASE),  # Instruction tags
    
    # Base64/encoding attempts
    re.compile(r'[A-Za-z0-9+/]{50,}={0,2}'),  # Long base64-like strings
    
    # Suspicious commands
    re.compile(r'(rm\s+-rf|del\s+/f|format\s+c:)', re.IGNORECASE),
    re.compile(r'(curl|wget)\s+.*?(http|https)', re.IGNORECASE),
    re.compile(r'eval\s*\(|exec\s*\(|system\s*\(', re.IGNORECASE),
]

# Exfiltration patterns
EXFIL_PATTERNS = [
    re.compile(r'send\s+(to|data|information)\s+.*?(email|http|api)', re.IGNORECASE),
    re.compile(r'post\s+to\s+.*?(http|api|endpoint)', re.IGNORECASE),
    re.compile(r'upload\s+to\s+.*?(http|s3|gcs)', re.IGNORECASE),
    re.compile(r'export\s+.*?(password|token|key|secret)', re.IGNORECASE),
]

# Suspicious file type patterns
SUSPICIOUS_FILE_PATTERNS = [
    re.compile(r'\.(exe|bat|cmd|ps1|sh|py|js)\s*$', re.IGNORECASE),
    re.compile(r'<script|</script>', re.IGNORECASE),
    re.compile(r'javascript:', re.IGNORECASE),
    re.compile(r'on\w+\s*=', re.IGNORECASE),  # Event handlers
]


class AntiviralScanner:
    """ANTIVIRAL scanner for prompt injection and poisoning detection."""
    
    def __init__(self):
        """Initialize ANTIVIRAL scanner."""
        self.injection_patterns = INJECTION_PATTERNS
        self.exfil_patterns = EXFIL_PATTERNS
        self.suspicious_file_patterns = SUSPICIOUS_FILE_PATTERNS
    
    def check_deterministic_rules(self, content: str) -> tuple[float, List[str]]:
        """Check content against deterministic rule patterns.
        
        Args:
            content: Content to scan
            
        Returns:
            Tuple of (threat_score, flags)
        """
        threat_score = 0.0
        flags: List[str] = []
        
        # Check injection patterns
        for pattern in self.injection_patterns:
            matches = pattern.findall(content)
            if matches:
                threat_score += 0.3
                flags.append(f"injection_pattern: {pattern.pattern[:50]}")
        
        # Check exfiltration patterns
        for pattern in self.exfil_patterns:
            matches = pattern.findall(content)
            if matches:
                threat_score += 0.25
                flags.append(f"exfil_pattern: {pattern.pattern[:50]}")
        
        # Check suspicious file patterns
        for pattern in self.suspicious_file_patterns:
            matches = pattern.findall(content)
            if matches:
                threat_score += 0.2
                flags.append(f"suspicious_file: {pattern.pattern[:50]}")
        
        # Cap deterministic score at 0.6
        threat_score = min(threat_score, 0.6)
        
        return threat_score, flags
    
    def check_heuristics(self, content: str) -> tuple[float, List[str]]:
        """Check content using heuristic detectors.
        
        Args:
            content: Content to scan
            
        Returns:
            Tuple of (threat_score, flags)
        """
        threat_score = 0.0
        flags: List[str] = []
        
        if not content:
            return 0.0, []
        
        # Check for unusual repetition
        words = content.lower().split()
        if len(words) > 10:
            word_counts = Counter(words)
            most_common_count = word_counts.most_common(1)[0][1] if word_counts else 0
            repetition_ratio = most_common_count / len(words)
            
            if repetition_ratio > 0.3:  # More than 30% repetition
                threat_score += 0.15
                flags.append("high_repetition")
        
        # Check for code-like sequences
        code_indicators = [
            r'def\s+\w+\s*\(',  # Function definitions
            r'import\s+\w+',  # Imports
            r'class\s+\w+',  # Class definitions
            r'<\?php|<\?=',  # PHP tags
            r'SELECT\s+.*\s+FROM',  # SQL
            r'\.(get|post|put|delete)\s*\(',  # HTTP methods
        ]
        
        code_score = 0.0
        for pattern_str in code_indicators:
            if re.search(pattern_str, content, re.IGNORECASE):
                code_score += 0.1
        
        if code_score > 0.2:  # Multiple code indicators
            threat_score += 0.2
            flags.append("code_like_sequence")
        
        # Check for unusual token distributions
        # High ratio of special characters
        special_char_ratio = len(re.findall(r'[^\w\s]', content)) / len(content) if content else 0
        if special_char_ratio > 0.3:  # More than 30% special chars
            threat_score += 0.1
            flags.append("unusual_special_chars")
        
        # Check for very long words (potential obfuscation)
        words = content.split()
        long_words = [w for w in words if len(w) > 50]
        if len(long_words) > 0:
            threat_score += 0.15
            flags.append("obfuscated_content")
        
        # Check for unusual whitespace patterns
        if re.search(r'\s{5,}', content):  # 5+ consecutive spaces
            threat_score += 0.05
            flags.append("unusual_whitespace")
        
        # Cap heuristic score at 0.4
        threat_score = min(threat_score, 0.4)
        
        return threat_score, flags
    
    def check_provenance(self, provenance: List[Dict[str, Any]] | None = None) -> tuple[float, List[str]]:
        """Check provenance for suspicious patterns.
        
        Args:
            provenance: Provenance chain
            
        Returns:
            Tuple of (threat_score, flags)
        """
        threat_score = 0.0
        flags: List[str] = []
        
        if not provenance or len(provenance) == 0:
            threat_score += 0.1
            flags.append("no_provenance")
        elif len(provenance) == 1:
            # Single source with no transformations
            source = provenance[0]
            if source.get("source") not in ["api", "user"]:
                threat_score += 0.05
                flags.append("unusual_source")
        
        return threat_score, flags
    
    def model_based_classifier(self, content: str) -> float:
        """Model-based classifier (stub for now).
        
        In production, this would use a trained transformer model.
        For now, returns a simple heuristic-based score.
        
        Args:
            content: Content to classify
            
        Returns:
            Threat score from model [0, 1]
        """
        # TODO: Implement actual transformer-based classifier
        # For now, use a simple heuristic that combines length and suspicious patterns
        
        if not content:
            return 0.0
        
        # Simple heuristic: longer content with suspicious patterns = higher score
        base_score = 0.0
        
        # Check for suspicious keyword density
        suspicious_keywords = [
            'ignore', 'override', 'system', 'admin', 'root',
            'execute', 'eval', 'exec', 'bypass', 'hack'
        ]
        
        content_lower = content.lower()
        keyword_count = sum(1 for kw in suspicious_keywords if kw in content_lower)
        keyword_density = keyword_count / len(content.split()) if content.split() else 0
        
        if keyword_density > 0.05:  # More than 5% suspicious keywords
            base_score = 0.3
        
        return min(base_score, 0.3)  # Cap at 0.3 for model component
    
    def determine_action(self, threat_score: float) -> str:
        """Determine action based on threat score.
        
        Args:
            threat_score: Calculated threat score [0, 1]
            
        Returns:
            Action: "quarantine", "low_trust", or "pass"
        """
        if threat_score > QUARANTINE_THRESHOLD:
            return "quarantine"
        elif threat_score > LOW_TRUST_THRESHOLD:
            return "low_trust"
        else:
            return "pass"


# Global scanner instance
_scanner = AntiviralScanner()


async def scan_content(
    content: str,
    provenance: List[Dict[str, Any]] | None = None
) -> Dict[str, Any]:
    """Scan content for prompt injection and poisoning.
    
    Multi-stage approach:
    1. Deterministic rule-based detection
    2. Heuristic detectors
    3. Model-based classifier (stub)
    4. Provenance analysis
    
    Args:
        content: Content to scan
        provenance: Optional provenance chain for analysis
        
    Returns:
        Scan results with:
        - threat_score: Score from 0.0 to 1.0
        - is_safe: Boolean indicating if content is safe
        - flags: List of detected issues
        - action: Recommended action (pass, low_trust, quarantine)
        - breakdown: Detailed score breakdown
    """
    if not content:
        return {
            "threat_score": 0.0,
            "is_safe": True,
            "flags": [],
            "action": "pass",
            "breakdown": {},
        }
    
    # Step 1: Deterministic rules
    rule_score, rule_flags = _scanner.check_deterministic_rules(content)
    
    # Step 2: Heuristic detectors
    heuristic_score, heuristic_flags = _scanner.check_heuristics(content)
    
    # Step 3: Model-based classifier
    model_score = _scanner.model_based_classifier(content)
    
    # Step 4: Provenance analysis
    provenance_score, provenance_flags = _scanner.check_provenance(provenance)
    
    # Combine scores (weighted)
    threat_score = (
        rule_score * 0.4 +      # Deterministic rules are most important
        heuristic_score * 0.3 +  # Heuristics provide context
        model_score * 0.2 +      # Model provides ML-based detection
        provenance_score * 0.1   # Provenance is supporting evidence
    )
    
    # Cap at 1.0
    threat_score = min(threat_score, 1.0)
    
    # Combine all flags
    all_flags = rule_flags + heuristic_flags + provenance_flags
    
    # Determine action
    action = _scanner.determine_action(threat_score)
    is_safe = action == "pass"
    
    logger.info(
        "ANTIVIRAL scan completed",
        threat_score=threat_score,
        action=action,
        flag_count=len(all_flags),
        content_length=len(content)
    )
    
    if not is_safe:
        logger.warning(
            "ANTIVIRAL detected threat",
            threat_score=threat_score,
            action=action,
            flags=all_flags[:5]  # Log first 5 flags
        )
    
    return {
        "threat_score": round(threat_score, 3),
        "is_safe": is_safe,
        "flags": all_flags,
        "action": action,
        "breakdown": {
            "deterministic": round(rule_score, 3),
            "heuristic": round(heuristic_score, 3),
            "model": round(model_score, 3),
            "provenance": round(provenance_score, 3),
        },
    }
