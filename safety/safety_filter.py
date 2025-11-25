"""
Safety Filter - Content filtering and security measures

Demonstrates safety and security features with:
- Content safety checking
- Malicious input detection
- Privacy protection
- Rate limiting and abuse prevention
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import re
import hashlib

@dataclass
class SafetyCheckResult:
    """Result of safety check"""
    is_safe: bool
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    reason: str
    detected_patterns: List[str]
    confidence: float
    recommendations: List[str]

@dataclass
class SecurityEvent:
    """Security event for monitoring"""
    timestamp: datetime
    user_id: str
    event_type: str
    severity: str
    details: Dict[str, Any]

class SafetyFilter:
    """
    Comprehensive safety and security filter for content processing.
    
    Features:
    1. Malicious content detection
    2. Privacy protection and PII detection
    3. Injection attack prevention
    4. Rate limiting and abuse detection
    5. Content appropriateness filtering
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.security_events = []
        self.user_request_counts = {}
        
        # Malicious patterns
        self.malicious_patterns = {
            'sql_injection': [
                r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b.*\b(FROM|INTO|TABLE|DATABASE)\b)",
                r"(\b(UNION|OR|AND)\s+\d+\s*=\s*\d+)",
                r"(\b('\s*OR\s*'.*'=.*'|&quot;.*&quot;\s*OR\s*&quot;.*&quot;=.*&quot;))",
                r"(--|#|\/\*|\*\/)"  # SQL comments
            ],
            'xss': [
                r"(<script[^>]*>.*?</script>)",
                r"(javascript\s*:)",
                r"(on\w+\s*=)",
                r"(<iframe[^>]*>)",
                r"(<object[^>]*>)",
                r"(<embed[^>]*>)"
            ],
            'command_injection': [
                r"(\;\s*(rm|del|format|fdisk|mkfs)\s)",
                r"(\|\s*(cat|type|dir|ls)\s)",
                r"(&&\s*(shutdown|reboot|halt)\s)",
                r"(`.*`)",
                r"(\$\([^)]*\))"
            ],
            'path_traversal': [
                r"(\.\.[\\/])",
                r"(\.\.%2f)",
                r"(%2e%2e%2f)",
                r"(/etc/passwd)",
                r"(/proc/)",
                r"(\\windows\\system32)"
            ],
            'code_injection': [
                r"(eval\s*\()",
                r"(exec\s*\()",
                r"(system\s*\()",
                r"(__import__\s*\()",
                r"(open\s*\()",
                r"(file\s*\()"
            ]
        }
        
        # PII patterns for privacy protection
        self.pii_patterns = {
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'api_key': r'\b[A-Za-z0-9]{20,}\b',  # Generic API key pattern
            'password': r'(?i)(password|pwd|pass)\s*[:=]\s*\S+',
            'secret': r'(?i)(secret|key|token)\s*[:=]\s*\S+'
        }
        
        # Inappropriate content patterns
        self.inappropriate_patterns = {
            'hate_speech': [
                r'(?i)(hate|kill|destroy)\s+(?:group|people|race)',
                r'(?i)(violent|terrorism|extremist)',
            ],
            'adult_content': [
                r'(?i)(porn|adult|explicit|nude)',
                r'(?i)(sexual|erotic)',
            ],
            'illegal_activities': [
                r'(?i)(hack|crack|pirate|steal)',
                r'(?i)(drug|weapon|illegal)',
            ]
        }
        
        # Rate limiting
        self.rate_limits = {
            'requests_per_minute': 60,
            'requests_per_hour': 1000,
            'suspicious_threshold': 10  # Suspicious if >10 failed checks per hour
        }
    
    async def check_content(self, content: str, user_id: str = "anonymous") -> SafetyCheckResult:
        """
        Comprehensive safety check for content
        
        Args:
            content: Content to check
            user_id: User identifier for tracking
            
        Returns:
            SafetyCheckResult with detailed analysis
        """
        # Rate limiting check
        if not await self._check_rate_limit(user_id):
            return SafetyCheckResult(
                is_safe=False,
                risk_level="high",
                reason="Rate limit exceeded",
                detected_patterns=["rate_limit_exceeded"],
                confidence=1.0,
                recommendations=["Please wait before making more requests"]
            )
        
        # Initialize safety check
        risk_level = "low"
        detected_patterns = []
        reasons = []
        recommendations = []
        total_confidence = 0
        
        # Check for malicious patterns
        malicious_result = self._check_malicious_patterns(content)
        if malicious_result["detected"]:
            detected_patterns.extend(malicious_result["patterns"])
            reasons.append(malicious_result["reason"])
            risk_level = max(risk_level, "critical", key=lambda x: ["low", "medium", "high", "critical"].index(x))
            total_confidence += malicious_result["confidence"]
            recommendations.extend(malicious_result["recommendations"])
        
        # Check for PII/privacy issues
        privacy_result = self._check_privacy_issues(content)
        if privacy_result["detected"]:
            detected_patterns.extend(privacy_result["patterns"])
            reasons.append(privacy_result["reason"])
            risk_level = max(risk_level, "high", key=lambda x: ["low", "medium", "high", "critical"].index(x))
            total_confidence += privacy_result["confidence"]
            recommendations.extend(privacy_result["recommendations"])
        
        # Check for inappropriate content
        appropriateness_result = self._check_appropriateness(content)
        if appropriateness_result["detected"]:
            detected_patterns.extend(appropriateness_result["patterns"])
            reasons.append(appropriateness_result["reason"])
            risk_level = max(risk_level, "medium", key=lambda x: ["low", "medium", "high", "critical"].index(x))
            total_confidence += appropriateness_result["confidence"]
            recommendations.extend(appropriateness_result["recommendations"])
        
        # Content length and complexity checks
        complexity_result = self._check_content_complexity(content)
        if complexity_result["suspicious"]:
            detected_patterns.extend(complexity_result["patterns"])
            reasons.append(complexity_result["reason"])
            risk_level = max(risk_level, "medium", key=lambda x: ["low", "medium", "high", "critical"].index(x))
            total_confidence += complexity_result["confidence"]
        
        # Calculate final confidence
        final_confidence = min(1.0, total_confidence)
        
        # Determine if content is safe
        is_safe = risk_level in ["low", "medium"] and final_confidence < 0.7
        
        # Log security event if needed
        if not is_safe:
            await self._log_security_event(user_id, "content_blocked", risk_level, {
                "content_length": len(content),
                "detected_patterns": detected_patterns,
                "confidence": final_confidence
            })
        
        # Add default recommendations
        if not recommendations:
            recommendations = ["Content appears safe for processing"]
        
        return SafetyCheckResult(
            is_safe=is_safe,
            risk_level=risk_level,
            reason="; ".join(reasons) if reasons else "No security issues detected",
            detected_patterns=detected_patterns,
            confidence=final_confidence,
            recommendations=recommendations
        )
    
    def _check_malicious_patterns(self, content: str) -> Dict[str, Any]:
        """Check for malicious injection patterns"""
        detected_patterns = []
        total_confidence = 0
        
        for attack_type, patterns in self.malicious_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                    detected_patterns.append(f"{attack_type}:{pattern}")
                    total_confidence += 0.2
        
        if detected_patterns:
            return {
                "detected": True,
                "patterns": detected_patterns,
                "reason": f"Potential {' or '.join(set(p.split(':')[0] for p in detected_patterns))} attack detected",
                "confidence": min(1.0, total_confidence),
                "recommendations": [
                    "Remove suspicious code patterns",
                    "Use parameterized queries instead of string concatenation",
                    "Validate and sanitize all inputs"
                ]
            }
        
        return {"detected": False, "patterns": [], "confidence": 0}
    
    def _check_privacy_issues(self, content: str) -> Dict[str, Any]:
        """Check for privacy/PII issues"""
        detected_patterns = []
        total_confidence = 0
        
        for pii_type, pattern in self.pii_patterns.items():
            if re.search(pattern, content):
                detected_patterns.append(f"pii:{pii_type}")
                total_confidence += 0.15
        
        if detected_patterns:
            return {
                "detected": True,
                "patterns": detected_patterns,
                "reason": f"Potential PII leakage detected: {', '.join(set(p.split(':')[1] for p in detected_patterns))}",
                "confidence": min(1.0, total_confidence),
                "recommendations": [
                    "Remove or mask personal information",
                    "Use data anonymization techniques",
                    "Ensure compliance with privacy regulations"
                ]
            }
        
        return {"detected": False, "patterns": [], "confidence": 0}
    
    def _check_appropriateness(self, content: str) -> Dict[str, Any]:
        """Check for inappropriate content"""
        detected_patterns = []
        total_confidence = 0
        
        for category, patterns in self.inappropriate_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content):
                    detected_patterns.append(f"{category}:{pattern}")
                    total_confidence += 0.1
        
        if detected_patterns:
            categories = set(p.split(':')[0] for p in detected_patterns)
            return {
                "detected": True,
                "patterns": detected_patterns,
                "reason": f"Inappropriate content detected: {', '.join(categories)}",
                "confidence": min(1.0, total_confidence),
                "recommendations": [
                    "Remove inappropriate content",
                    "Ensure content is suitable for all audiences",
                    "Follow community guidelines"
                ]
            }
        
        return {"detected": False, "patterns": [], "confidence": 0}
    
    def _check_content_complexity(self, content: str) -> Dict[str, Any]:
        """Check for suspicious content complexity"""
        issues = []
        confidence = 0
        
        # Check for extremely long content (potential DoS)
        if len(content) > 100000:  # 100KB
            issues.append("content_too_long")
            confidence += 0.3
        
        # Check for excessive special characters (potential obfuscation)
        special_chars = sum(1 for c in content if not c.isalnum() and not c.isspace())
        if special_chars > len(content) * 0.3:  # More than 30% special chars
            issues.append("excessive_special_chars")
            confidence += 0.2
        
        # Check for repetitive content (potential spam)
        words = content.lower().split()
        if len(set(words)) < len(words) * 0.3:  # Less than 30% unique words
            issues.append("repetitive_content")
            confidence += 0.1
        
        # Check for suspicious encoding
        if content.count('%') > len(content) * 0.1:  # Many URL-encoded chars
            issues.append("excessive_encoding")
            confidence += 0.2
        
        if issues:
            return {
                "suspicious": True,
                "patterns": issues,
                "reason": f"Suspicious content patterns: {', '.join(issues)}",
                "confidence": min(1.0, confidence)
            }
        
        return {"suspicious": False, "patterns": [], "confidence": 0}
    
    async def _check_rate_limit(self, user_id: str) -> bool:
        """Check if user is within rate limits"""
        current_time = datetime.now()
        
        # Initialize user tracking if needed
        if user_id not in self.user_request_counts:
            self.user_request_counts[user_id] = []
        
        # Clean old requests
        cutoff_time = current_time - timedelta(hours=1)
        self.user_request_counts[user_id] = [
            req_time for req_time in self.user_request_counts[user_id] 
            if req_time > cutoff_time
        ]
        
        # Check rate limits
        recent_requests = self.user_request_counts[user_id]
        
        # Per minute check
        minute_cutoff = current_time - timedelta(minutes=1)
        minute_requests = [req for req in recent_requests if req > minute_cutoff]
        if len(minute_requests) >= self.rate_limits['requests_per_minute']:
            return False
        
        # Per hour check
        if len(recent_requests) >= self.rate_limits['requests_per_hour']:
            return False
        
        # Add current request
        self.user_request_counts[user_id].append(current_time)
        return True
    
    async def _log_security_event(self, user_id: str, event_type: str, severity: str, details: Dict[str, Any]):
        """Log security event for monitoring"""
        event = SecurityEvent(
            timestamp=datetime.now(),
            user_id=user_id,
            event_type=event_type,
            severity=severity,
            details=details
        )
        
        self.security_events.append(event)
        
        # Keep only recent events (last 1000)
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]
        
        self.logger.warning(f"Security event: {event_type} for user {user_id} - {severity}")
    
    async def sanitize_content(self, content: str) -> Tuple[str, List[str]]:
        """
        Sanitize content by removing or masking problematic elements
        
        Returns:
            Tuple of (sanitized_content, list_of_changes)
        """
        sanitized = content
        changes = []
        
        # Remove PII
        for pii_type, pattern in self.pii_patterns.items():
            if pii_type in ['ssn', 'credit_card']:
                sanitized = re.sub(pattern, '[REDACTED]', sanitized)
                changes.append(f"Masked {pii_type}")
            elif pii_type in ['email', 'phone']:
                sanitized = re.sub(pattern, lambda m: f"[{pii_type.upper()}]", sanitized)
                changes.append(f"Masked {pii_type}")
        
        # Remove malicious code patterns
        for attack_type, patterns in self.malicious_patterns.items():
            for pattern in patterns:
                if re.search(pattern, sanitized):
                    sanitized = re.sub(pattern, '[REMOVED]', sanitized, flags=re.IGNORECASE)
                    changes.append(f"Removed {attack_type} pattern")
        
        # Limit content length
        if len(sanitized) > 50000:  # 50KB limit
            sanitized = sanitized[:50000] + "... [CONTENT TRUNCATED]"
            changes.append("Truncated excessive content")
        
        return sanitized, changes
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of security events"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_events = [e for e in self.security_events if e.timestamp > cutoff_time]
        
        if not recent_events:
            return {"message": "No security events in the specified timeframe"}
        
        # Event statistics
        event_types = {}
        severity_counts = {}
        user_events = {}
        
        for event in recent_events:
            # Event types
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
            
            # Severity levels
            severity_counts[event.severity] = severity_counts.get(event.severity, 0) + 1
            
            # User activity
            user_events[event.user_id] = user_events.get(event.user_id, 0) + 1
        
        # Suspicious users (high event count)
        suspicious_users = {
            user: count for user, count in user_events.items() 
            if count > self.rate_limits['suspicious_threshold']
        }
        
        return {
            "timeframe_hours": hours,
            "total_events": len(recent_events),
            "event_types": event_types,
            "severity_distribution": severity_counts,
            "unique_users": len(user_events),
            "suspicious_users": suspicious_users,
            "top_event_types": sorted(event_types.items(), key=lambda x: x[1], reverse=True)[:5],
            "peak_activity_hour": self._get_peak_activity_hour(recent_events)
        }
    
    def _get_peak_activity_hour(self, events: List[SecurityEvent]) -> int:
        """Find the hour with most security events"""
        hour_counts = {}
        
        for event in events:
            hour = event.timestamp.hour
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        return max(hour_counts.items(), key=lambda x: x[1])[0] if hour_counts else 0
    
    def is_user_suspicious(self, user_id: str, hours: int = 24) -> bool:
        """Check if a user has suspicious activity"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        user_events = [
            e for e in self.security_events 
            if e.user_id == user_id and e.timestamp > cutoff_time
        ]
        
        return len(user_events) > self.rate_limits['suspicious_threshold']