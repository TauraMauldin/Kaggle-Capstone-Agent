"""
Document Analysis Tool - Advanced document processing and analysis

Demonstrates tool integration with:
- Multiple document format support (PDF, DOCX, TXT, HTML)
- Text extraction and preprocessing
- Sentiment analysis and entity extraction
- Document summarization
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import re
import json
from pathlib import Path
import hashlib

@dataclass
class Document:
    """Document representation with metadata"""
    id: str
    title: str
    content: str
    file_type: str
    size: int
    created_at: datetime
    metadata: Dict[str, Any]

@dataclass
class AnalysisResult:
    """Result of document analysis"""
    document_id: str
    summary: str
    key_entities: List[Dict[str, Any]]
    sentiment: Dict[str, float]
    topics: List[str]
    readability_score: float
    word_count: int
    analysis_time: float

class DocumentAnalysisTool:
    """
    Advanced document analysis tool for processing various document types.
    
    Features:
    1. Support for multiple document formats
    2. Text extraction and preprocessing
    3. Entity extraction and sentiment analysis
    4. Document summarization
    5. Topic modeling and keyword extraction
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.documents = {}
        
        # Common entities patterns
        self.entity_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            'currency': r'\$\d+(?:,\d{3})*(?:\.\d{2})?|USD\d+(?:,\d{3})*(?:\.\d{2})?',
            'percentage': r'\d+(?:\.\d+)?%'
        }
        
        # Sentiment words (simplified)
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'positive',
            'success', 'successful', 'achieve', 'achieved', 'improve', 'improved', 'best'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'negative', 'fail', 'failed',
            'poor', 'worst', 'difficult', 'problem', 'issue', 'error', 'mistake'
        }
    
    async def analyze_document(self, file_path: str, file_content: Optional[str] = None) -> AnalysisResult:
        """
        Analyze a document file or content
        
        Args:
            file_path: Path to document file
            file_content: Optional content if file is not accessible
            
        Returns:
            AnalysisResult with comprehensive analysis
        """
        start_time = datetime.now()
        
        try:
            # Load or extract document content
            if file_content:
                content = file_content
                file_type = self._detect_file_type_from_content(content)
            else:
                content, file_type = await self._extract_text(file_path)
            
            # Create document object
            document = Document(
                id=hashlib.md5(content.encode()).hexdigest()[:16],
                title=Path(file_path).stem if file_path else "Untitled",
                content=content,
                file_type=file_type,
                size=len(content),
                created_at=start_time,
                metadata={"file_path": file_path}
            )
            
            self.documents[document.id] = document
            
            # Perform analysis
            analysis = await self._perform_analysis(document)
            
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            return AnalysisResult(
                document_id=document.id,
                summary=analysis["summary"],
                key_entities=analysis["entities"],
                sentiment=analysis["sentiment"],
                topics=analysis["topics"],
                readability_score=analysis["readability"],
                word_count=analysis["word_count"],
                analysis_time=analysis_time
            )
            
        except Exception as e:
            self.logger.error(f"Document analysis failed: {e}")
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            return AnalysisResult(
                document_id="error",
                summary=f"Analysis failed: {str(e)}",
                key_entities=[],
                sentiment={"positive": 0, "negative": 0, "neutral": 1},
                topics=[],
                readability_score=0,
                word_count=0,
                analysis_time=analysis_time
            )
    
    async def _extract_text(self, file_path: str) -> tuple[str, str]:
        """Extract text from various document formats"""
        file_path = Path(file_path)
        file_type = file_path.suffix.lower()
        
        try:
            if file_type == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
            elif file_type == '.pdf':
                # Use system PDF tools
                import subprocess
                result = subprocess.run(
                    ['pdftotext', '-layout', str(file_path), '-'],
                    capture_output=True, text=True
                )
                content = result.stdout
                
            elif file_type in ['.doc', '.docx']:
                # Use antiword for .doc, python-docx for .docx
                if file_type == '.doc':
                    result = subprocess.run(
                        ['antiword', str(file_path)],
                        capture_output=True, text=True
                    )
                    content = result.stdout
                else:
                    # For docx, we'd need python-docx, but for simplicity:
                    content = "DOCX file detected - would use python-docx library"
                    
            elif file_type in ['.html', '.htm']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                # Simple HTML tag removal
                content = re.sub(r'<[^>]+>', ' ', html_content)
                content = re.sub(r'\s+', ' ', content).strip()
                
            else:
                content = f"Unsupported file type: {file_type}"
                file_type = 'unknown'
                
            return content, file_type[1:] if file_type.startswith('.') else file_type
            
        except Exception as e:
            self.logger.error(f"Failed to extract text from {file_path}: {e}")
            return f"Error extracting text: {str(e)}", 'error'
    
    def _detect_file_type_from_content(self, content: str) -> str:
        """Detect file type from content"""
        if '<html' in content.lower() or '<body' in content.lower():
            return 'html'
        elif content.startswith('%PDF'):
            return 'pdf'
        else:
            return 'text'
    
    async def _perform_analysis(self, document: Document) -> Dict[str, Any]:
        """Perform comprehensive document analysis"""
        content = document.content
        words = content.split()
        
        # Basic statistics
        word_count = len(words)
        char_count = len(content)
        sentence_count = len(re.findall(r'[.!?]+', content))
        
        # Readability score (simplified Flesch-Kincaid)
        avg_sentence_length = word_count / max(1, sentence_count)
        syllable_count = sum(self._count_syllables(word) for word in words)
        avg_syllables = syllable_count / max(1, word_count)
        readability_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
        
        # Entity extraction
        entities = self._extract_entities(content)
        
        # Sentiment analysis
        sentiment = self._analyze_sentiment(content)
        
        # Topic extraction (simplified keyword extraction)
        topics = self._extract_topics(content)
        
        # Summarization (extractive)
        summary = self._generate_summary(content)
        
        return {
            "word_count": word_count,
            "readability": max(0, min(100, readability_score)),
            "entities": entities,
            "sentiment": sentiment,
            "topics": topics,
            "summary": summary
        }
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)"""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        prev_char_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_char_was_vowel:
                syllable_count += 1
            prev_char_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
            
        return max(1, syllable_count)
    
    def _extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract entities from text using patterns"""
        entities = []
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                entities.append({
                    "type": entity_type,
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })
        
        # Extract proper nouns (simplified - capitalized words)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        for noun in set(proper_nouns):
            if len(noun) > 2:  # Filter out single letters
                entities.append({
                    "type": "proper_noun",
                    "value": noun,
                    "start": content.find(noun),
                    "end": content.find(noun) + len(noun)
                })
        
        return entities[:50]  # Limit to top 50 entities
    
    def _analyze_sentiment(self, content: str) -> Dict[str, float]:
        """Analyze sentiment of text"""
        words = content.lower().split()
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        total_words = len(words)
        
        if total_words == 0:
            return {"positive": 0, "negative": 0, "neutral": 1}
        
        positive_score = positive_count / total_words
        negative_score = negative_count / total_words
        neutral_score = 1 - positive_score - negative_score
        
        return {
            "positive": positive_score,
            "negative": negative_score,
            "neutral": max(0, neutral_score)
        }
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract key topics from text (simplified keyword extraction)"""
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
        
        # Filter stop words and count frequency
        word_freq = {}
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top words
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return [word for word, freq in top_words]
    
    def _generate_summary(self, content: str) -> str:
        """Generate extractive summary of content"""
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 3:
            return content[:200] + "..." if len(content) > 200 else content
        
        # Score sentences by position and length
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            # Position score (first and last sentences are important)
            position_score = 1.0
            if i == 0 or i == len(sentences) - 1:
                position_score = 1.5
            elif i < len(sentences) * 0.3:  # First 30%
                position_score = 1.2
            
            # Length score (moderate length is better)
            length = len(sentence.split())
            length_score = min(1.0, length / 10) if length < 30 else 0.8
            
            # Keyword score (contains topic words)
            topics = self._extract_topics(sentence)
            keyword_score = min(1.0, len(topics) / 3)
            
            total_score = position_score + length_score + keyword_score
            scored_sentences.append((total_score, sentence))
        
        # Select top sentences
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        summary_sentences = [sent for score, sent in scored_sentences[:3]]
        
        # Reorder by original position
        original_order = []
        for sentence in summary_sentences:
            original_idx = sentences.index(sentence)
            original_order.append((original_idx, sentence))
        original_order.sort()
        
        summary = ". ".join(sent for idx, sent in original_order)
        return summary + "." if not summary.endswith(".") else summary
    
    async def compare_documents(self, doc1_id: str, doc2_id: str) -> Dict[str, Any]:
        """Compare two documents and find similarities/differences"""
        if doc1_id not in self.documents or doc2_id not in self.documents:
            return {"error": "One or both documents not found"}
        
        doc1 = self.documents[doc1_id]
        doc2 = self.documents[doc2_id]
        
        # Text similarity (simple word overlap)
        words1 = set(doc1.content.lower().split())
        words2 = set(doc2.content.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity_score = len(intersection) / len(union) if union else 0
        
        # Entity overlap
        entities1 = set(e["value"] for e in self._extract_entities(doc1.content))
        entities2 = set(e["value"] for e in self._extract_entities(doc2.content))
        
        entity_overlap = entities1.intersection(entities2)
        
        # Topic overlap
        topics1 = set(self._extract_topics(doc1.content))
        topics2 = set(self._extract_topics(doc2.content))
        
        topic_overlap = topics1.intersection(topics2)
        
        return {
            "similarity_score": similarity_score,
            "entity_overlap": list(entity_overlap),
            "topic_overlap": list(topic_overlap),
            "doc1_stats": {
                "word_count": len(doc1.content.split()),
                "entity_count": len(entities1),
                "topic_count": len(topics1)
            },
            "doc2_stats": {
                "word_count": len(doc2.content.split()),
                "entity_count": len(entities2),
                "topic_count": len(topics2)
            }
        }
    
    def get_document_info(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a processed document"""
        if doc_id not in self.documents:
            return None
        
        doc = self.documents[doc_id]
        return {
            "id": doc.id,
            "title": doc.title,
            "file_type": doc.file_type,
            "size": doc.size,
            "created_at": doc.created_at.isoformat(),
            "metadata": doc.metadata,
            "preview": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
        }
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all processed documents"""
        return [self.get_document_info(doc_id) for doc_id in self.documents.keys()]