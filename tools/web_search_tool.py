"""
Web Search Tool - Advanced web information gathering capabilities

Demonstrates tool integration with:
- Real-time web search functionality
- Source credibility assessment
- Content extraction and summarization
- Rate limiting and caching
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import re
from urllib.parse import urljoin, urlparse
import hashlib

@dataclass
class SearchResult:
    """Individual search result with metadata"""
    title: str
    url: str
    snippet: str
    source: str
    credibility_score: float
    timestamp: datetime
    content: Optional[str] = None

@dataclass
class SearchQuery:
    """Search query with parameters"""
    query: str
    max_results: int = 10
    time_range: str = "any"  # 'day', 'week', 'month', 'year', 'any'
    source_type: str = "web"  # 'web', 'news', 'academic'
    
class WebSearchTool:
    """
    Advanced web search tool for information gathering.
    
    Features:
    1. Multi-source search capabilities
    2. Source credibility assessment
    3. Content extraction and summarization
    4. Rate limiting and caching
    5. Search result ranking
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = None
        self.cache = {}
        self.rate_limiter = {}
        
        # Credible sources for different domains
        self.credible_sources = {
            'academic': ['arxiv.org', 'scholar.google.com', 'pubmed.ncbi.nlm.nih.gov', 'nature.com', 'science.org'],
            'news': ['reuters.com', 'apnews.com', 'bbc.com', 'npr.org', 'wsj.com', 'nytimes.com'],
            'tech': ['techcrunch.com', 'wired.com', 'arstechnica.com', 'venturebeat.com', 'theverge.com'],
            'research': ['ieee.org', 'acm.org', 'springer.com', 'sciencedirect.com', 'dl.acm.org']
        }
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """
        Perform web search with given parameters
        
        Args:
            query: SearchQuery object with search parameters
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Check rate limiting
            if not await self._check_rate_limit():
                raise Exception("Rate limit exceeded")
            
            # Check cache first
            cache_key = self._get_cache_key(query)
            if cache_key in self.cache:
                self.logger.info(f"Cache hit for query: {query.query}")
                return self.cache[cache_key]
            
            # Perform search
            results = await self._perform_search(query)
            
            # Assess credibility and rank results
            ranked_results = await self._rank_results(results)
            
            # Cache results
            self.cache[cache_key] = ranked_results
            
            return ranked_results
            
        except Exception as e:
            self.logger.error(f"Search failed for query '{query.query}': {e}")
            return []
    
    async def _perform_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform the actual search using available APIs"""
        
        # For demonstration, we'll simulate search results
        # In a real implementation, this would integrate with search APIs
        
        simulated_results = []
        
        # Simulate different types of sources based on query
        query_lower = query.query.lower()
        
        if any(word in query_lower for word in ['research', 'paper', 'study', 'academic']):
            # Academic sources
            simulated_results.extend(self._simulate_academic_results(query))
        elif any(word in query_lower for word in ['news', 'current', 'today', 'latest']):
            # News sources
            simulated_results.extend(self._simulate_news_results(query))
        elif any(word in query_lower for word in ['technology', 'tech', 'programming', 'code']):
            # Tech sources
            simulated_results.extend(self._simulate_tech_results(query))
        else:
            # General web search
            simulated_results.extend(self._simulate_general_results(query))
        
        # Limit results
        return simulated_results[:query.max_results]
    
    def _simulate_academic_results(self, query: SearchQuery) -> List[SearchResult]:
        """Simulate academic search results"""
        return [
            SearchResult(
                title=f"Recent Advances in {query.query.title()}: A Comprehensive Review",
                url="https://arxiv.org/abs/2023.12345",
                snippet=f"This paper provides a comprehensive overview of recent developments in {query.query}...",
                source="arXiv",
                credibility_score=0.9,
                timestamp=datetime.now() - timedelta(days=30),
                content=f"Abstract: We present a comprehensive review of {query.query}, covering theoretical foundations, practical applications, and future directions..."
            ),
            SearchResult(
                title=f"Machine Learning Approaches to {query.query.title()}",
                url="https://scholar.google.com/citations?user=example",
                snippet=f"Novel machine learning techniques for solving {query.query} problems...",
                source="Google Scholar",
                credibility_score=0.85,
                timestamp=datetime.now() - timedelta(days=60),
                content=f"This study explores advanced machine learning approaches to {query.query}..."
            )
        ]
    
    def _simulate_news_results(self, query: SearchQuery) -> List[SearchResult]:
        """Simulate news search results"""
        return [
            SearchResult(
                title=f"Breaking: Latest Developments in {query.query.title()}",
                url="https://reuters.com/technology/example-article",
                snippet=f"Latest news and developments regarding {query.query} from reliable sources...",
                source="Reuters",
                credibility_score=0.95,
                timestamp=datetime.now() - timedelta(hours=2),
                content=f"Recent developments in {query.query} have garnered significant attention from experts and policymakers..."
            ),
            SearchResult(
                title=f"Industry Report: {query.query.title()} Market Analysis",
                url="https://wsj.com/articles/example",
                snippet=f"In-depth analysis of {query.query} market trends and projections...",
                source="Wall Street Journal",
                credibility_score=0.9,
                timestamp=datetime.now() - timedelta(days=1),
                content=f"The {query.query} market has experienced significant growth in recent quarters..."
            )
        ]
    
    def _simulate_tech_results(self, query: SearchQuery) -> List[SearchResult]:
        """Simulate technology search results"""
        return [
            SearchResult(
                title=f"Implementing {query.query.title()} with Modern Frameworks",
                url="https://github.com/example/repository",
                snippet=f"Open-source implementation and tutorials for {query.query}...",
                source="GitHub",
                credibility_score=0.8,
                timestamp=datetime.now() - timedelta(days=7),
                content=f"This repository provides comprehensive implementations of {query.query} using modern frameworks..."
            ),
            SearchResult(
                title=f"Developer Guide: {query.query.title()} Best Practices",
                url="https://techcrunch.com/guides/example",
                snippet=f"Complete developer guide for {query.query} with code examples...",
                source="TechCrunch",
                credibility_score=0.85,
                timestamp=datetime.now() - timedelta(days=3),
                content=f"A comprehensive guide covering best practices for {query.query} development..."
            )
        ]
    
    def _simulate_general_results(self, query: SearchQuery) -> List[SearchResult]:
        """Simulate general web search results"""
        return [
            SearchResult(
                title=f"Complete Guide to {query.query.title()}",
                url="https://example.com/guide",
                snippet=f"Comprehensive guide covering all aspects of {query.query}...",
                source="Example.com",
                credibility_score=0.7,
                timestamp=datetime.now() - timedelta(days=14),
                content=f"This guide provides detailed information about {query.query}..."
            ),
            SearchResult(
                title=f"{query.query.title()}: FAQ and Resources",
                url="https://example.org/faq",
                snippet=f"Frequently asked questions and resources about {query.query}...",
                source="Example.org",
                credibility_score=0.75,
                timestamp=datetime.now() - timedelta(days=21),
                content=f"Common questions and answers about {query.query} with additional resources..."
            )
        ]
    
    async def _rank_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Rank search results by relevance and credibility"""
        
        # Enhanced ranking algorithm
        for result in results:
            # Base score from credibility
            base_score = result.credibility_score
            
            # Freshness factor (newer is better for news)
            freshness_factor = max(0.1, 1.0 - (datetime.now() - result.timestamp).days / 30)
            
            # Content length factor (longer content might be more comprehensive)
            content_factor = min(1.0, len(result.content or result.snippet) / 500)
            
            # Combined score
            result.credibility_score = base_score * 0.6 + freshness_factor * 0.2 + content_factor * 0.2
        
        # Sort by final score
        results.sort(key=lambda x: x.credibility_score, reverse=True)
        
        return results
    
    async def _check_rate_limit(self) -> bool:
        """Check if search requests are within rate limits"""
        current_time = datetime.now()
        
        # Clean old entries
        cutoff = current_time - timedelta(minutes=1)
        self.rate_limiter = {k: v for k, v in self.rate_limiter.items() if v > cutoff}
        
        # Check current rate (max 10 requests per minute)
        if len(self.rate_limiter) >= 10:
            return False
        
        # Add current request
        self.rate_limiter[current_time] = current_time
        return True
    
    def _get_cache_key(self, query: SearchQuery) -> str:
        """Generate cache key for search query"""
        key_data = f"{query.query}:{query.max_results}:{query.time_range}:{query.source_type}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def extract_content(self, url: str) -> Optional[str]:
        """Extract main content from a URL"""
        try:
            # Check rate limit
            if not await self._check_rate_limit():
                return None
            
            # For demonstration, return simulated content
            # In a real implementation, this would use web scraping libraries
            
            domain = urlparse(url).netloc
            if 'arxiv' in domain:
                return "Abstract: This paper presents novel research findings with detailed methodology and results..."
            elif 'github' in domain:
                return "Repository containing source code, documentation, and examples for the project..."
            elif 'news' in domain or any(news_site in domain for news_site in ['reuters', 'bbc', 'wsj']):
                return "Article covering the latest developments with expert commentary and analysis..."
            else:
                return "General web content with relevant information about the topic..."
                
        except Exception as e:
            self.logger.error(f"Failed to extract content from {url}: {e}")
            return None
    
    async def get_source_analysis(self, url: str) -> Dict[str, Any]:
        """Analyze the credibility and reliability of a source"""
        try:
            domain = urlparse(url).netloc.lower()
            
            # Check against credible sources list
            credibility_score = 0.5  # Base score
            
            for category, sources in self.credible_sources.items():
                if any(source in domain for source in sources):
                    credibility_score = 0.8 + (0.1 if category in ['academic', 'news'] else 0)
                    break
            
            # Additional factors
            factors = {
                "domain_authority": credibility_score,
                "content_freshness": True,  # Would check actual content date
                "peer_reviewed": 'arxiv' in domain or 'scholar' in domain,
                "cited_sources": True,  # Would check for citations
                "bias_indicators": []  # Would analyze for bias
            }
            
            return {
                "credibility_score": credibility_score,
                "category": self._get_source_category(domain),
                "factors": factors
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze source {url}: {e}")
            return {"credibility_score": 0.3, "error": str(e)}
    
    def _get_source_category(self, domain: str) -> str:
        """Categorize the source type"""
        for category, sources in self.credible_sources.items():
            if any(source in domain for source in sources):
                return category
        
        if 'blog' in domain:
            return 'blog'
        elif 'wiki' in domain:
            return 'wiki'
        elif 'gov' in domain:
            return 'government'
        elif 'edu' in domain:
            return 'education'
        else:
            return 'general'
    
    async def search_and_extract(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Combined search and content extraction
        """
        search_query = SearchQuery(query=query, max_results=max_results)
        results = await self.search(search_query)
        
        # Extract full content for top results
        enriched_results = []
        for result in results[:3]:  # Extract content for top 3
            content = await self.extract_content(result.url)
            source_analysis = await self.get_source_analysis(result.url)
            
            enriched_result = result
            enriched_result.content = content
            enriched_results.append({
                "result": enriched_result,
                "source_analysis": source_analysis
            })
        
        return {
            "query": query,
            "results": enriched_results,
            "total_found": len(results),
            "search_time": datetime.now().isoformat()
        }