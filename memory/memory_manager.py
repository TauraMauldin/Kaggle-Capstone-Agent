"""
Memory Manager - Handles both short-term and long-term memory systems

Demonstrates sophisticated memory management with:
- Session-based short-term memory
- Persistent long-term knowledge storage  
- Context retrieval and relevance scoring
- Memory consolidation and cleanup
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib

@dataclass
class MemoryItem:
    """Individual memory item with metadata"""
    id: str
    content: str
    type: str  # 'task', 'result', 'conversation', 'knowledge'
    timestamp: datetime
    user_id: str
    session_id: str
    importance_score: float
    tags: List[str]
    embedding: Optional[List[float]] = None

@dataclass
class Session:
    """User session information"""
    session_id: str
    user_id: str
    start_time: datetime
    last_activity: datetime
    context_summary: str
    task_count: int

class MemoryManager:
    """
    Advanced memory management system for the intelligent assistant.
    
    Features:
    1. Short-term session memory for active conversations
    2. Long-term persistent knowledge storage
    3. Context-aware retrieval with relevance scoring
    4. Memory consolidation and cleanup
    5. Personalized memory per user
    """
    
    def __init__(self, storage_path: str = "memory_storage.json"):
        self.logger = logging.getLogger(__name__)
        self.storage_path = storage_path
        self.sessions: Dict[str, Session] = {}
        self.memories: Dict[str, MemoryItem] = {}
        self.user_profiles: Dict[str, Dict] = {}
        
        # Load existing data
        self._load_storage()
        
        # Start cleanup task
        asyncio.create_task(self._periodic_cleanup())
    
    def _load_storage(self):
        """Load existing memory data from storage"""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                
            # Reconstruct memories
            for mem_data in data.get('memories', []):
                mem_data['timestamp'] = datetime.fromisoformat(mem_data['timestamp'])
                memory = MemoryItem(**mem_data)
                self.memories[memory.id] = memory
            
            # Reconstruct sessions
            for sess_data in data.get('sessions', []):
                sess_data['start_time'] = datetime.fromisoformat(sess_data['start_time'])
                sess_data['last_activity'] = datetime.fromisoformat(sess_data['last_activity'])
                session = Session(**sess_data)
                self.sessions[session.session_id] = session
            
            self.user_profiles = data.get('user_profiles', {})
            self.logger.info(f"Loaded {len(self.memories)} memories and {len(self.sessions)} sessions")
            
        except FileNotFoundError:
            self.logger.info("No existing memory storage found, starting fresh")
        except Exception as e:
            self.logger.error(f"Error loading memory storage: {e}")
    
    def _save_storage(self):
        """Save memory data to persistent storage"""
        try:
            data = {
                'memories': [asdict(mem) for mem in self.memories.values()],
                'sessions': [asdict(sess) for sess in self.sessions.values()],
                'user_profiles': self.user_profiles,
                'last_saved': datetime.now().isoformat()
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Error saving memory storage: {e}")
    
    async def initialize_session(self, session_id: str, user_id: str) -> Session:
        """Initialize a new user session"""
        session = Session(
            session_id=session_id,
            user_id=user_id,
            start_time=datetime.now(),
            last_activity=datetime.now(),
            context_summary="",
            task_count=0
        )
        
        self.sessions[session_id] = session
        
        # Initialize user profile if needed
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'first_seen': datetime.now().isoformat(),
                'total_sessions': 0,
                'preferences': {},
                'expertise_areas': []
            }
        
        self.user_profiles[user_id]['total_sessions'] += 1
        self._save_storage()
        
        return session
    
    async def store_task(self, task_id: str, task: str, user_id: str, session_id: Optional[str] = None):
        """Store a task in memory"""
        if not session_id:
            session_id = self._get_latest_session(user_id)
        
        memory = MemoryItem(
            id=f"task_{task_id}",
            content=task,
            type="task",
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=session_id or "",
            importance_score=self._calculate_importance(task),
            tags=self._extract_tags(task)
        )
        
        self.memories[memory.id] = memory
        
        # Update session
        if session_id and session_id in self.sessions:
            self.sessions[session_id].last_activity = datetime.now()
            self.sessions[session_id].task_count += 1
        
        self._save_storage()
    
    async def store_result(self, task_id: str, result: Any, user_id: str, session_id: Optional[str] = None):
        """Store a task result in memory"""
        if not session_id:
            session_id = self._get_latest_session(user_id)
        
        # Convert result to string for storage
        result_str = str(result) if not isinstance(result, str) else result
        
        memory = MemoryItem(
            id=f"result_{task_id}",
            content=result_str,
            type="result",
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=session_id or "",
            importance_score=self._calculate_importance(result_str),
            tags=self._extract_tags(result_str)
        )
        
        self.memories[memory.id] = memory
        
        # Update session context
        if session_id and session_id in self.sessions:
            self.sessions[session_id].last_activity = datetime.now()
            self._update_session_context(session_id, result_str)
        
        self._save_storage()
    
    async def store_knowledge(self, knowledge: str, user_id: str, tags: List[str] = None):
        """Store important knowledge in long-term memory"""
        memory = MemoryItem(
            id=f"knowledge_{hashlib.md5(knowledge.encode()).hexdigest()[:8]}",
            content=knowledge,
            type="knowledge",
            timestamp=datetime.now(),
            user_id=user_id,
            session_id="",
            importance_score=1.0,  # Knowledge items are always important
            tags=tags or []
        )
        
        self.memories[memory.id] = memory
        self._save_storage()
    
    async def get_relevant_context(self, current_task: str, user_id: str, max_items: int = 5) -> str:
        """Retrieve relevant context from memory"""
        user_memories = [mem for mem in self.memories.values() if mem.user_id == user_id]
        
        # Sort by relevance (simple keyword matching for now)
        task_words = set(current_task.lower().split())
        scored_memories = []
        
        for memory in user_memories:
            memory_words = set(memory.content.lower().split())
            overlap = len(task_words.intersection(memory_words))
            
            # Time decay factor
            days_old = (datetime.now() - memory.timestamp).days
            time_factor = max(0.1, 1.0 - (days_old / 30))  # Decay over 30 days
            
            # Combined score
            score = (overlap / len(task_words)) * memory.importance_score * time_factor
            scored_memories.append((score, memory))
        
        # Get top memories
        scored_memories.sort(reverse=True, key=lambda x: x[0])
        top_memories = [mem for _, mem in scored_memories[:max_items]]
        
        # Format context
        context_parts = []
        for memory in top_memories:
            context_parts.append(f"[{memory.type.title()} - {memory.timestamp.strftime('%Y-%m-%d')}]: {memory.content[:200]}...")
        
        return "\n".join(context_parts) if context_parts else ""
    
    async def get_user_summary(self, user_id: str) -> Dict[str, Any]:
        """Get a comprehensive summary of user's memory and activity"""
        user_memories = [mem for mem in self.memories.values() if mem.user_id == user_id]
        user_sessions = [sess for sess in self.sessions.values() if sess.user_id == user_id]
        
        # Activity statistics
        task_memories = [mem for mem in user_memories if mem.type == "task"]
        result_memories = [mem for mem in user_memories if mem.type == "result"]
        
        # Time analysis
        recent_memories = [mem for mem in user_memories if mem.timestamp > datetime.now() - timedelta(days=7)]
        
        # Expertise areas from tags
        all_tags = []
        for memory in user_memories:
            all_tags.extend(memory.tags)
        
        tag_counts = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "user_id": user_id,
            "profile": self.user_profiles.get(user_id, {}),
            "statistics": {
                "total_memories": len(user_memories),
                "total_sessions": len(user_sessions),
                "total_tasks": len(task_memories),
                "total_results": len(result_memories),
                "recent_activity": len(recent_memories)
            },
            "expertise_areas": top_tags,
            "last_activity": max([mem.timestamp for mem in user_memories]) if user_memories else None
        }
    
    def _get_latest_session(self, user_id: str) -> Optional[str]:
        """Get the most recent session for a user"""
        user_sessions = [sess for sess in self.sessions.values() if sess.user_id == user_id]
        if not user_sessions:
            return None
        return max(user_sessions, key=lambda s: s.last_activity).session_id
    
    def _calculate_importance(self, content: str) -> float:
        """Calculate importance score for a memory item"""
        # Simple heuristic based on content characteristics
        importance = 0.5  # Base importance
        
        # Length factor
        if len(content) > 500:
            importance += 0.2
        
        # Question marks (indicates questions/queries)
        if '?' in content:
            importance += 0.1
        
        # Numbers and data indicators
        if any(char.isdigit() for char in content):
            importance += 0.1
        
        # Keywords indicating importance
        important_keywords = ['important', 'critical', 'urgent', 'key', 'essential', 'summary']
        if any(keyword in content.lower() for keyword in important_keywords):
            importance += 0.2
        
        return min(1.0, importance)
    
    def _extract_tags(self, content: str) -> List[str]:
        """Extract relevant tags from content"""
        # Simple keyword extraction
        content_lower = content.lower()
        
        # Common domains/areas
        domains = ['research', 'analysis', 'code', 'data', 'python', 'machine learning', 
                  'ai', 'web', 'database', 'api', 'visualization', 'statistics']
        
        tags = []
        for domain in domains:
            if domain in content_lower:
                tags.append(domain)
        
        return tags[:5]  # Limit to 5 tags
    
    def _update_session_context(self, session_id: str, new_content: str):
        """Update the context summary for a session"""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        
        # Simple context update - concatenate and truncate
        if session.context_summary:
            session.context_summary = session.context_summary + " | " + new_content[:100]
        else:
            session.context_summary = new_content[:200]
        
        # Keep context manageable
        if len(session.context_summary) > 500:
            session.context_summary = session.context_summary[-500:]
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of old memories"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                cutoff_date = datetime.now() - timedelta(days=90)
                old_memories = [mem_id for mem_id, mem in self.memories.items() 
                              if mem.timestamp < cutoff_date and mem.importance_score < 0.3]
                
                for mem_id in old_memories:
                    del self.memories[mem_id]
                
                if old_memories:
                    self.logger.info(f"Cleaned up {len(old_memories)} old memories")
                    self._save_storage()
                    
            except Exception as e:
                self.logger.error(f"Error in periodic cleanup: {e}")