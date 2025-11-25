"""
Intelligent Research Assistant - Main Agent Implementation

This is the main agent that orchestrates the entire research and analysis workflow.
It demonstrates memory management, tool integration, and multi-agent coordination.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from google.adk import Agent, LlmAgent, Sequential, Parallel
from google.adk.tools import Tool

from ..memory.memory_manager import MemoryManager
from ..tools.web_search_tool import WebSearchTool
from ..tools.code_execution_tool import CodeExecutionTool
from ..tools.document_analysis_tool import DocumentAnalysisTool
from ..evaluation.evaluator import AgentEvaluator
from ..safety.safety_filter import SafetyFilter

@dataclass
class TaskResult:
    """Container for task execution results"""
    task_id: str
    status: str
    result: Any
    metadata: Dict[str, Any]
    execution_time: float
    confidence: float

class IntelligentAssistant:
    """
    Main intelligent assistant agent that coordinates all capabilities.
    
    Demonstrates:
    1. Memory management (short-term and long-term)
    2. Tool integration and orchestration
    3. Multi-agent coordination
    4. Evaluation and quality control
    5. Safety and security features
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.memory_manager = MemoryManager()
        self.safety_filter = SafetyFilter()
        self.evaluator = AgentEvaluator()
        
        # Initialize tools
        self.web_search = WebSearchTool()
        self.code_executor = CodeExecutionTool()
        self.doc_analyzer = DocumentAnalysisTool()
        
        # Initialize sub-agents
        self._setup_agents()
        
        # Session management
        self.session_id = None
        self.conversation_history = []
        
    def _setup_agents(self):
        """Setup specialized sub-agents for different tasks"""
        
        # Research Agent - handles information gathering
        self.research_agent = LlmAgent(
            name="research_agent",
            model="gemini-pro",
            instruction="""You are a research specialist agent. Your role is to:
            1. Understand research queries and identify key information needs
            2. Use web search tools to gather current, relevant information
            3. Evaluate source credibility and extract key insights
            4. Summarize findings with proper citations
            Always provide structured, well-organized results.""",
            tools=[self.web_search]
        )
        
        # Analysis Agent - processes and analyzes data
        self.analysis_agent = LlmAgent(
            name="analysis_agent", 
            model="gemini-pro",
            instruction="""You are a data analysis specialist. Your role is to:
            1. Process and analyze collected research data
            2. Identify patterns, trends, and key insights
            3. Perform statistical analysis when needed
            4. Create visualizations and summaries
            Provide clear, actionable analytical insights.""",
            tools=[self.code_executor, self.doc_analyzer]
        )
        
        # Coordination Agent - orchestrates the workflow
        self.coordination_agent = LlmAgent(
            name="coordination_agent",
            model="gemini-pro", 
            instruction="""You are a workflow coordinator. Your role is to:
            1. Break down complex tasks into manageable subtasks
            2. Determine the optimal sequence of operations
            3. Coordinate between research and analysis agents
            4. Ensure task completion and quality standards
            Maintain clear communication and efficient workflow.""",
            tools=[]
        )
        
        # Main workflow - sequential orchestration
        self.main_workflow = Sequential(
            agents=[self.coordination_agent, self.research_agent, self.analysis_agent]
        )
    
    async def process_task(self, task: str, user_id: str = "default") -> TaskResult:
        """
        Process a user task through the intelligent assistant workflow.
        
        Args:
            task: The user's research/analysis task
            user_id: User identifier for personalization
            
        Returns:
            TaskResult with the execution outcome
        """
        start_time = datetime.now()
        task_id = f"task_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Safety check
            safety_result = await self.safety_filter.check_content(task)
            if not safety_result.is_safe:
                return TaskResult(
                    task_id=task_id,
                    status="rejected",
                    result="Task rejected due to safety concerns",
                    metadata={"safety_reason": safety_result.reason},
                    execution_time=0,
                    confidence=0.0
                )
            
            # Initialize session and memory
            if not self.session_id:
                self.session_id = f"session_{start_time.strftime('%Y%m%d_%H%M%S')}"
                await self.memory_manager.initialize_session(self.session_id, user_id)
            
            # Store task in memory
            await self.memory_manager.store_task(task_id, task, user_id)
            
            # Load relevant context from memory
            context = await self.memory_manager.get_relevant_context(task, user_id)
            
            # Process through workflow
            enhanced_task = f"Context: {context}\n\nCurrent Task: {task}"
            workflow_result = await self.main_workflow.run(enhanced_task)
            
            # Store results in memory
            await self.memory_manager.store_result(task_id, workflow_result, user_id)
            
            # Evaluate result quality
            evaluation = await self.evaluator.evaluate_result(task, workflow_result)
            
            # Update conversation history
            self.conversation_history.append({
                "timestamp": start_time,
                "task": task,
                "result": workflow_result,
                "evaluation": evaluation
            })
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return TaskResult(
                task_id=task_id,
                status="completed",
                result=workflow_result,
                metadata={
                    "evaluation": evaluation,
                    "context_used": bool(context),
                    "session_id": self.session_id
                },
                execution_time=execution_time,
                confidence=evaluation.get("confidence", 0.8)
            )
            
        except Exception as e:
            self.logger.error(f"Error processing task {task_id}: {str(e)}")
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return TaskResult(
                task_id=task_id,
                status="error",
                result=f"Error processing task: {str(e)}",
                metadata={"error_type": type(e).__name__},
                execution_time=execution_time,
                confidence=0.0
            )
    
    async def get_memory_summary(self, user_id: str = "default") -> Dict[str, Any]:
        """Get a summary of stored memories for a user"""
        return await self.memory_manager.get_user_summary(user_id)
    
    async def evaluate_performance(self, num_tasks: int = 10) -> Dict[str, Any]:
        """Evaluate agent performance across recent tasks"""
        recent_tasks = self.conversation_history[-num_tasks:]
        return await self.evaluator.evaluate_performance(recent_tasks)
    
    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities"""
        return [
            "Web research and information gathering",
            "Data analysis and visualization", 
            "Code execution and computation",
            "Document analysis and summarization",
            "Multi-step task coordination",
            "Context-aware conversation",
            "Safety and content filtering",
            "Performance evaluation and quality control"
        ]