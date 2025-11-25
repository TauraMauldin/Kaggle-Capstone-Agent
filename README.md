# Intelligent Research Assistant - Capstone Project

## Overview
This project demonstrates a comprehensive AI agent built with Google's Agent Development Kit (ADK) that showcases multiple advanced capabilities required for the Kaggle Agents Intensive Capstone Project.

## Agent Capabilities Demonstrated
1. **Memory Systems**: Short-term conversation memory and long-term knowledge storage
2. **Tool Integration**: Web search, code execution, and document analysis tools
3. **Multi-Agent Orchestration**: Coordinated workflow between specialized sub-agents
4. **Evaluation Framework**: Systematic performance assessment and quality control
5. **Safety Features**: Content filtering and security measures

## Architecture
- **Main Agent**: Coordinates overall task execution
- **Research Agent**: Handles web searches and information gathering
- **Analysis Agent**: Processes and analyzes collected data
- **Code Agent**: Executes code and performs computations
- **Memory Manager**: Manages conversation context and knowledge base

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
from capstone_agent.src.intelligent_assistant import IntelligentAssistant

assistant = IntelligentAssistant()
result = assistant.process_task("Research the latest developments in quantum computing")
```

## Features
- Natural language task understanding
- Multi-step task decomposition
- Real-time web information retrieval
- Code execution and analysis
- Persistent memory across sessions
- Comprehensive evaluation metrics
- Safety and security safeguards

## Capstone Requirements Met
✅ Memory systems (short-term and long-term)  
✅ Tool integration (multiple external APIs)  
✅ Orchestration (multi-agent coordination)  
✅ Evaluation framework (performance metrics)  
✅ Safety features (content filtering)