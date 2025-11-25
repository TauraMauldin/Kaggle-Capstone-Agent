"""
Agent Evaluator - Comprehensive evaluation framework for AI agents

Demonstrates evaluation capabilities with:
- Performance metrics calculation
- Quality assessment scoring
- Comparative analysis
- Benchmark testing
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import statistics
from collections import defaultdict

@dataclass
class EvaluationMetric:
    """Individual evaluation metric"""
    name: str
    value: float
    description: str
    weight: float = 1.0
    threshold: Optional[float] = None

@dataclass
class EvaluationResult:
    """Complete evaluation result"""
    agent_id: str
    task_id: str
    overall_score: float
    metrics: List[EvaluationMetric]
    timestamp: datetime
    passed_thresholds: bool
    recommendations: List[str]

@dataclass
class BenchmarkSuite:
    """Test suite for benchmarking"""
    name: str
    tasks: List[Dict[str, Any]]
    expected_results: List[Any]
    evaluation_criteria: Dict[str, float]

class AgentEvaluator:
    """
    Comprehensive evaluation framework for AI agent performance.
    
    Features:
    1. Multi-dimensional performance metrics
    2. Quality assessment and scoring
    3. Comparative analysis between agents
    4. Benchmark testing suites
    5. Automated improvement recommendations
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.evaluation_history = []
        self.benchmark_suites = self._initialize_benchmarks()
        self.performance_baselines = {}
        
        # Quality assessment weights
        self.quality_weights = {
            "accuracy": 0.3,
            "completeness": 0.2,
            "relevance": 0.2,
            "clarity": 0.15,
            "efficiency": 0.15
        }
    
    def _initialize_benchmarks(self) -> Dict[str, BenchmarkSuite]:
        """Initialize standard benchmark suites"""
        return {
            "research_tasks": BenchmarkSuite(
                name="Research Task Evaluation",
                tasks=[
                    {
                        "task": "Find recent advances in machine learning",
                        "type": "research",
                        "difficulty": "medium"
                    },
                    {
                        "task": "Analyze market trends for electric vehicles",
                        "type": "research", 
                        "difficulty": "hard"
                    }
                ],
                expected_results=[],
                evaluation_criteria={
                    "source_credibility": 0.25,
                    "information_completeness": 0.25,
                    "summary_quality": 0.25,
                    "citation_quality": 0.25
                }
            ),
            "analysis_tasks": BenchmarkSuite(
                name="Data Analysis Evaluation",
                tasks=[
                    {
                        "task": "Analyze sales data and provide insights",
                        "type": "analysis",
                        "difficulty": "medium"
                    },
                    {
                        "task": "Perform statistical analysis on experimental data",
                        "type": "analysis",
                        "difficulty": "hard"
                    }
                ],
                expected_results=[],
                evaluation_criteria={
                    "statistical_correctness": 0.3,
                    "insight_quality": 0.25,
                    "visualization_quality": 0.25,
                    "interpretation": 0.2
                }
            ),
            "code_tasks": BenchmarkSuite(
                name="Code Execution Evaluation",
                tasks=[
                    {
                        "task": "Write a function to sort an array",
                        "type": "coding",
                        "difficulty": "easy"
                    },
                    {
                        "task": "Implement a simple machine learning algorithm",
                        "type": "coding",
                        "difficulty": "hard"
                    }
                ],
                expected_results=[],
                evaluation_criteria={
                    "correctness": 0.4,
                    "efficiency": 0.2,
                    "code_quality": 0.2,
                    "documentation": 0.2
                }
            )
        }
    
    async def evaluate_result(self, task: str, result: Any, expected: Optional[Any] = None) -> Dict[str, float]:
        """
        Evaluate a single task result
        
        Args:
            task: The original task
            result: The agent's result
            expected: Expected result (if available)
            
        Returns:
            Dictionary with evaluation scores
        """
        evaluation_scores = {}
        
        # Convert result to string for analysis
        result_str = str(result) if not isinstance(result, str) else result
        
        # Accuracy evaluation
        accuracy_score = self._evaluate_accuracy(task, result_str, expected)
        evaluation_scores["accuracy"] = accuracy_score
        
        # Completeness evaluation
        completeness_score = self._evaluate_completeness(task, result_str)
        evaluation_scores["completeness"] = completeness_score
        
        # Relevance evaluation
        relevance_score = self._evaluate_relevance(task, result_str)
        evaluation_scores["relevance"] = relevance_score
        
        # Clarity evaluation
        clarity_score = self._evaluate_clarity(result_str)
        evaluation_scores["clarity"] = clarity_score
        
        # Efficiency evaluation (based on length and structure)
        efficiency_score = self._evaluate_efficiency(result_str)
        evaluation_scores["efficiency"] = efficiency_score
        
        # Calculate overall score
        overall_score = sum(
            score * self.quality_weights[criterion] 
            for criterion, score in evaluation_scores.items()
        )
        evaluation_scores["overall"] = overall_score
        
        return evaluation_scores
    
    def _evaluate_accuracy(self, task: str, result: str, expected: Optional[Any]) -> float:
        """Evaluate the accuracy of the result"""
        if expected is not None:
            # Compare with expected result
            expected_str = str(expected)
            
            # Simple string similarity (could be enhanced with semantic similarity)
            result_words = set(result.lower().split())
            expected_words = set(expected_str.lower().split())
            
            intersection = result_words.intersection(expected_words)
            union = result_words.union(expected_words)
            
            similarity = len(intersection) / len(union) if union else 0
            return similarity
        
        # If no expected result, evaluate based on factual indicators
        factual_indicators = [
            'according to', 'research shows', 'data indicates', 'statistics show',
            'studies have found', 'analysis reveals', 'evidence suggests'
        ]
        
        factual_score = 0
        for indicator in factual_indicators:
            if indicator in result.lower():
                factual_score += 0.2
        
        # Check for specific numbers and data
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?', result)
        if numbers:
            factual_score += min(0.4, len(numbers) * 0.1)
        
        return min(1.0, factual_score)
    
    def _evaluate_completeness(self, task: str, result: str) -> float:
        """Evaluate the completeness of the result"""
        # Check if result addresses key aspects of the task
        task_words = set(task.lower().split())
        result_words = set(result.lower().split())
        
        # Coverage of task keywords
        coverage = len(task_words.intersection(result_words)) / len(task_words) if task_words else 0
        
        # Length-based completeness (longer answers tend to be more complete)
        length_score = min(1.0, len(result) / 500)  # Normalize to 500 characters
        
        # Structure-based completeness (has sections, bullet points, etc.)
        structure_indicators = ['•', '-', '1.', '2.', 'firstly', 'secondly', 'finally']
        structure_score = min(0.5, sum(1 for indicator in structure_indicators if indicator in result.lower()) * 0.1)
        
        return (coverage * 0.4 + length_score * 0.3 + structure_score * 0.3)
    
    def _evaluate_relevance(self, task: str, result: str) -> float:
        """Evaluate the relevance of the result to the task"""
        # Keyword relevance
        task_keywords = self._extract_keywords(task)
        result_keywords = self._extract_keywords(result)
        
        keyword_overlap = len(task_keywords.intersection(result_keywords)) / len(task_keywords) if task_keywords else 0
        
        # Semantic relevance (simplified)
        task_concepts = self._extract_concepts(task)
        result_concepts = self._extract_concepts(result)
        
        concept_overlap = len(task_concepts.intersection(result_concepts)) / len(task_concepts) if task_concepts else 0
        
        # Avoid off-topic content (penalty)
        off_topic_indicators = ['unrelated', 'off topic', 'not relevant', 'different subject']
        off_topic_penalty = 0
        for indicator in off_topic_indicators:
            if indicator in result.lower():
                off_topic_penalty += 0.2
        
        relevance_score = (keyword_overlap * 0.5 + concept_overlap * 0.5) * max(0, 1 - off_topic_penalty)
        return max(0, relevance_score)
    
    def _evaluate_clarity(self, result: str) -> float:
        """Evaluate the clarity of the result"""
        # Readability metrics
        sentences = result.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        # Optimal sentence length is 15-20 words
        length_score = 1.0 - abs(avg_sentence_length - 17.5) / 17.5
        length_score = max(0, length_score)
        
        # Vocabulary complexity (simplified)
        words = result.split()
        complex_words = sum(1 for word in words if len(word) > 6)
        complexity_score = 1.0 - (complex_words / len(words)) if words else 1.0
        
        # Structure indicators
        structure_indicators = ['therefore', 'however', 'moreover', 'consequently', 'in conclusion']
        structure_score = min(0.5, sum(1 for indicator in structure_indicators if indicator in result.lower()) * 0.1)
        
        return (length_score * 0.4 + complexity_score * 0.3 + structure_score * 0.3)
    
    def _evaluate_efficiency(self, result: str) -> float:
        """Evaluate the efficiency of the result"""
        # Conciseness - not too verbose
        word_count = len(result.split())
        
        # Ideal length depends on complexity, but we'll use a baseline
        if word_count < 50:
            conciseness_score = 0.7  # Too short
        elif word_count < 200:
            conciseness_score = 1.0  # Good length
        elif word_count < 500:
            conciseness_score = 0.8  # A bit long
        else:
            conciseness_score = 0.6  # Too verbose
        
        # Redundancy check
        sentences = [s.strip() for s in result.split('.') if s.strip()]
        unique_sentences = set(sentences)
        redundancy_penalty = 1 - (len(unique_sentences) / len(sentences)) if sentences else 0
        
        efficiency_score = conciseness_score * (1 - redundancy_penalty * 0.3)
        return max(0, efficiency_score)
    
    def _extract_keywords(self, text: str) -> set:
        """Extract keywords from text"""
        # Simple keyword extraction
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had'
        }
        
        words = text.lower().split()
        keywords = {word for word in words if word not in stop_words and len(word) > 2}
        return keywords
    
    def _extract_concepts(self, text: str) -> set:
        """Extract concepts from text (simplified)"""
        # Look for noun phrases and important terms
        import re
        
        # Find multi-word concepts (2-3 word phrases)
        words = text.lower().split()
        concepts = set()
        
        # Two-word phrases
        for i in range(len(words) - 1):
            if words[i] not in ['the', 'a', 'an'] and words[i+1] not in ['the', 'a', 'an']:
                concepts.add(f"{words[i]} {words[i+1]}")
        
        # Single important words
        important_words = self._extract_keywords(text)
        concepts.update(important_words)
        
        return concepts
    
    async def evaluate_performance(self, task_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate overall agent performance across multiple tasks
        """
        if not task_history:
            return {"error": "No task history provided"}
        
        # Collect all evaluation metrics
        all_metrics = defaultdict(list)
        
        for task_record in task_history:
            if "evaluation" in task_record:
                evaluation = task_record["evaluation"]
                for metric, score in evaluation.items():
                    all_metrics[metric].append(score)
        
        # Calculate statistics
        performance_stats = {}
        for metric, scores in all_metrics.items():
            if scores:
                performance_stats[metric] = {
                    "mean": statistics.mean(scores),
                    "median": statistics.median(scores),
                    "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0,
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores)
                }
        
        # Calculate overall performance
        overall_scores = [task.get("evaluation", {}).get("overall", 0) for task in task_history if "evaluation" in task.get("evaluation", {})]
        
        if overall_scores:
            performance_stats["overall"] = {
                "mean": statistics.mean(overall_scores),
                "trend": self._calculate_trend(overall_scores),
                "improvement_rate": self._calculate_improvement_rate(overall_scores)
            }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(performance_stats)
        
        return {
            "performance_stats": performance_stats,
            "task_count": len(task_history),
            "recommendations": recommendations,
            "evaluation_timestamp": datetime.now().isoformat()
        }
    
    def _calculate_trend(self, scores: List[float]) -> str:
        """Calculate trend in scores"""
        if len(scores) < 2:
            return "insufficient_data"
        
        # Simple linear trend
        first_half = scores[:len(scores)//2]
        second_half = scores[len(scores)//2:]
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        if second_avg > first_avg + 0.05:
            return "improving"
        elif second_avg < first_avg - 0.05:
            return "declining"
        else:
            return "stable"
    
    def _calculate_improvement_rate(self, scores: List[float]) -> float:
        """Calculate rate of improvement"""
        if len(scores) < 2:
            return 0.0
        
        # Calculate slope of improvement
        x = list(range(len(scores)))
        n = len(scores)
        
        sum_x = sum(x)
        sum_y = sum(scores)
        sum_xy = sum(x[i] * scores[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        
        return slope
    
    def _generate_recommendations(self, performance_stats: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations based on performance"""
        recommendations = []
        
        for metric, stats in performance_stats.items():
            if isinstance(stats, dict) and "mean" in stats:
                mean_score = stats["mean"]
                
                if metric == "accuracy" and mean_score < 0.8:
                    recommendations.append("Focus on improving factual accuracy and verification")
                
                elif metric == "completeness" and mean_score < 0.7:
                    recommendations.append("Ensure responses cover all aspects of the task")
                
                elif metric == "relevance" and mean_score < 0.8:
                    recommendations.append("Improve task understanding and relevance of responses")
                
                elif metric == "clarity" and mean_score < 0.7:
                    recommendations.append("Work on clearer communication and better structure")
                
                elif metric == "efficiency" and mean_score < 0.7:
                    recommendations.append("Make responses more concise and reduce redundancy")
        
        if not recommendations:
            recommendations.append("Performance is good across all metrics")
        
        return recommendations
    
    async def run_benchmark_suite(self, suite_name: str, agent_function) -> Dict[str, Any]:
        """Run a benchmark suite against an agent"""
        if suite_name not in self.benchmark_suites:
            return {"error": f"Benchmark suite '{suite_name}' not found"}
        
        suite = self.benchmark_suites[suite_name]
        results = []
        
        for i, task in enumerate(suite.tasks):
            try:
                # Run the task
                start_time = datetime.now()
                result = await agent_function(task["task"])
                execution_time = (datetime.now() - start_time).total_seconds()
                
                # Evaluate the result
                evaluation = await self.evaluate_result(task["task"], result)
                
                results.append({
                    "task": task,
                    "result": result,
                    "evaluation": evaluation,
                    "execution_time": execution_time
                })
                
            except Exception as e:
                self.logger.error(f"Benchmark task failed: {e}")
                results.append({
                    "task": task,
                    "result": None,
                    "evaluation": {"error": str(e)},
                    "execution_time": 0
                })
        
        # Calculate suite-level metrics
        suite_metrics = self._calculate_suite_metrics(results, suite.evaluation_criteria)
        
        return {
            "suite_name": suite_name,
            "total_tasks": len(suite.tasks),
            "results": results,
            "suite_metrics": suite_metrics,
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_suite_metrics(self, results: List[Dict], criteria: Dict[str, float]) -> Dict[str, Any]:
        """Calculate metrics for the entire benchmark suite"""
        successful_results = [r for r in results if r.get("evaluation") and "error" not in r["evaluation"]]
        
        if not successful_results:
            return {"error": "No successful results to evaluate"}
        
        # Calculate weighted scores
        overall_scores = []
        execution_times = []
        
        for result in successful_results:
            evaluation = result["evaluation"]
            
            # Calculate weighted score based on criteria
            weighted_score = 0
            total_weight = 0
            
            for criterion, weight in criteria.items():
                if criterion in evaluation:
                    weighted_score += evaluation[criterion] * weight
                    total_weight += weight
            
            if total_weight > 0:
                weighted_score /= total_weight
                overall_scores.append(weighted_score)
            
            execution_times.append(result["execution_time"])
        
        return {
            "overall_score": statistics.mean(overall_scores) if overall_scores else 0,
            "score_std_dev": statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0,
            "average_execution_time": statistics.mean(execution_times) if execution_times else 0,
            "success_rate": len(successful_results) / len(results),
            "total_evaluation_criteria": criteria
        }
    
    def get_evaluation_summary(self, num_recent: int = 10) -> Dict[str, Any]:
        """Get summary of recent evaluations"""
        recent_evaluations = self.evaluation_history[-num_recent:]
        
        if not recent_evaluations:
            return {"message": "No evaluations found"}
        
        # Calculate summary statistics
        all_scores = []
        metric_scores = defaultdict(list)
        
        for eval_result in recent_evaluations:
            all_scores.append(eval_result.overall_score)
            for metric in eval_result.metrics:
                metric_scores[metric.name].append(metric.value)
        
        summary = {
            "total_evaluations": len(recent_evaluations),
            "average_overall_score": statistics.mean(all_scores),
            "metric_averages": {},
            "trend": self._calculate_trend(all_scores),
            "date_range": {
                "start": min(e.timestamp for e in recent_evaluations).isoformat(),
                "end": max(e.timestamp for e in recent_evaluations).isoformat()
            }
        }
        
        for metric, scores in metric_scores.items():
            summary["metric_averages"][metric] = statistics.mean(scores)
        
        return summary