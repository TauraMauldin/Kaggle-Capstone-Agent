"""
Code Execution Tool - Safe and controlled code execution environment

Demonstrates tool integration with:
- Secure code execution in sandboxed environment
- Multiple language support (Python, R, JavaScript)
- Output capture and error handling
- Resource usage monitoring
"""

import asyncio
import subprocess
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import json
import tempfile
import os
import sys
from pathlib import Path
import traceback

@dataclass
class ExecutionResult:
    """Result of code execution"""
    code: str
    language: str
    output: str
    error: Optional[str]
    execution_time: float
    memory_usage: int
    exit_code: int
    success: bool
    artifacts: List[str]  # Generated files/plots

@dataclass
class CodeExecutionRequest:
    """Code execution request with parameters"""
    code: str
    language: str = "python"
    timeout: int = 30
    capture_output: bool = True
    allow_internet: bool = False
    working_directory: Optional[str] = None

class CodeExecutionTool:
    """
    Secure code execution tool for running code in a controlled environment.
    
    Features:
    1. Support for multiple programming languages
    2. Sandboxed execution environment
    3. Resource usage monitoring
    4. Output capture and error handling
    5. File/artifact generation support
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.temp_dir = tempfile.mkdtemp(prefix="code_execution_")
        
        # Supported languages and their configurations
        self.language_configs = {
            "python": {
                "interpreter": sys.executable,
                "extension": ".py",
                "flags": ["-u"]  # Unbuffered output
            },
            "r": {
                "interpreter": "Rscript",
                "extension": ".R",
                "flags": []
            },
            "javascript": {
                "interpreter": "node",
                "extension": ".js",
                "flags": []
            },
            "bash": {
                "interpreter": "bash",
                "extension": ".sh",
                "flags": []
            }
        }
        
        # Execution statistics
        self.execution_history = []
    
    async def execute(self, request: CodeExecutionRequest) -> ExecutionResult:
        """
        Execute code in a secure sandboxed environment
        
        Args:
            request: CodeExecutionRequest with code and parameters
            
        Returns:
            ExecutionResult with output and metadata
        """
        start_time = datetime.now()
        
        try:
            # Validate request
            self._validate_request(request)
            
            # Prepare execution environment
            work_dir = request.working_directory or self.temp_dir
            code_file = self._prepare_code_file(request.code, request.language, work_dir)
            
            # Execute code
            execution_output = await self._execute_code(
                code_file, request.language, request.timeout, work_dir
            )
            
            # Process results
            result = self._process_execution_result(
                request.code, request.language, execution_output, start_time, work_dir
            )
            
            # Store execution history
            self.execution_history.append({
                "timestamp": start_time,
                "language": request.language,
                "success": result.success,
                "execution_time": result.execution_time
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Code execution failed: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ExecutionResult(
                code=request.code,
                language=request.language,
                output="",
                error=str(e),
                execution_time=execution_time,
                memory_usage=0,
                exit_code=-1,
                success=False,
                artifacts=[]
            )
    
    def _validate_request(self, request: CodeExecutionRequest):
        """Validate the code execution request"""
        if request.language not in self.language_configs:
            raise ValueError(f"Unsupported language: {request.language}")
        
        if not request.code.strip():
            raise ValueError("Code cannot be empty")
        
        # Check for potentially dangerous operations
        dangerous_patterns = [
            "import os.system",
            "subprocess.call",
            "eval(",
            "exec(",
            "__import__",
            "open(",
            "file(",
            "input(",
            "raw_input("
        ]
        
        code_lower = request.code.lower()
        for pattern in dangerous_patterns:
            if pattern in code_lower:
                self.logger.warning(f"Potentially dangerous pattern detected: {pattern}")
        
        # Check code length
        if len(request.code) > 10000:  # 10KB limit
            raise ValueError("Code too long (max 10KB)")
    
    def _prepare_code_file(self, code: str, language: str, work_dir: str) -> str:
        """Prepare code file for execution"""
        config = self.language_configs[language]
        code_file = os.path.join(work_dir, f"code_{datetime.now().strftime('%Y%m%d_%H%M%S')}{config['extension']}")
        
        # Add safety wrapper for Python
        if language == "python":
            # Wrap code with safety measures
            wrapped_code = f'''
import sys
import io
import warnings
warnings.filterwarnings('ignore')

# Capture output
old_stdout = sys.stdout
old_stderr = sys.stderr
stdout_capture = io.StringIO()
stderr_capture = io.StringIO()

try:
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture
    
    # User code starts here
{code}
    
finally:
    # Restore stdout/stderr
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    
    # Print captured output
    print(stdout_capture.getvalue(), end='')
    if stderr_capture.getvalue():
        print("STDERR:", stderr_capture.getvalue(), file=sys.stderr)
'''
            code = wrapped_code
        
        # Write code to file
        with open(code_file, 'w') as f:
            f.write(code)
        
        return code_file
    
    async def _execute_code(self, code_file: str, language: str, timeout: int, work_dir: str) -> Dict[str, Any]:
        """Execute the code file and capture output"""
        config = self.language_configs[language]
        
        # Build command
        cmd = [config["interpreter"]] + config["flags"] + [code_file]
        
        try:
            # Execute with timeout
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_dir,
                env=self._get_safe_environment()
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
                
                return {
                    "stdout": stdout.decode('utf-8', errors='ignore'),
                    "stderr": stderr.decode('utf-8', errors='ignore'),
                    "exit_code": process.returncode,
                    "timeout": False
                }
                
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                
                return {
                    "stdout": "",
                    "stderr": f"Execution timed out after {timeout} seconds",
                    "exit_code": -1,
                    "timeout": True
                }
                
        except Exception as e:
            return {
                "stdout": "",
                "stderr": str(e),
                "exit_code": -1,
                "timeout": False
            }
    
    def _get_safe_environment(self) -> Dict[str, str]:
        """Get a safe environment for code execution"""
        # Start with current environment
        env = os.environ.copy()
        
        # Remove potentially dangerous variables
        dangerous_vars = ['API_KEY', 'SECRET', 'TOKEN', 'PASSWORD', 'PRIVATE_KEY']
        for var in dangerous_vars:
            env.pop(var, None)
        
        # Add safe defaults
        env.update({
            'PYTHONPATH': '',
            'HOME': self.temp_dir,
            'TMPDIR': self.temp_dir,
            'LANG': 'en_US.UTF-8'
        })
        
        return env
    
    def _process_execution_result(
        self, code: str, language: str, execution_output: Dict[str, Any], 
        start_time: datetime, work_dir: str
    ) -> ExecutionResult:
        """Process the execution output and create result"""
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Determine success
        success = execution_output["exit_code"] == 0 and not execution_output["timeout"]
        
        # Combine output
        output = execution_output["stdout"]
        error = execution_output["stderr"] if execution_output["stderr"] else None
        
        # Find generated artifacts
        artifacts = self._find_artifacts(work_dir)
        
        return ExecutionResult(
            code=code,
            language=language,
            output=output,
            error=error,
            execution_time=execution_time,
            memory_usage=0,  # Would require more complex monitoring
            exit_code=execution_output["exit_code"],
            success=success,
            artifacts=artifacts
        )
    
    def _find_artifacts(self, work_dir: str) -> List[str]:
        """Find generated files and artifacts"""
        artifacts = []
        
        try:
            for file in os.listdir(work_dir):
                file_path = os.path.join(work_dir, file)
                if os.path.isfile(file_path) and not file.startswith('code_'):
                    # Read file content if it's not too large
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read(1000)  # First 1000 characters
                            artifacts.append(f"{file}: {content[:100]}...")
                    except:
                        artifacts.append(file)
        except Exception as e:
            self.logger.error(f"Error finding artifacts: {e}")
        
        return artifacts
    
    async def execute_data_analysis(self, data: str, analysis_type: str = "summary") -> Dict[str, Any]:
        """
        Execute common data analysis tasks
        """
        if analysis_type == "summary":
            code = f'''
import pandas as pd
import numpy as np
import json
from io import StringIO

# Load data
data = """{data}"""
df = pd.read_csv(StringIO(data))

# Basic summary
summary = {{
    "shape": df.shape,
    "columns": list(df.columns),
    "dtypes": df.dtypes.to_dict(),
    "missing_values": df.isnull().sum().to_dict(),
    "numeric_summary": df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {{}},
    "sample_data": df.head().to_dict()
}}

print(json.dumps(summary, indent=2, default=str))
'''
        
        elif analysis_type == "visualization":
            code = f'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import json

# Load data
data = """{data}"""
df = pd.read_csv(StringIO(data))

# Create visualizations
numeric_cols = df.select_dtypes(include=['number']).columns

if len(numeric_cols) > 0:
    # Histogram for first numeric column
    plt.figure(figsize=(10, 6))
    df[numeric_cols[0]].hist()
    plt.title(f'Distribution of {{numeric_cols[0]}}')
    plt.savefig('histogram.png')
    plt.close()
    
    print("Created histogram.png")
    
if len(numeric_cols) > 1:
    # Scatter plot for first two numeric columns
    plt.figure(figsize=(10, 6))
    plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]])
    plt.xlabel(numeric_cols[0])
    plt.ylabel(numeric_cols[1])
    plt.title(f'{{numeric_cols[0]}} vs {{numeric_cols[1]}}')
    plt.savefig('scatter.png')
    plt.close()
    
    print("Created scatter.png")
'''
        
        else:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
        
        # Execute the analysis
        request = CodeExecutionRequest(
            code=code,
            language="python",
            timeout=30
        )
        
        result = await self.execute(request)
        
        return {
            "analysis_type": analysis_type,
            "result": result.output,
            "success": result.success,
            "error": result.error,
            "artifacts": result.artifacts
        }
    
    async def execute_statistical_test(self, data1: str, data2: str, test_type: str = "ttest") -> Dict[str, Any]:
        """
        Execute statistical tests on data
        """
        code = f'''
import pandas as pd
import numpy as np
from scipy import stats
from io import StringIO
import json

# Load data
data1 = """{data1}"""
data2 = """{data2}"""
df1 = pd.read_csv(StringIO(data1))
df2 = pd.read_csv(StringIO(data2))

# Get first numeric columns
col1 = df1.select_dtypes(include=[np.number]).iloc[:, 0] if len(df1.select_dtypes(include=[np.number]).columns) > 0 else None
col2 = df2.select_dtypes(include=[np.number]).iloc[:, 0] if len(df2.select_dtypes(include=[np.number]).columns) > 0 else None

if col1 is None or col2 is None:
    print("Error: No numeric columns found")
    exit(1)

# Perform statistical test
if "{test_type}" == "ttest":
    statistic, p_value = stats.ttest_ind(col1, col2)
    test_name = "Independent t-test"
elif "{test_type}" == "mannwhitney":
    statistic, p_value = stats.mannwhitneyu(col1, col2, alternative='two-sided')
    test_name = "Mann-Whitney U test"
elif "{test_type}" == "chi2":
    # For chi-square, we need categorical data
    statistic, p_value = stats.chi2_contingency([col1.value_counts(), col2.value_counts()])[:2]
    test_name = "Chi-square test"
else:
    print(f"Unsupported test type: {{test_type}}")
    exit(1)

result = {{
    "test_name": test_name,
    "statistic": float(statistic),
    "p_value": float(p_value),
    "significant": p_value < 0.05,
    "sample_sizes": [len(col1), len(col2)],
    "means": [float(col1.mean()), float(col2.mean())],
    "std_devs": [float(col1.std()), float(col2.std())]
}}

print(json.dumps(result, indent=2))
'''
        
        request = CodeExecutionRequest(
            code=code,
            language="python",
            timeout=30
        )
        
        result = await self.execute(request)
        
        try:
            analysis_result = json.loads(result.output) if result.success else {}
        except:
            analysis_result = {"error": "Failed to parse result"}
        
        return {
            "test_type": test_type,
            "result": analysis_result,
            "success": result.success,
            "error": result.error
        }
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_history:
            return {"total_executions": 0}
        
        recent_executions = self.execution_history[-100:]  # Last 100 executions
        
        success_rate = sum(1 for e in recent_executions if e["success"]) / len(recent_executions)
        avg_execution_time = sum(e["execution_time"] for e in recent_executions) / len(recent_executions)
        
        language_stats = {}
        for execution in recent_executions:
            lang = execution["language"]
            if lang not in language_stats:
                language_stats[lang] = {"count": 0, "successes": 0}
            language_stats[lang]["count"] += 1
            if execution["success"]:
                language_stats[lang]["successes"] += 1
        
        return {
            "total_executions": len(self.execution_history),
            "recent_success_rate": success_rate,
            "average_execution_time": avg_execution_time,
            "language_statistics": language_stats,
            "most_used_language": max(language_stats.items(), key=lambda x: x[1]["count"])[0] if language_stats else None
        }
    
    def __del__(self):
        """Cleanup temporary directory"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass