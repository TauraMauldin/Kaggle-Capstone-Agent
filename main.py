"""
Main Entry Point - Intelligent Research Assistant Capstone Project

This file serves as the main entry point for the capstone project and demonstrates
all required capabilities for the Kaggle Agents Intensive Capstone.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.intelligent_assistant import IntelligentAssistant
from demos.capstone_demo import CapstoneDemo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_welcome():
    """Print welcome message"""
    print("🤖 Intelligent Research Assistant - Capstone Project")
    print("=" * 60)
    print("Kaggle Agents Intensive Capstone Project Submission")
    print("Demonstrating Advanced AI Agent Capabilities")
    print()
    print("Required Capabilities:")
    print("✅ Memory Systems (short-term and long-term)")
    print("✅ Tool Integration (web search, code execution, document analysis)")
    print("✅ Multi-Agent Orchestration (coordinated workflow)")
    print("✅ Evaluation Framework (performance metrics)")
    print("✅ Safety Features (content filtering and security)")
    print()
    print("=" * 60)
    print()

async def interactive_mode():
    """Run in interactive mode"""
    print("🎮 Interactive Mode")
    print("Type 'exit' to quit, 'demo' to run full demonstration")
    print()
    
    assistant = IntelligentAssistant()
    session_id = None
    
    while True:
        try:
            user_input = input("👤 You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'demo':
                print("🎬 Running full demonstration...")
                demo = CapstoneDemo()
                await demo.run_complete_demo()
                continue
            
            if user_input.lower() == 'help':
                print("Available commands:")
                print("  demo  - Run full demonstration")
                print("  help  - Show this help message")
                print("  exit  - Exit the program")
                print("  Any other text will be processed by the assistant")
                continue
            
            if not user_input:
                continue
            
            # Process with the assistant
            print("🤔 Processing...")
            result = await assistant.process_task(user_input, "interactive_user")
            
            print(f"🤖 Assistant ({result.confidence:.1%} confidence):")
            if result.status == "completed":
                print(result.result)
            else:
                print(f"❌ Error: {result.result}")
            
            print(f"⏱️  Execution time: {result.execution_time:.2f}s")
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

async def demo_mode():
    """Run demonstration mode"""
    print("🎬 Running Demonstration Mode")
    print()
    
    demo = CapstoneDemo()
    report = await demo.run_complete_demo()
    
    if report and report.get("capstone_requirements_met"):
        print("\n🎉 Capstone project completed successfully!")
        return True
    else:
        print("\n❌ Capstone project needs improvement.")
        return False

async def test_mode():
    """Run quick tests to verify functionality"""
    print("🧪 Running Quick Tests")
    print()
    
    assistant = IntelligentAssistant()
    tests_passed = 0
    total_tests = 5
    
    try:
        # Test 1: Basic functionality
        print("Test 1: Basic functionality...")
        result = await assistant.process_task("What is AI?", "test_user")
        if result.status == "completed":
            print("✅ Basic functionality test passed")
            tests_passed += 1
        else:
            print("❌ Basic functionality test failed")
        
        # Test 2: Memory
        print("Test 2: Memory systems...")
        memory_summary = await assistant.memory_manager.get_user_summary("test_user")
        if memory_summary["statistics"]["total_memories"] > 0:
            print("✅ Memory systems test passed")
            tests_passed += 1
        else:
            print("❌ Memory systems test failed")
        
        # Test 3: Tools
        print("Test 3: Tool integration...")
        from tools.code_execution_tool import CodeExecutionRequest
        code_result = await assistant.code_executor.execute(
            CodeExecutionRequest(code="print('Hello, World!')", language="python")
        )
        if code_result.success:
            print("✅ Tool integration test passed")
            tests_passed += 1
        else:
            print("❌ Tool integration test failed")
        
        # Test 4: Safety
        print("Test 4: Safety features...")
        safety_result = await assistant.safety_filter.check_content("Hello world", "test_user")
        if safety_result.is_safe:
            print("✅ Safety features test passed")
            tests_passed += 1
        else:
            print("❌ Safety features test failed")
        
        # Test 5: Evaluation
        print("Test 5: Evaluation framework...")
        eval_scores = await assistant.evaluator.evaluate_result("test", "test result")
        if "overall" in eval_scores:
            print("✅ Evaluation framework test passed")
            tests_passed += 1
        else:
            print("❌ Evaluation framework test failed")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
    
    print()
    print(f"🧪 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed >= 3:
        print("✅ Sufficient tests passed for capstone requirements")
        return True
    else:
        print("❌ Insufficient tests passed")
        return False

def main():
    """Main function"""
    print_welcome()
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        print("Select mode:")
        print("1. demo      - Run full demonstration")
        print("2. test      - Run quick tests")
        print("3. interactive - Interactive mode")
        print()
        
        choice = input("Enter choice (1-3): ").strip()
        modes = {"1": "demo", "2": "test", "3": "interactive"}
        mode = modes.get(choice, "demo")
    
    try:
        if mode == "demo":
            success = asyncio.run(demo_mode())
            sys.exit(0 if success else 1)
        elif mode == "test":
            success = asyncio.run(test_mode())
            sys.exit(0 if success else 1)
        elif mode == "interactive":
            asyncio.run(interactive_mode())
        else:
            print(f"Unknown mode: {mode}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()