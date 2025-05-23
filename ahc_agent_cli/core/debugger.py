"""
Implementation debugger module for AHCAgent CLI.

This module provides functionality for debugging and testing C++ implementations.
"""

import os
import re
import logging
import tempfile
from typing import Dict, Any, Optional, List, Tuple

from ..utils.llm import LLMClient
from ..utils.docker_manager import DockerManager
from ..utils.file_io import read_file, write_file, ensure_directory

logger = logging.getLogger(__name__)

class ImplementationDebugger:
    """
    Debugger for C++ implementations.
    """
    
    def __init__(self, llm_client: LLMClient, docker_manager: DockerManager, config: Dict[str, Any] = None):
        """
        Initialize the implementation debugger.
        
        Args:
            llm_client: LLM client for code analysis and debugging
            docker_manager: Docker manager for code compilation and execution
            config: Configuration dictionary
        """
        self.llm_client = llm_client
        self.docker_manager = docker_manager
        self.config = config or {}
        
        # Get C++ compilation settings from config or use defaults
        self.cpp_compiler = self.config.get("cpp_compiler", "g++")
        self.cpp_flags = self.config.get("cpp_flags", "-std=c++17 -O2 -Wall")
        self.execution_timeout = self.config.get("execution_timeout", 10)
        
        logger.info("Initialized implementation debugger")
    
    async def compile_and_test(self, code: str, test_input: str, work_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Compile and test a C++ implementation.
        
        Args:
            code: C++ code
            test_input: Test input data
            work_dir: Working directory
            
        Returns:
            Dictionary with test results
        """
        logger.info("Compiling and testing C++ implementation")
        
        # Ensure work directory exists
        if work_dir:
            work_dir = ensure_directory(work_dir)
        else:
            work_dir = tempfile.mkdtemp(prefix="ahc_debug_")
        
        try:
            # Write code to file
            source_file = "solution.cpp"
            source_path = os.path.join(work_dir, source_file)
            write_file(source_path, code)
            
            # Write test input to file
            input_file = "input.txt"
            input_path = os.path.join(work_dir, input_file)
            write_file(input_path, test_input)
            
            # Compile code
            compile_result = self.docker_manager.compile_cpp(source_file, work_dir)
            
            if not compile_result["success"]:
                logger.error(f"Compilation failed: {compile_result['stderr']}")
                
                # Get compilation errors
                compilation_errors = compile_result["stderr"]
                
                # Fix compilation errors using LLM
                fixed_code = await self._fix_compilation_errors(code, compilation_errors)
                
                # Try compiling again with fixed code
                if fixed_code != code:
                    logger.info("Trying to compile with fixed code")
                    
                    # Write fixed code to file
                    write_file(source_path, fixed_code)
                    
                    # Compile fixed code
                    compile_result = self.docker_manager.compile_cpp(source_file, work_dir)
                    
                    if compile_result["success"]:
                        logger.info("Compilation successful with fixed code")
                        code = fixed_code
                    else:
                        logger.error(f"Compilation still failed with fixed code: {compile_result['stderr']}")
                        return {
                            "success": False,
                            "compilation_success": False,
                            "compilation_errors": compile_result["stderr"],
                            "execution_success": False,
                            "execution_output": "",
                            "execution_errors": "",
                            "execution_time": None,
                            "fixed_code": fixed_code
                        }
                else:
                    return {
                        "success": False,
                        "compilation_success": False,
                        "compilation_errors": compilation_errors,
                        "execution_success": False,
                        "execution_output": "",
                        "execution_errors": "",
                        "execution_time": None,
                        "fixed_code": None
                    }
            
            # Get executable path
            executable_file = os.path.splitext(source_file)[0]
            
            # Read test input
            with open(input_path, "r") as f:
                input_data = f.read()
            
            # Run executable with test input
            run_result = self.docker_manager.run_cpp(executable_file, work_dir, input_data, self.execution_timeout)
            
            if not run_result["success"]:
                logger.error(f"Execution failed: {run_result['stderr']}")
                
                # Get runtime errors
                runtime_errors = run_result["stderr"]
                
                # Fix runtime errors using LLM
                fixed_code = await self._fix_runtime_errors(code, runtime_errors, test_input)
                
                return {
                    "success": False,
                    "compilation_success": True,
                    "compilation_errors": "",
                    "execution_success": False,
                    "execution_output": run_result["stdout"],
                    "execution_errors": runtime_errors,
                    "execution_time": run_result.get("execution_time"),
                    "fixed_code": fixed_code
                }
            
            logger.info("Compilation and execution successful")
            
            return {
                "success": True,
                "compilation_success": True,
                "compilation_errors": "",
                "execution_success": True,
                "execution_output": run_result["stdout"],
                "execution_errors": run_result["stderr"],
                "execution_time": run_result.get("execution_time"),
                "fixed_code": None
            }
        
        except Exception as e:
            logger.error(f"Error in compile_and_test: {str(e)}")
            return {
                "success": False,
                "compilation_success": False,
                "compilation_errors": str(e),
                "execution_success": False,
                "execution_output": "",
                "execution_errors": str(e),
                "execution_time": None,
                "fixed_code": None
            }
    
    async def evaluate_solution(self, code: str, test_cases: List[Dict[str, Any]], work_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate a solution on multiple test cases.
        
        Args:
            code: C++ code
            test_cases: List of test cases, each with "input" and "expected_output" (optional)
            work_dir: Working directory
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating solution on {len(test_cases)} test cases")
        
        # Ensure work directory exists
        if work_dir:
            work_dir = ensure_directory(work_dir)
        else:
            work_dir = tempfile.mkdtemp(prefix="ahc_eval_")
        
        # Results for each test case
        test_results = []
        
        # Overall success flag
        overall_success = True
        
        # Total score
        total_score = 0
        
        try:
            # Write code to file
            source_file = "solution.cpp"
            source_path = os.path.join(work_dir, source_file)
            write_file(source_path, code)
            
            # Compile code
            compile_result = self.docker_manager.compile_cpp(source_file, work_dir)
            
            if not compile_result["success"]:
                logger.error(f"Compilation failed: {compile_result['stderr']}")
                
                # Get compilation errors
                compilation_errors = compile_result["stderr"]
                
                # Fix compilation errors using LLM
                fixed_code = await self._fix_compilation_errors(code, compilation_errors)
                
                # Try compiling again with fixed code
                if fixed_code != code:
                    logger.info("Trying to compile with fixed code")
                    
                    # Write fixed code to file
                    write_file(source_path, fixed_code)
                    
                    # Compile fixed code
                    compile_result = self.docker_manager.compile_cpp(source_file, work_dir)
                    
                    if compile_result["success"]:
                        logger.info("Compilation successful with fixed code")
                        code = fixed_code
                    else:
                        logger.error(f"Compilation still failed with fixed code: {compile_result['stderr']}")
                        return {
                            "success": False,
                            "compilation_success": False,
                            "compilation_errors": compile_result["stderr"],
                            "test_results": [],
                            "total_score": 0,
                            "average_score": 0,
                            "fixed_code": fixed_code
                        }
                else:
                    return {
                        "success": False,
                        "compilation_success": False,
                        "compilation_errors": compilation_errors,
                        "test_results": [],
                        "total_score": 0,
                        "average_score": 0,
                        "fixed_code": None
                    }
            
            # Get executable path
            executable_file = os.path.splitext(source_file)[0]
            
            # Run each test case
            for i, test_case in enumerate(test_cases):
                logger.info(f"Running test case {i+1}/{len(test_cases)}")
                
                # Get test input
                test_input = test_case["input"]
                
                # Write test input to file
                input_file = f"input_{i}.txt"
                input_path = os.path.join(work_dir, input_file)
                write_file(input_path, test_input)
                
                # Run executable with test input
                run_result = self.docker_manager.run_cpp(executable_file, work_dir, test_input, self.execution_timeout)
                
                # Check if execution was successful
                if not run_result["success"]:
                    logger.error(f"Execution failed for test case {i+1}: {run_result['stderr']}")
                    overall_success = False
                    
                    test_results.append({
                        "test_case_id": i,
                        "success": False,
                        "execution_output": run_result["stdout"],
                        "execution_errors": run_result["stderr"],
                        "execution_time": run_result.get("execution_time"),
                        "score": 0
                    })
                    
                    continue
                
                # Get execution output
                execution_output = run_result["stdout"]
                
                # Calculate score if score_function is provided in test case
                score = 0
                if "score_function" in test_case and callable(test_case["score_function"]):
                    try:
                        score = test_case["score_function"](execution_output)
                    except Exception as e:
                        logger.error(f"Error calculating score for test case {i+1}: {str(e)}")
                        score = 0
                
                # Add to total score
                total_score += score
                
                # Add test result
                test_results.append({
                    "test_case_id": i,
                    "success": True,
                    "execution_output": execution_output,
                    "execution_errors": run_result["stderr"],
                    "execution_time": run_result.get("execution_time"),
                    "score": score
                })
            
            # Calculate average score
            average_score = total_score / len(test_cases) if test_cases else 0
            
            logger.info(f"Evaluation complete: {len(test_results)} test cases, total score = {total_score}, average score = {average_score}")
            
            return {
                "success": overall_success,
                "compilation_success": True,
                "compilation_errors": "",
                "test_results": test_results,
                "total_score": total_score,
                "average_score": average_score,
                "fixed_code": None
            }
        
        except Exception as e:
            logger.error(f"Error in evaluate_solution: {str(e)}")
            return {
                "success": False,
                "compilation_success": False,
                "compilation_errors": str(e),
                "test_results": [],
                "total_score": 0,
                "average_score": 0,
                "fixed_code": None
            }
    
    async def _fix_compilation_errors(self, code: str, compilation_errors: str) -> str:
        """
        Fix compilation errors using LLM.
        
        Args:
            code: Original C++ code
            compilation_errors: Compilation error messages
            
        Returns:
            Fixed C++ code
        """
        logger.info("Attempting to fix compilation errors")
        
        prompt = f"""
        You are an expert C++ programmer. Please fix the compilation errors in the following code:
        
        ```cpp
        {code}
        ```
        
        Compilation errors:
        ```
        {compilation_errors}
        ```
        
        Please provide the fixed code. Make minimal changes to fix the errors.
        Return only the fixed C++ code without any explanations.
        ```cpp
        // Your fixed code here
        ```
        """
        
        try:
            response = await self.llm_client.generate(prompt)
            
            # Extract code from response
            fixed_code = self._extract_code(response)
            
            return fixed_code
        
        except Exception as e:
            logger.error(f"Error fixing compilation errors: {str(e)}")
            return code
    
    async def _fix_runtime_errors(self, code: str, runtime_errors: str, test_input: str) -> str:
        """
        Fix runtime errors using LLM.
        
        Args:
            code: Original C++ code
            runtime_errors: Runtime error messages
            test_input: Test input data
            
        Returns:
            Fixed C++ code
        """
        logger.info("Attempting to fix runtime errors")
        
        prompt = f"""
        You are an expert C++ programmer. Please fix the runtime errors in the following code:
        
        ```cpp
        {code}
        ```
        
        Runtime errors:
        ```
        {runtime_errors}
        ```
        
        Test input:
        ```
        {test_input}
        ```
        
        Please provide the fixed code. Make minimal changes to fix the errors.
        Return only the fixed C++ code without any explanations.
        ```cpp
        // Your fixed code here
        ```
        """
        
        try:
            response = await self.llm_client.generate(prompt)
            
            # Extract code from response
            fixed_code = self._extract_code(response)
            
            return fixed_code
        
        except Exception as e:
            logger.error(f"Error fixing runtime errors: {str(e)}")
            return code
    
    def _extract_code(self, response: str) -> str:
        """
        Extract code from LLM response.
        
        Args:
            response: LLM response
            
        Returns:
            Extracted code
        """
        # Try to extract code between ```cpp and ``` markers
        import re
        code_match = re.search(r'```cpp\s*(.*?)\s*```', response, re.DOTALL)
        
        if code_match:
            return code_match.group(1).strip()
        
        # If no markers found, return the entire response
        return response.strip()
