"""
HumanEval dataset loader and evaluation.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class HumanEvalProblem:
    """A single HumanEval problem."""
    task_id: str
    prompt: str
    canonical_solution: str
    test: str
    entry_point: str
    

def load_humaneval(n_samples: Optional[int] = None) -> List[HumanEvalProblem]:
    """
    Load HumanEval dataset from HuggingFace.
    
    Args:
        n_samples: Optional limit on number of problems to load
        
    Returns:
        List of HumanEvalProblem objects
    """
    from datasets import load_dataset
    
    dataset = load_dataset("openai_humaneval", split="test")
    
    problems = []
    for i, item in enumerate(dataset):
        if n_samples is not None and i >= n_samples:
            break
            
        problem = HumanEvalProblem(
            task_id=item["task_id"],
            prompt=item["prompt"],
            canonical_solution=item["canonical_solution"],
            test=item["test"],
            entry_point=item["entry_point"],
        )
        problems.append(problem)
    
    print(f"Loaded {len(problems)} HumanEval problems")
    return problems


def format_prompt_for_model(problem: HumanEvalProblem, model_type: str = "deepseek") -> str:
    """
    Format the problem prompt for a specific model.
    
    Args:
        problem: The HumanEval problem
        model_type: Type of model ("deepseek", "qwen", "generic")
        
    Returns:
        Formatted prompt string
    """
    if model_type == "deepseek":
        # DeepSeek-Coder instruction format
        return f"""You are an expert Python programmer. Complete the following function.

{problem.prompt}"""
    
    elif model_type == "qwen":
        # Qwen2.5-Coder instruct format
        return f"""<|im_start|>system
You are a helpful coding assistant. Complete the function below.<|im_end|>
<|im_start|>user
{problem.prompt}<|im_end|>
<|im_start|>assistant
"""
    
    elif model_type == "codellama":
        # CodeLlama instruct format
        return f"""[INST] Complete the following Python function:

{problem.prompt}
[/INST]
"""
    
    else:
        # Generic format - just the prompt (works for many models)
        return problem.prompt


def extract_function_code(generated: str, entry_point: str) -> str:
    """
    Extract the function code from generated output.
    
    Handles common patterns in LLM outputs like markdown code blocks,
    extra explanations, etc.
    """
    code = generated
    
    # Remove markdown code blocks if present
    if "```python" in code:
        code = code.split("```python")[1]
        if "```" in code:
            code = code.split("```")[0]
    elif "```" in code:
        code = code.split("```")[1]
        if "```" in code:
            code = code.split("```")[0]
    
    # Try to extract just the function if there's extra content
    lines = code.split('\n')
    func_lines = []
    in_function = False
    indent_level = None
    
    for line in lines:
        stripped = line.lstrip()
        current_indent = len(line) - len(stripped)
        
        if stripped.startswith(f"def {entry_point}"):
            in_function = True
            indent_level = current_indent
            func_lines.append(line)
        elif in_function:
            if stripped == "" or current_indent > indent_level:
                func_lines.append(line)
            elif stripped.startswith("def ") or (current_indent <= indent_level and stripped):
                # New function or dedent - end of our function
                break
            else:
                func_lines.append(line)
    
    if func_lines:
        return '\n'.join(func_lines)
    
    return code.strip()


def check_solution(problem: HumanEvalProblem, solution: str, timeout: float = 5.0) -> bool:
    """
    Check if a solution passes the test cases.
    
    Args:
        problem: The HumanEval problem
        solution: Generated solution code
        timeout: Maximum execution time in seconds
        
    Returns:
        True if solution passes all tests, False otherwise
    """
    import multiprocessing
    import sys
    from io import StringIO
    
    # Combine the prompt (which has function signature) with solution
    # and the test cases
    full_code = problem.prompt + solution + "\n" + problem.test + f"\ncheck({problem.entry_point})"
    
    def run_code(code: str, result_queue):
        try:
            # Capture stdout/stderr
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            
            exec_globals = {}
            exec(code, exec_globals)
            
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            result_queue.put(True)
        except Exception as e:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            result_queue.put(False)
    
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=run_code, args=(full_code, result_queue))
    process.start()
    process.join(timeout=timeout)
    
    if process.is_alive():
        process.terminate()
        process.join()
        return False
    
    try:
        return result_queue.get_nowait()
    except:
        return False

