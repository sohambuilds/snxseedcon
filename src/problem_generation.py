"""
Problem generation utilities.

Instead of generating SOLUTIONS to problems, we generate the PROBLEMS themselves.
This is a better test of creativity since:
1. No single correct answer exists
2. Diversity is meaningful (different topics, difficulties, styles)
3. Quality can still be verified (valid syntax, solvable)
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import ast
import re


@dataclass
class GeneratedProblem:
    """A generated coding problem."""
    function_name: str
    signature: str
    docstring: str
    full_prompt: str
    test_cases: List[str]
    raw_output: str
    is_valid: bool
    validation_errors: List[str]


PROBLEM_GENERATION_PROMPT = '''You are a creative programming instructor. Generate a novel Python coding problem.

Requirements:
1. Create a function with a clear name and type hints
2. Write a detailed docstring with:
   - Description of what the function should do
   - At least 2 example test cases with >>> syntax
3. The problem should be solvable in 10-20 lines of code
4. Be creative - don't just copy standard problems

Output format:
```python
def function_name(param1: type1, param2: type2) -> return_type:
    """
    Description of the problem.
    
    >>> function_name(example_input)
    expected_output
    >>> function_name(another_input)
    another_output
    """
    pass
```

Generate a unique and interesting problem:'''


TOPIC_PROMPTS = {
    "string": "Generate a creative string manipulation problem:",
    "list": "Generate a creative list/array processing problem:",
    "math": "Generate a creative mathematical computation problem:",
    "recursion": "Generate a problem that requires recursive thinking:",
    "dictionary": "Generate a problem involving dictionaries/hashmaps:",
    "sorting": "Generate a creative sorting or ordering problem:",
    "game": "Generate a problem simulating a simple game or puzzle:",
    "text": "Generate a text processing or parsing problem:",
    "geometry": "Generate a computational geometry problem:",
    "simulation": "Generate a simulation or state-machine problem:",
}


def get_generation_prompt(topic: Optional[str] = None) -> str:
    """Get a prompt for problem generation, optionally with a topic."""
    base = PROBLEM_GENERATION_PROMPT
    if topic and topic in TOPIC_PROMPTS:
        base = base.replace(
            "Generate a unique and interesting problem:",
            TOPIC_PROMPTS[topic]
        )
    return base


def extract_problem_from_output(output: str) -> Tuple[str, List[str]]:
    """
    Extract the function definition and test cases from generated output.
    
    Returns:
        (function_code, list_of_test_cases)
    """
    # Try to find code block
    code = output
    if "```python" in output:
        match = re.search(r"```python\s*(.*?)\s*```", output, re.DOTALL)
        if match:
            code = match.group(1)
    elif "```" in output:
        match = re.search(r"```\s*(.*?)\s*```", output, re.DOTALL)
        if match:
            code = match.group(1)
    
    # Extract test cases from docstring
    test_cases = re.findall(r">>> (.+)", code)
    
    return code.strip(), test_cases


def validate_problem(code: str) -> Tuple[bool, List[str]]:
    """
    Validate a generated problem.
    
    Checks:
    1. Valid Python syntax
    2. Contains a function definition
    3. Has a docstring
    4. Has at least one test case
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    # Check syntax
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"Syntax error: {e.msg}"]
    
    # Find function definition
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    if not functions:
        errors.append("No function definition found")
        return False, errors
    
    func = functions[0]
    
    # Check for docstring
    if not (func.body and isinstance(func.body[0], ast.Expr) and 
            isinstance(func.body[0].value, (ast.Str, ast.Constant))):
        errors.append("No docstring found")
    
    # Check for test cases in docstring
    docstring = ast.get_docstring(func) or ""
    if ">>>" not in docstring:
        errors.append("No test cases (>>>) found in docstring")
    
    # Check function has parameters
    if not func.args.args:
        errors.append("Function has no parameters")
    
    return len(errors) == 0, errors


def parse_generated_problem(output: str) -> GeneratedProblem:
    """
    Parse and validate a generated problem.
    """
    code, test_cases = extract_problem_from_output(output)
    is_valid, errors = validate_problem(code)
    
    # Extract function name
    func_name = "unknown"
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                break
    except:
        pass
    
    # Extract signature and docstring
    signature = ""
    docstring = ""
    lines = code.split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            signature = line.strip()
            # Find docstring
            for j in range(i+1, len(lines)):
                if '"""' in lines[j] or "'''" in lines[j]:
                    doc_start = j
                    doc_lines = [lines[j]]
                    for k in range(j+1, len(lines)):
                        doc_lines.append(lines[k])
                        if '"""' in lines[k] or "'''" in lines[k]:
                            break
                    docstring = "\n".join(doc_lines)
                    break
            break
    
    return GeneratedProblem(
        function_name=func_name,
        signature=signature,
        docstring=docstring,
        full_prompt=code,
        test_cases=test_cases,
        raw_output=output,
        is_valid=is_valid,
        validation_errors=errors,
    )


def compute_problem_diversity(problems: List[GeneratedProblem]) -> Dict[str, float]:
    """
    Compute diversity metrics for a set of generated problems.
    
    Metrics:
    - unique_names: Fraction of unique function names
    - unique_signatures: Fraction of unique signatures
    - topic_diversity: Entropy of detected topics
    - structural_diversity: Variation in problem length/complexity
    """
    if not problems:
        return {}
    
    valid_problems = [p for p in problems if p.is_valid]
    if not valid_problems:
        return {"valid_rate": 0.0}
    
    # Unique names
    names = [p.function_name for p in valid_problems]
    unique_names = len(set(names)) / len(names)
    
    # Unique signatures
    sigs = [p.signature for p in valid_problems]
    unique_sigs = len(set(sigs)) / len(sigs)
    
    # Problem lengths (proxy for complexity)
    lengths = [len(p.full_prompt) for p in valid_problems]
    avg_length = sum(lengths) / len(lengths)
    length_std = (sum((l - avg_length)**2 for l in lengths) / len(lengths)) ** 0.5
    
    # Number of test cases
    test_counts = [len(p.test_cases) for p in valid_problems]
    avg_tests = sum(test_counts) / len(test_counts)
    
    return {
        "valid_rate": len(valid_problems) / len(problems),
        "unique_names": unique_names,
        "unique_signatures": unique_sigs,
        "avg_length": avg_length,
        "length_std": length_std,
        "avg_test_cases": avg_tests,
    }



