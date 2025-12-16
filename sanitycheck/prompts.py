"""
Prompt set for the competitive programming problem generation task.

Per protocol (Section 7):
- 10 distinct prompts
- Slight wording variations
- All ask for the same task (Codeforces Div-2 B style problem)
- Fixed across all conditions
"""

PROBLEM_GENERATION_PROMPTS = [
    # Prompt 0: Base template from protocol
    """Generate a competitive programming problem suitable for Codeforces Div-2 B.
Include:
• problem statement
• input format
• output format
• constraints
• short solution outline.""",
    
    # Prompt 1: Slightly different phrasing
    """Create a competitive programming problem at Codeforces Division 2 difficulty, problem B level.
The problem should include:
• A clear problem statement
• Input format specification
• Output format specification
• Constraint details
• Brief solution approach.""",
    
    # Prompt 2: More directive tone
    """Design a Codeforces Div-2 B difficulty problem.
Provide:
• Problem description
• Input specification
• Output specification
• Constraints
• Solution outline.""",
    
    # Prompt 3: Emphasize completeness
    """Write a complete competitive programming problem for Codeforces Division 2, problem B.
Must contain:
• Full problem statement
• Input format
• Output format
• All constraints
• Short solution strategy.""",
    
    # Prompt 4: Alternative structure
    """Compose a Codeforces-style programming problem, Div-2 B difficulty.
Include these sections:
• Statement of the problem
• Input requirements
• Output requirements
• Constraint bounds
• Brief solving approach.""",
    
    # Prompt 5: Focus on completeness
    """Generate a programming contest problem suitable for Codeforces Division 2, position B.
Ensure it has:
• Problem statement
• Input format
• Output format
• Constraints
• Solution outline.""",
    
    # Prompt 6: Slightly formal
    """Construct a competitive programming problem matching Codeforces Div-2 B specifications.
Requirements:
• Problem statement
• Input format
• Output format
• Constraints
• Solution sketch.""",
    
    # Prompt 7: Emphasize structure
    """Create a structured competitive programming problem for Codeforces Division 2, level B.
Components needed:
• Problem description
• Input specification
• Output specification
• Constraint details
• Solution approach.""",
    
    # Prompt 8: Direct and concise
    """Write a Codeforces Div-2 B problem with:
• Problem statement
• Input format
• Output format
• Constraints
• Solution outline.""",
    
    # Prompt 9: Slightly varied emphasis
    """Develop a competitive programming problem at Codeforces Div-2 B difficulty.
Include:
• Detailed problem statement
• Input format
• Output format
• All constraints
• Brief solution method.""",
]

# Validate we have exactly 10 prompts
assert len(PROBLEM_GENERATION_PROMPTS) == 10, "Must have exactly 10 prompts"

def get_prompt(idx: int) -> str:
    """Get a prompt by index (0-9)."""
    if not 0 <= idx < len(PROBLEM_GENERATION_PROMPTS):
        raise ValueError(f"Prompt index must be 0-9, got {idx}")
    return PROBLEM_GENERATION_PROMPTS[idx]

def get_all_prompts() -> list[str]:
    """Get all prompts as a list."""
    return PROBLEM_GENERATION_PROMPTS.copy()

