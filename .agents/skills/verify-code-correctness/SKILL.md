---
name: code-correctness
version: 2
description: >
  Load when the user wants to check if their Python code is correct.
  Identifies bugs, logic errors, and edge case failures. Python only.
---

# Code Correctness Skill — Python

Simple, focused code review for Python. No fluff — just find what's wrong.

***

## When to Load

Load this skill when the user shares Python code and asks:
- "Is this correct?"
- "Does this work?"
- "What's wrong with this?"
- "Can you review this?"

***

## What to Do

1. **Read the code** and understand what it's supposed to do. If unclear, ask in one sentence before proceeding.

2. **Check for issues** in this order:
   - **Syntax errors** — missing colons, brackets, indentation
   - **Logic errors** — wrong operators, off-by-one, incorrect conditions
   - **Runtime errors** — division by zero, index out of range, KeyError, AttributeError on None
   - **Common Python traps**:
     - Mutable default argument: `def f(x=[])` — always a bug
     - `is` vs `==` for value comparison
     - `dict[key]` vs `dict.get(key)` — KeyError risk
     - Integer vs float division: `//` vs `/`
     - Modifying a list while iterating over it
     - Generator exhausted after first use

3. **Run the code** using `execute_code` if it's self-contained, to confirm whether issues are real.

***

## Output Format

Keep it concise. Use this structure:

```
VERDICT: PASS or FAIL

ISSUES:
- [Line N] <type>: <what's wrong>
  Fix: <one-line fix>

(If PASS, just say "No issues found. Code looks correct.")
```

Only list real bugs — not style preferences, not nitpicks. If there are no issues, say so clearly.

***

## Rules

- Be direct. Don't pad the response.
- Show a fix for every issue found.
- If the code's intent is unclear, ask before guessing.
- Don't rewrite the whole code unless asked — just point out the problems.