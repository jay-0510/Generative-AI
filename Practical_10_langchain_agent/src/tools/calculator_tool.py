"""
calculator_tool.py
--------------------
A calculator tool for the agent. Deliberately does NOT use Python's raw
`eval()` — passing LLM-generated strings straight to eval() is a classic
arbitrary-code-execution risk, since a prompt-injected or malformed action
input could contain arbitrary Python, not just arithmetic. Instead we parse
the expression into an AST and only evaluate a whitelisted set of numeric
operators, so the worst a bad input can do is fail to parse.
"""

import ast
import operator

from langchain_core.tools import tool

# Whitelist of AST node types this calculator is willing to evaluate.
# Anything else (function calls, attribute access, imports, etc.) is rejected.
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluates an AST node, rejecting anything not in the whitelist."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


@tool
def calculator_tool(expression: str) -> str:
    """
    Evaluates a basic arithmetic expression (+, -, *, /, %, **) and returns
    the numeric result as a string. Use this for any math the question
    requires instead of guessing the answer yourself.

    Example input: "12 * (4 + 3)"
    """
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _safe_eval(parsed.body)
        return str(result)
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError) as exc:
        # Returned as a string, not raised, so the agent sees it as a tool
        # Observation and can recover (e.g. ask the user to clarify) instead
        # of crashing the whole run.
        return f"Error: could not evaluate '{expression}' ({exc})"
