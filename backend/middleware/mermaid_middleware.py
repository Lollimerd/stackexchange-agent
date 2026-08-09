"""
Mermaid Validation Middleware
------------------------------
Intercepts agent responses containing Mermaid code blocks, validates syntax
server-side, and prompts the model to regenerate the diagram if errors are found.
The middleware uses the `aafter_model` hook to inspect the latest AI message,
extract Mermaid blocks, run lightweight regex-based validation, and inject a
correction HumanMessage + `jump_to: "model"` to trigger a retry loop.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage
from langchain.agents.middleware.types import AgentMiddleware, AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Valid first-token diagram types supported by Mermaid 10+
_VALID_DIAGRAM_TYPES = {
    "graph",
    "flowchart",
    "sequencediagram",
    "classDiagram",
    "statediagram",
    "statediagram-v2",
    "erdiagram",
    "journey",
    "gantt",
    "pie",
    "quadrantchart",
    "requirementdiagram",
    "gitgraph",
    "mindmap",
    "timeline",
    "xychart-beta",
    "block-beta",
    "packet-beta",
    "kanban",
    "architecture-beta",
}

# Mermaid reserved words that must NOT be used as bare node IDs
_RESERVED_WORDS = {
    "graph",
    "subgraph",
    "end",
    "style",
    "classDef",
    "click",
    "call",
    "href",
    "linkStyle",
    "class",
    "direction",
}

# Default max retries to prevent infinite loops
_DEFAULT_MAX_RETRIES = 2

# State key used to track how many times the middleware has already retried
_RETRY_COUNT_KEY = "mermaid_retry_count"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _extract_mermaid_blocks(text: str) -> list[str]:
    """Extract all ```mermaid ... ``` code blocks from *text*."""
    pattern = r"```mermaid\s+(.*?)\s*```"
    return re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)


def _validate_mermaid_block(code: str) -> list[str]:
    """
    Perform lightweight server-side Mermaid syntax validation.

    Returns a list of human-readable error strings. An empty list means
    the block passed all checks.
    """
    errors: list[str] = []
    lines = [ln.strip() for ln in code.strip().splitlines() if ln.strip()]

    if not lines:
        errors.append("Mermaid block is empty.")
        return errors

    # -----------------------------------------------------------------
    # 1. Check that the first token is a recognised diagram type
    # -----------------------------------------------------------------
    first_line = lines[0]
    first_token = first_line.split()[0].lower().rstrip(":")
    if first_token not in {d.lower() for d in _VALID_DIAGRAM_TYPES}:
        errors.append(
            f"Unknown or missing diagram type on the first line: '{first_line}'. "
            f"Expected one of: flowchart, sequenceDiagram, classDiagram, etc."
        )

    # -----------------------------------------------------------------
    # 2. Detect reserved words used as bare node IDs
    #    Pattern: bare word immediately followed by [ or { or ( or >
    # -----------------------------------------------------------------
    reserved_node_pattern = re.compile(
        r"\b(" + "|".join(re.escape(w) for w in _RESERVED_WORDS) + r")\s*[\[\({>]",
        re.IGNORECASE,
    )
    for i, line in enumerate(lines[1:], start=2):
        match = reserved_node_pattern.search(line)
        if match:
            errors.append(
                f"Line {i}: Reserved word '{match.group(1)}' used as a node ID. "
                "Use a different alphanumeric identifier instead."
            )

    # -----------------------------------------------------------------
    # 3. Check for unclosed brackets / parentheses / braces
    # -----------------------------------------------------------------
    bracket_map = {"[": "]", "(": ")", "{": "}"}
    stack: list[tuple[str, int]] = []
    for i, line in enumerate(lines, start=1):
        # Skip comment lines
        if line.startswith("%%"):
            continue
        for ch in line:
            if ch in bracket_map:
                stack.append((ch, i))
            elif ch in bracket_map.values():
                if stack and bracket_map[stack[-1][0]] == ch:
                    stack.pop()
                # Mismatched close bracket — skip to avoid false positives
    if stack:
        locs = ", ".join(f"line {ln}" for _, ln in stack)
        errors.append(f"Unclosed bracket(s) found at: {locs}.")

    # -----------------------------------------------------------------
    # 4. Spaces / hyphens / special chars in node IDs (flowchart/graph)
    # -----------------------------------------------------------------
    if first_token in {"graph", "flowchart"}:
        # Node ID pattern: word before [ or { or ( or >
        # A valid ID should be purely alphanumeric (no spaces, hyphens allowed
        # only inside quoted strings, but not in bare IDs).
        bad_id_pattern = re.compile(
            r"\b([A-Za-z0-9_]+(?:[\s\-][A-Za-z0-9_]+)+)\s*[\[\({>]"
        )
        for i, line in enumerate(lines[1:], start=2):
            match = bad_id_pattern.search(line)
            if match:
                errors.append(
                    f"Line {i}: Node ID '{match.group(1)}' contains spaces or hyphens. "
                    "Node IDs must be a single alphanumeric word (e.g. 'MyNode')."
                )

    # -----------------------------------------------------------------
    # 5. Check that descriptive node labels are wrapped in double-quotes
    #    e.g.  NodeA[This is bad]  vs  NodeA["This is good"]
    # -----------------------------------------------------------------
    # Only enforce for flowchart/graph diagrams
    if first_token in {"graph", "flowchart"}:
        unquoted_label_pattern = re.compile(
            r'\[(?!")([^\]]*\s[^\]]*)\]'  # [text with spaces] but NOT ["..."]
        )
        for i, line in enumerate(lines[1:], start=2):
            if line.startswith("%%"):
                continue
            match = unquoted_label_pattern.search(line)
            if match:
                errors.append(
                    f"Line {i}: Node label '{match.group(1)}' contains spaces but is "
                    "not quoted. Wrap multi-word labels in double quotes: "
                    f'["{match.group(1)}"].'
                )

    return errors


# ---------------------------------------------------------------------------
# Middleware class
# ---------------------------------------------------------------------------
class MermaidValidationMiddleware(AgentMiddleware):
    """
    Validates Mermaid diagram syntax in every AI response and prompts the
    model to regenerate if errors are found.

    Parameters
    ----------
    max_retries:
        Maximum number of regeneration attempts before the middleware gives
        up and passes the (possibly broken) response through.  Defaults to 2.
    """

    def __init__(self, max_retries: int = _DEFAULT_MAX_RETRIES) -> None:
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # Async hook – runs after the model produces a response
    # ------------------------------------------------------------------

    async def aafter_model(
        self, state: AgentState, runtime: Any
    ) -> dict[str, Any] | None:
        """
        Inspect the latest AI message for Mermaid blocks.

        If any block contains syntax errors AND the retry budget has not been
        exhausted, inject a correction HumanMessage and return
        ``jump_to="model"`` so the agent loops back to the model node.
        """
        messages = state.get("messages", [])
        if not messages:
            return None

        # The latest message should be the AI response
        latest = messages[-1]
        content: str = (
            latest.content
            if hasattr(latest, "content") and isinstance(latest.content, str)
            else ""
        )

        if not content:
            return None

        # Extract Mermaid blocks
        blocks = _extract_mermaid_blocks(content)
        if not blocks:
            return None  # No Mermaid diagram – nothing to validate

        # Collect all errors across all blocks
        all_errors: list[str] = []
        for idx, block in enumerate(blocks, start=1):
            block_errors = _validate_mermaid_block(block)
            if block_errors:
                prefix = f"[Diagram {idx}] " if len(blocks) > 1 else ""
                all_errors.extend(f"{prefix}{err}" for err in block_errors)

        if not all_errors:
            logger.info("MermaidValidationMiddleware: all Mermaid blocks are valid.")
            return None

        # Check retry budget
        retry_count: int = state.get(_RETRY_COUNT_KEY, 0)
        if retry_count >= self.max_retries:
            logger.warning(
                "MermaidValidationMiddleware: max retries (%d) reached, "
                "passing through response with %d Mermaid error(s).",
                self.max_retries,
                len(all_errors),
            )
            return None  # Give up – let the broken diagram through

        error_summary = "\n".join(f"  - {e}" for e in all_errors)
        correction_prompt = (
            "[SYSTEM: Mermaid Diagram Validation Failed]\n"
            "The Mermaid diagram(s) in your previous response contain syntax "
            "errors that will prevent them from rendering correctly:\n"
            f"{error_summary}\n\n"
            "Please regenerate the diagram(s) fixing the above issues. "
            "Remember:\n"
            "  • Node IDs must be single alphanumeric words (no spaces/hyphens).\n"
            '  • Wrap multi-word node labels in double-quotes: ["My Label"].\n'
            "  • Do NOT use reserved words (graph, subgraph, end, style, "
            "classDef) as node IDs.\n"
            "  • Ensure every opening bracket has a matching closing bracket.\n"
            "  • Start the diagram with a valid type: flowchart, "
            "sequenceDiagram, classDiagram, etc."
        )

        logger.info(
            "MermaidValidationMiddleware: found %d error(s) (retry %d/%d), "
            "injecting correction prompt.",
            len(all_errors),
            retry_count + 1,
            self.max_retries,
        )

        return {
            "messages": [HumanMessage(content=correction_prompt)],
            _RETRY_COUNT_KEY: retry_count + 1,
            "jump_to": "model",
        }
