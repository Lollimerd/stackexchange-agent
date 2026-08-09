from langchain_core.documents import Document
from typing import List, Any
import json, docker, re, os, socket

def escape_lucene_chars(text: str) -> str:
    """
    Escapes special characters in a string for safe use in a Lucene query.
    """
    # List of special characters in Lucene syntax
    special_chars = r'([+\-&|!(){}\[\]^"~*?:\\/])'
    # Prepend each special character with a backslash
    return re.sub(special_chars, r'\\\1', text)

# --- Dynamic Container Discovery ---
def find_container_by_port(port: int) -> str:
    """Inspects running Docker containers to find which one is using the specified port."""
    if not port:
        return "Invalid port"
    
    try:
        # Connect to the Docker daemon
        client = docker.from_env()
        containers = client.containers.list()
        target_port = str(port)

        for container in containers:
            # The .ports attribute is a dictionary like: {'7687/tcp': [{'HostIp': '0.0.0.0', 'HostPort': '7687'}]}
            port_mappings = container.ports
            for port_key, host_mappings in port_mappings.items():

                # when backend is dockerised
                if port_key.split('/')[0] == target_port:
                    return container.name
                
                # when running from bash/uvicorn
                if host_mappings:
                    for mapping in host_mappings:
                        if mapping.get("HostPort") == str(port):
                            return container.name # Found it!
        return "No matching container found"
    
    except docker.errors.DockerException:
        if os.path.exists('/.dockerenv'):
            hostname = socket.gethostname()
            return f"Self ({hostname}) - Docker socket not mounted?"
        return "Docker daemon not running or not accessible"
    except Exception as e:
        return f"An error occurred: {e}"
    
    
def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)

def _format_value_readable(value: Any, indent: int = 0, max_list_items: int = 10) -> str:
    pad = "  " * indent
    next_pad = "  " * (indent + 1)

    # Dict: pretty print as key: value with nesting
    if isinstance(value, dict):
        if not value:
            return "{}"
        lines: list[str] = []
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{pad}{k}:")
                lines.append(_format_value_readable(v, indent + 1, max_list_items))
            else:
                lines.append(f"{pad}{k}: {_format_scalar(v)}")
        return "\n".join(lines)

    # List: if primitives, inline; if complex, bullets per item
    if isinstance(value, list):
        if not value:
            return "[]"
        # Truncate long lists for readability
        sliced = value[:max_list_items]
        omitted = len(value) - len(sliced)
        if all(not isinstance(x, (dict, list)) for x in sliced):
            return f"{', '.join(_format_scalar(x) for x in sliced)}" + (f" …(+{omitted})" if omitted > 0 else "")
        lines = []
        for item in sliced:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines.append(_format_value_readable(item, indent + 1, max_list_items))
            else:
                lines.append(f"{pad}- {_format_scalar(item)}")
        if omitted > 0:
            lines.append(f"{pad}- …(+{omitted} more)")
        return "\n".join(lines)

    # Fallback scalar
    return f"{pad}{_format_scalar(value)}"


def format_docs_with_metadata(docs: list[Document]) -> str:
    """Formats documents and metadata into a Unicode-safe, human-readable string.

    - Preserves Unicode characters (no JSON escaping).
    - Prints nested metadata (dicts/lists) as readable sections and bullet lists.
    - Truncates very long lists to keep context compact.
    """
    formatted_blocks: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        # Page content (already Unicode-safe in Python 3)
        content_section = (
            "\n--------- CONTENT ---------\n"
            f"{doc.page_content}"
        )

        # Metadata as readable lines
        metadata_lines: list[str] = []
        for key, value in doc.metadata.items():
            # Section header for complex values
            if isinstance(value, (dict, list)):
                metadata_lines.append(f"{key}:")
                metadata_lines.append(_format_value_readable(value, indent=1))
            else:
                metadata_lines.append(f"{key}: {_format_scalar(value)}")
        metadata_str = "\n".join(metadata_lines)

        metadata_section = (
            "\n--------- METADATA ---------\n"
            f"{metadata_str}"
        )

        formatted_blocks.append(content_section + metadata_section)

    final_context_str = "\n\n".join(formatted_blocks)

    # Debug log (stdout) for developers; does not change returned value
    print("\n" + "=" * 100)
    print("--- 📄 RETRIEVED CONTEXT FOR LLM ---")
    print(final_context_str)
    print(f"\n--- 📊 Documents retrieved: {len(docs)} ---")
    print("=" * 100 + "\n")

    return final_context_str