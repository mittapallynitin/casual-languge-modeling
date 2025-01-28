from typing import Any

import datasets as hf_datasets
from tqdm import tqdm


def filter_non_ascii(text: str) -> str:
    """
    Remove non-ASCII characters from text.
    """
    return ''.join(char for char in text if ord(char) < 128)
    
def clean_docstring(doc_string: str) -> str:
    """
    Preprocess the documentation string
    """
    # Split the documentation into lines
    lines = doc_string.split("\n")
    processed_lines = []

    for line in lines:
        stripped_line = line.strip()
        # Stop if we encounter an empty line
        if not stripped_line:
            break
        processed_lines.append(stripped_line)
    return filter_non_ascii(". ".join(processed_lines))

def clean_code(code: str) -> str:
    """
    Normalize code indentation to PEP 8 standards:
    - Use 4 spaces per indentation level.
    - Dynamically adjust indentation levels based on leading spaces.
    - Skip empty lines for indentation calculations.
    """
    lines = code.split("\n")
    cleaned_lines = []
    current_indent_level = 0  # Track the current indentation level
    previous_spaces = 0  # Track the leading spaces of the last non-empty line

    for line in lines:
        stripped_line = line.lstrip()  # Remove leading whitespace
        leading_spaces = len(line) - len(stripped_line)  # Count leading spaces

        if not stripped_line:  # If the line is empty
            cleaned_lines.append("")  # Preserve it as a blank line
            continue  # Skip further processing for this line

        # Compare leading spaces with the previous meaningful line
        if leading_spaces > previous_spaces:
            current_indent_level += 1  # Increase indentation level
        elif leading_spaces < previous_spaces:
            current_indent_level = max(0, current_indent_level - 1)  # Decrease indentation level

        # Update the previous_spaces for the next comparison
        previous_spaces = leading_spaces

        # Construct the cleaned line with spaces
        cleaned_line = (" " * (current_indent_level * 4)) + stripped_line
        cleaned_lines.append(cleaned_line)

    return filter_non_ascii("\n".join(cleaned_lines))
    
def preprocess_batch(dataset: hf_datasets.Dataset)-> list[dict[str, str]]:

    filtered_data = []
    for record in tqdm(dataset):
        if record['func_documentation_string'] and record['func_code_string']:
            filtered_data.append({
                "description": clean_docstring(record['func_documentation_string']),
                "code": clean_code(record['func_code_string'])
            })
    return filtered_data

def preprocess_record(record: dict[Any, Any])-> dict[str, str] | None:
    if record['func_documentation_string'] and record['func_code_string']:
        return {
            "description": clean_docstring(record['func_documentation_string']),
            "code": clean_code(record['func_code_string'])
        }
    return None
