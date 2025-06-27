"""
ast_chunker.py

This module defines ASTCodeChunker which:
- Parses Python code into AST
- Extracts functions and classes
- Includes their docstrings in the embedding text
- Returns them as LangChain Documents

Author: Your Name
"""

import ast
from typing import List
from langchain.schema import Document

class ASTCodeChunker:
    """
    ASTCodeChunker parses Python source code and extracts
    functions and classes as logical chunks, including their docstrings.

    This enables semantically rich and well-bounded embeddings.
    """

    def chunk_code(self, code: str, file_path: str) -> List[Document]:
        """
        Splits the given Python code into AST-aware chunks.

        Args:
            code (str): Raw Python source code.
            file_path (str): Path to the file (for metadata).

        Returns:
            List[Document]: LangChain Documents with code chunks and metadata.
        """
        try:
            # Parse source into an AST
            tree = ast.parse(code)
        except SyntaxError:
            # Fallback: store entire code if it can't be parsed
            return [Document(page_content=code, metadata={"file_path": file_path})]

        code_lines = code.splitlines()
        chunks = []

        # Walk through all AST nodes
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                start_lineno = node.lineno - 1
                end_lineno = getattr(node, 'end_lineno', None)
                if end_lineno is None:
                    end_lineno = start_lineno + 1

                # Extract code lines for this node
                body_lines = code_lines[start_lineno:end_lineno]
                body = "\n".join(body_lines)

                # Get associated docstring
                docstring = ast.get_docstring(node)
                if docstring:
                    combined = f'"""\n{docstring}\n"""\n{body}'
                else:
                    combined = body

                # Add as a Document with metadata
                chunks.append(Document(page_content=combined, metadata={"file_path": file_path}))
        return chunks
