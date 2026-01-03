"""
Build a proper Kaggle notebook that embeds all E2E test code inline.
This avoids issues with script kernel type and secrets.
"""
import json
from pathlib import Path

def create_notebook():
    runner_code = Path("kaggle/E2ETests/runner.py").read_text()
    
    # Cell 1: Setup and run everything inline (not importing runner.py)
    # We embed the entire runner.py content as a single cell
    
    notebook = {
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": runner_code.split('\n')
            }
        ]
    }
    
    # Properly format source as list of lines with newlines
    for cell in notebook["cells"]:
        if isinstance(cell["source"], list):
            cell["source"] = [line + '\n' if i < len(cell["source"]) - 1 else line 
                              for i, line in enumerate(cell["source"])]
    
    Path("kaggle/E2ETests/e2e_tests.ipynb").write_text(json.dumps(notebook, indent=2, ensure_ascii=False))
    print("Notebook created: kaggle/E2ETests/e2e_tests.ipynb")

if __name__ == "__main__":
    create_notebook()
