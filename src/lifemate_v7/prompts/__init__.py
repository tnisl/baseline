"""Prompts are stored as text assets, separately from execution code."""
from pathlib import Path

PLANNER_SYSTEM = Path(__file__).with_name('planner.txt').read_text(encoding='utf-8')
CALL_SYSTEM = Path(__file__).with_name('caller.txt').read_text(encoding='utf-8')
SUMMARIZER_SYSTEM = Path(__file__).with_name('summarizer.txt').read_text(encoding='utf-8')
