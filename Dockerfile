FROM arc53/docsgpt:develop
# Patch: replace broken completions API handler with modern messages API
COPY application/llm/anthropic.py /app/application/llm/anthropic.py

