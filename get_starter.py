import re

with open('bundle.js', 'r', encoding='utf-8') as f:
    text = f.read()

# Let's search for "pip install kafka-python requests"
idx = text.find('pip install kafka-python requests')
if idx != -1:
    snippet = text[max(0, idx - 4000):idx + 1000]
    # Look for python code inside template strings or string literals
    print("Found around index", idx)
    with open('starter_snippet.txt', 'w', encoding='utf-8') as sf:
        sf.write(snippet)
    print("Wrote starter_snippet.txt")
