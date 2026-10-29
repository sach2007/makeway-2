with open('bundle.js', 'r', encoding='utf-8') as f:
    text = f.read()

idx = text.find('Python:{install:"pip install kafka-python requests"')
print("Python dict at:", idx)

# Search backwards for "o=`"
idx_o = text.rfind('o=`', 0, idx)
print("o=` at:", idx_o)
if idx_o != -1:
    end_o = text.find('`', idx_o + 3)
    py_code = text[idx_o + 3:end_o]
    with open('official_starter.py', 'w', encoding='utf-8') as f:
        f.write(py_code)
    print("Extracted official_starter.py successfully!")
    print(py_code[:1500])
