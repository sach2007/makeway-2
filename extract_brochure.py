import sys
import io

from PyPDF2 import PdfReader
reader = PdfReader(r'C:\Users\moham\Desktop\Manthan brochure.pdf')

with open('brochure_text.md', 'w', encoding='utf-8') as f:
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            f.write(f'## PAGE {i+1}\n\n')
            f.write(text)
            f.write('\n\n')

print(f"Extracted {len(reader.pages)} pages to brochure_text.md")
