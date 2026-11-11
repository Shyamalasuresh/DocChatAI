import os, tempfile
from src.loader import load_pdf
import fitz

temp_dir = tempfile.gettempdir()
pdf_path = os.path.join(temp_dir, 'temp_test.pdf')
doc = fitz.open()
page = doc.new_page()
page.insert_text((72,72), 'Hello')
doc.save(pdf_path)
doc.close()

result = load_pdf(pdf_path)
print('Result:', result)
