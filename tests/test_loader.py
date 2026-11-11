import os
import sys
sys.path.append(os.path.abspath('.'))
from src.loader import load_pdf


def test_load_valid_pdf(tmp_path):
    """Create a simple PDF and ensure loader extracts the text correctly."""
    import pymupdf
    pdf_path = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello DocuChat")
    doc.save(pdf_path)
    doc.close()

    result = load_pdf(str(pdf_path))
    assert len(result) == 1
    assert result[0]["text"] == "Hello DocuChat"
    assert result[0]["page_number"] == 1
    assert result[0]["document_name"] == "sample.pdf"


def test_load_invalid_pdf():
    """Pass a non‑PDF file path and expect a ValueError."""
    invalid_path = "not_a_pdf.txt"
    # create dummy file
    with open(invalid_path, "w") as f:
        f.write("just text")
    try:
        load_pdf(invalid_path)
    except ValueError:
        assert True
    else:
        assert False
    finally:
        os.remove(invalid_path)
