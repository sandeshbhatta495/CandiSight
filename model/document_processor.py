import os
import PyPDF2
from docx import Document
from typing import Union, Optional

class DocumentProcessor:
    """Utility class for processing different document types (PDF, DOCX, TXT)."""
    
    @staticmethod
    def read_file(file_path: str) -> Optional[str]:
        """Read text from a file (PDF, DOCX, or TXT)."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        try:
            if ext == '.pdf':
                return DocumentProcessor._read_pdf(file_path)
            elif ext == '.docx':
                return DocumentProcessor._read_docx(file_path)
            elif ext == '.txt':
                return DocumentProcessor._read_txt(file_path)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            return None
    
    @staticmethod
    def _read_pdf(file_path: str) -> str:
        """Extract text from a PDF file."""
        text = []
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text.append(page.extract_text() or '')
        return '\n'.join(text)
    
    @staticmethod
    def _read_docx(file_path: str) -> str:
        """Extract text from a DOCX file."""
        doc = Document(file_path)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    
    @staticmethod
    def _read_txt(file_path: str) -> str:
        """Read text from a TXT file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

# Example usage
if __name__ == "__main__":
    # Example usage
    processor = DocumentProcessor()
    
    # Example with a PDF file
    # pdf_text = processor.read_file("path/to/resume.pdf")
    # print(f"PDF content: {pdf_text[:200]}...")
    
    # Example with a DOCX file
    # docx_text = processor.read_file("path/to/job_description.docx")
    # print(f"DOCX content: {docx_text[:200]}...")
    
    # Example with a TXT file
    # txt_text = processor.read_file("path/to/notes.txt")
    # print(f"TXT content: {txt_text[:200]}...")
