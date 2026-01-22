import json
import re
from typing import List, Dict, Any
from pathlib import Path
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import CHUNK_SIZE, CHUNK_OVERLAP

class DataProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def extract_pdf_text(self, pdf_path: Path) -> List[Dict[str, str]]:
        """Extract text from PDF file with page tracking"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                pages = []
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        pages.append({
                            "page_number": page_num + 1,
                            "text": page_text.strip()
                        })
                return pages
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {str(e)}")
            return []
    
    def process_json_file(self, json_path: Path) -> List[Dict[str, Any]]:
        """Process JSON files and extract documents"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            file_name = json_path.name
            
            if isinstance(data, list):
                for i, item in enumerate(data):
                    doc = self._extract_document_from_item(item, file_name, i)
                    if doc:
                        documents.append(doc)
            elif isinstance(data, dict):
                doc = self._extract_document_from_item(data, file_name, 0)
                if doc:
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"Error processing JSON {json_path}: {str(e)}")
            return []
    
    def _extract_document_from_item(self, item: Dict, source_file: str, index: int) -> Dict[str, Any]:
        """Extract document fields from JSON item"""
        content = ""
        title = ""
        metadata = {
            "source_file": source_file,
            "index": index
        }
        
        # Handle different JSON structures
        if 'content' in item:
            content = item['content']
        elif 'text' in item:
            content = item['text']
        elif 'body' in item:
            content = item['body']
        
        if 'title' in item:
            title = item['title']
        elif 'headline' in item:
            title = item['headline']
        
        # Add metadata fields
        for key in ['url', 'author', 'article_date', 'scraped_at', 'score', 'subreddit']:
            if key in item:
                metadata[key] = item[key]
        
        # Handle sections (like from booking blog data)
        if 'sections' in item and isinstance(item['sections'], list):
            section_contents = []
            for section in item['sections']:
                if isinstance(section, dict):
                    heading = section.get('heading', '')
                    section_content = section.get('content', [])
                    if isinstance(section_content, list):
                        section_text = ' '.join(section_content)
                    else:
                        section_text = str(section_content)
                    
                    if heading:
                        section_contents.append(f"{heading}\n{section_text}")
                    else:
                        section_contents.append(section_text)
            
            if section_contents:
                content = '\n\n'.join(section_contents)
        
        if content and len(content.strip()) > 50:
            return {
                "content": content.strip(),
                "title": title,
                "metadata": metadata
            }
        
        return None
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split document into chunks with page/index tracking"""
        content = document['content']
        title = document['title']
        metadata = document['metadata']
        
        chunks = self.text_splitter.split_text(content)
        
        chunked_docs = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_id": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk),
                "original_document_id": f"{metadata.get('source_file', 'unknown')}_{metadata.get('index', 0)}",
                "full_content": content,  # Store full content for context retrieval
                "start_char": content.find(chunk),
                "end_char": content.find(chunk) + len(chunk)
            })
            
            chunked_docs.append({
                "content": chunk,
                "title": title,
                "metadata": chunk_metadata
            })
        
        return chunked_docs
    
    def process_pdf_file(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Process PDF file and return chunked documents with page tracking"""
        pages = self.extract_pdf_text(pdf_path)
        if not pages:
            return []
        
        all_chunked_docs = []
        
        for page_data in pages:
            document = {
                "content": page_data["text"],
                "title": f"{pdf_path.stem} - Page {page_data['page_number']}",
                "metadata": {
                    "source_file": pdf_path.name,
                    "file_type": "pdf",
                    "page_number": page_data["page_number"],
                    "total_pages": len(pages)
                }
            }
            
            chunks = self.chunk_document(document)
            all_chunked_docs.extend(chunks)
        
        return all_chunked_docs