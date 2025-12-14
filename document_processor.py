"""Enhanced document processing with section mapping"""
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import PyPDF2
import pdfplumber
from dataclasses import dataclass

@dataclass
class DocumentChunk:
    """Represents a chunk of document text with metadata"""
    content: str
    page_number: int
    chunk_id: int
    source: str
    metadata: Dict

class DocumentProcessor:
    """Handles PDF processing with section detection"""
    
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 400):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, pdf_path: str) -> tuple[str, Dict]:
        """Extract text from PDF with page information"""
        text = ""
        page_texts = []
        metadata = {
            "pages": 0,
            "title": Path(pdf_path).stem,
            "extraction_method": None
        }
        
        # Try pdfplumber first
        try:
            with pdfplumber.open(pdf_path) as pdf:
                metadata["pages"] = len(pdf.pages)
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        page_texts.append((i+1, page_text))
                        text += f"\n[PAGE {i+1}]\n{page_text}\n"
                metadata["extraction_method"] = "pdfplumber"
        except Exception as e:
            print(f"pdfplumber failed: {e}, trying PyPDF2...")
            
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    metadata["pages"] = len(pdf_reader.pages)
                    
                    for i, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text()
                        if page_text:
                            page_texts.append((i+1, page_text))
                            text += f"\n[PAGE {i+1}]\n{page_text}\n"
                    metadata["extraction_method"] = "PyPDF2"
            except Exception as e:
                raise ValueError(f"Failed to extract text from PDF: {e}")
        
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF")
        
        # Clean the extracted text
        text = self._clean_text(text)
        
        return text, metadata
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text while preserving structure"""
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n\n+', '\n\n', text)
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace("'", "'").replace("'", "'")
        return text.strip()
    
    def build_section_map(self, text: str) -> Tuple[Dict[int, str], Dict[int, str]]:
        """Build a map of page numbers to sections"""
        section_map = {}
        page_section_splits = {}  # Store as STRING, not list
        current_section = 'introduction'
        
        page_pattern = r'\[PAGE (\d+)\](.*?)(?=\[PAGE \d+\]|$)'
        pages = re.findall(page_pattern, text, re.DOTALL)
        
        section_patterns = [
            (r'\b(\d+\.?\s+)(abstract|summary)\b', 'abstract'),
            (r'\b(\d+\.?\s+)(introduction|background)\b', 'introduction'),
            (r'\b(\d+\.?\s+)(literature\s+review|related\s+work|prior\s+work)\b', 'literature_review'),
            (r'\b(\d+\.?\s+)(method|methodology|methods|approach|implementation)\b', 'method'),
            (r'\b(\d+\.?\s+)(result|results|findings|evaluation|performance|analysis)\b', 'results'),
            (r'\b(\d+\.?\s+)(discussion|interpretation)\b', 'discussion'),
            (r'\b(\d+\.?\s+)(conclusion|conclusions|concluding|summary)\b', 'conclusion'),
            (r'\b(\d+\.?\s+)(reference|references|bibliography)\b', 'references'),
            (r'\b([ivxIVX]+\.?\s+)(results?|findings?)\b', 'results'),
            (r'\b([ivxIVX]+\.?\s+)(conclusions?|concluding)\b', 'conclusion'),
            (r'\b([ivxIVX]+\.?\s+)(discussion)\b', 'discussion'),
            (r'^(abstract|summary)\b', 'abstract'),
            (r'^(introduction|background)\b', 'introduction'),
            (r'^(literature\s+review|related\s+work)\b', 'literature_review'),
            (r'^(method|methodology|methods)\b', 'method'),
            (r'^(result|results|findings)\b', 'results'),
            (r'^(discussion)\b', 'discussion'),
            (r'^(conclusion|conclusions)\b', 'conclusion'),
            (r'^(reference|references|bibliography)\b', 'references'),
        ]
        
        print("\n🔍 Scanning document for section headers...")
        
        for page_num_str, page_content in pages:
            page_num = int(page_num_str)
            page_lower = page_content.lower()
            
            sections_on_page = []
            
            for pattern, section_name in section_patterns:
                matches = re.finditer(pattern, page_lower, re.MULTILINE | re.IGNORECASE)
                for match in matches:
                    sections_on_page.append({
                        'section': section_name,
                        'position': match.start(),
                        'text': match.group(0)
                    })
            
            if sections_on_page:
                sections_on_page.sort(key=lambda x: x['position'])
                
                if len(sections_on_page) > 1:
                    last_section = sections_on_page[-1]
                    current_section = last_section['section']
                    section_map[page_num] = current_section
                    
                    # Store split sections as COMMA-SEPARATED STRING
                    section_names = [s['section'] for s in sections_on_page]
                    page_section_splits[page_num] = ','.join(section_names)
                    
                    print(f"  📄 Page {page_num} has {len(sections_on_page)} sections:")
                    for i, sec in enumerate(sections_on_page):
                        marker = "✓" if sec == last_section else " "
                        print(f"     {marker} '{sec['section']}': {sec['text']}")
                    print(f"     → Using: '{last_section['section']}' (continues on next pages)")
                else:
                    current_section = sections_on_page[0]['section']
                    section_map[page_num] = current_section
                    print(f"  ✓ Found '{current_section}' on page {page_num}: {sections_on_page[0]['text']}")
            else:
                if page_num not in section_map:
                    section_map[page_num] = current_section
        
        all_pages = sorted([int(p[0]) for p in pages])
        for i, page_num in enumerate(all_pages):
            if page_num not in section_map:
                if i > 0:
                    section_map[page_num] = section_map.get(all_pages[i-1], 'body')
                else:
                    section_map[page_num] = 'introduction'
        
        print(f"\n📋 Section map created: {len(section_map)} pages")
        print(f"   Sections found: {set(section_map.values())}")
        print(f"   📍 Page-to-Section mapping:")
        for page in sorted(section_map.keys()):
            if page in page_section_splits:
                print(f"      Page {page}: {section_map[page]} (split page: {page_section_splits[page]})")
            else:
                print(f"      Page {page}: {section_map[page]}")
        
        return section_map, page_section_splits
    
    def chunk_text(self, text: str, source: str) -> List[DocumentChunk]:
        """Create chunks with section awareness using section map"""
        
        section_map, page_section_splits = self.build_section_map(text)
        
        page_pattern = r'\[PAGE (\d+)\]'
        parts = re.split(page_pattern, text)
        
        chunks = []
        chunk_id = 0
        current_chunk = ""
        current_page = 1
        sections_found = set()
        
        for i in range(1, len(parts), 2):
            if i < len(parts):
                page_num = int(parts[i])
                page_content = parts[i+1] if i+1 < len(parts) else ""
                
                paragraphs = page_content.split('\n\n')
                
                for para in paragraphs:
                    para = para.strip()
                    if not para or len(para) < 50:
                        continue
                    
                    if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                        section_type = section_map.get(current_page, 'body')
                        sections_found.add(section_type)
                        
                        # Get split sections as STRING (already stored as string)
                        split_sections_str = page_section_splits.get(current_page, '')
                        
                        chunks.append(DocumentChunk(
                            content=current_chunk.strip(),
                            page_number=current_page,
                            chunk_id=chunk_id,
                            source=source,
                            metadata={
                                "length": len(current_chunk),
                                "section": section_type,
                                "has_conclusion": section_type in ['conclusion', 'discussion'],
                                "split_page_sections": split_sections_str
                            }
                        ))
                        chunk_id += 1
                        
                        words = current_chunk.split()
                        overlap_size = min(50, len(words))
                        overlap_text = ' '.join(words[-overlap_size:])
                        current_chunk = overlap_text + "\n\n" + para
                        current_page = page_num
                    else:
                        current_chunk = current_chunk + "\n\n" + para if current_chunk else para
                        current_page = page_num
        
        if current_chunk.strip():
            section_type = section_map.get(current_page, 'body')
            sections_found.add(section_type)
            split_sections_str = page_section_splits.get(current_page, '')
            
            chunks.append(DocumentChunk(
                content=current_chunk.strip(),
                page_number=current_page,
                chunk_id=chunk_id,
                source=source,
                metadata={
                    "length": len(current_chunk),
                    "section": section_type,
                    "has_conclusion": section_type in ['conclusion', 'discussion'],
                    "split_page_sections": split_sections_str
                }
            ))
        
        print(f"\n✅ Created {len(chunks)} chunks")
        section_counts = {}
        for chunk in chunks:
            section = chunk.metadata.get('section', 'unknown')
            section_counts[section] = section_counts.get(section, 0) + 1
        print(f"   Section distribution: {section_counts}")
        print(f"   Unique sections: {sections_found}")
        
        print("\n🔍 DEBUG - First 3 chunks:")
        for i, chunk in enumerate(chunks[:3]):
            split_info = chunk.metadata.get('split_page_sections', '')
            if split_info:
                print(f"  Chunk {i}: Page {chunk.page_number}, Section: {chunk.metadata.get('section')} (split: {split_info})")
            else:
                print(f"  Chunk {i}: Page {chunk.page_number}, Section: {chunk.metadata.get('section')}")
        
        last_split = chunks[-1].metadata.get('split_page_sections', '')
        if last_split:
            print(f"  Last chunk: Page {chunks[-1].page_number}, Section: {chunks[-1].metadata.get('section')} (split: {last_split})")
        else:
            print(f"  Last chunk: Page {chunks[-1].page_number}, Section: {chunks[-1].metadata.get('section')}")
        
        return chunks
    
    def process_document(self, pdf_path: str) -> tuple[List[DocumentChunk], Dict]:
        """Complete document processing pipeline"""
        text, metadata = self.extract_text_from_pdf(pdf_path)
        chunks = self.chunk_text(text, source=Path(pdf_path).name)
        
        metadata["num_chunks"] = len(chunks)
        metadata["total_characters"] = len(text)
        
        return chunks, metadata