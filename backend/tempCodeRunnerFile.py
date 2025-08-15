# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.config import Settings
import numpy as np
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx
import json
import os
import re
import uuid
from datetime import datetime
import asyncio
import aiofiles
from collections import defaultdict
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Legal Document RAG System", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    document_type: str
    chunks_created: int
    status: str

class QueryRequest(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None
    max_results: int = 10
    similarity_threshold: float = 0.7

class Citation(BaseModel):
    document_id: str
    document_name: str
    document_type: str
    section: str
    page_number: Optional[int] = None
    relevance_score: float
    text_snippet: str

class ConflictAnalysis(BaseModel):
    conflicting_citations: List[Citation]
    analysis: str
    confidence: float

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    conflicts: Optional[ConflictAnalysis] = None
    processing_time: float
    total_documents_searched: int

class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    document_type: str
    upload_date: str
    chunk_count: int
    status: str

# Global variables
embedding_model = None
chroma_client = None
collection = None

class LegalDocumentProcessor:
    def __init__(self):
        self.legal_sections = [
            "article", "section", "clause", "paragraph", "subsection",
            "whereas", "now therefore", "definitions", "terms and conditions",
            "obligations", "representations", "warranties", "indemnification",
            "termination", "governing law", "jurisdiction", "force majeure"
        ]
        
        self.document_types = {
            "contract": ["agreement", "contract", "terms", "conditions"],
            "case_law": ["court", "judge", "plaintiff", "defendant", "ruling"],
            "statute": ["statute", "law", "act", "code", "regulation"]
        }

    def detect_document_type(self, text: str) -> str:
        """Detect the type of legal document based on content"""
        text_lower = text.lower()
        
        scores = {}
        for doc_type, keywords in self.document_types.items():
            score = sum(text_lower.count(keyword) for keyword in keywords)
            scores[doc_type] = score
        
        return max(scores, key=scores.get) if max(scores.values()) > 0 else "general_legal"

    def extract_legal_sections(self, text: str) -> List[Dict[str, Any]]:
        """Extract legal sections from document text"""
        sections = []
        
        # Pattern to match numbered sections
        section_patterns = [
            r'(?i)(section|article|clause)\s+(\d+(?:\.\d+)*)',
            r'(?i)(\d+(?:\.\d+)*)\.\s*([A-Z][^.]*)',
            r'(?i)(whereas)',
            r'(?i)(now\s+therefore)'
        ]
        
        lines = text.split('\n')
        current_section = None
        current_content = []
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
                
            # Check if line starts a new section
            is_section_start = False
            for pattern in section_patterns:
                if re.match(pattern, line_stripped):
                    is_section_start = True
                    if current_section:
                        sections.append({
                            'section': current_section,
                            'content': '\n'.join(current_content),
                            'start_position': len('\n'.join(lines[:i-len(current_content)])),
                            'end_position': len('\n'.join(lines[:i]))
                        })
                    current_section = line_stripped[:100]  # First 100 chars as section title
                    current_content = [line_stripped]
                    break
            
            if not is_section_start and current_section:
                current_content.append(line_stripped)
        
        # Add the last section
        if current_section:
            sections.append({
                'section': current_section,
                'content': '\n'.join(current_content),
                'start_position': len(text) - len('\n'.join(current_content)),
                'end_position': len(text)
            })
        
        return sections

    def intelligent_chunking(self, text: str, doc_type: str) -> List[Dict[str, Any]]:
        """Create intelligent chunks based on legal document structure"""
        chunks = []
        
        # First, try to extract legal sections
        sections = self.extract_legal_sections(text)
        
        if sections:
            # Use section-based chunking
            for section in sections:
                # If section is too long, split it further
                if len(section['content']) > 1000:
                    sub_chunks = self._split_long_section(section['content'])
                    for i, sub_chunk in enumerate(sub_chunks):
                        chunks.append({
                            'text': sub_chunk,
                            'section': f"{section['section']} (Part {i+1})",
                            'chunk_type': 'section_part',
                            'metadata': {
                                'parent_section': section['section'],
                                'part_number': i+1,
                                'document_type': doc_type
                            }
                        })
                else:
                    chunks.append({
                        'text': section['content'],
                        'section': section['section'],
                        'chunk_type': 'section',
                        'metadata': {
                            'document_type': doc_type
                        }
                    })
        else:
            # Fall back to sentence-based chunking
            sentences = sent_tokenize(text)
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                if current_length + len(sentence) > 800 and current_chunk:
                    chunks.append({
                        'text': ' '.join(current_chunk),
                        'section': f"Paragraph {len(chunks) + 1}",
                        'chunk_type': 'paragraph',
                        'metadata': {
                            'document_type': doc_type
                        }
                    })
                    current_chunk = [sentence]
                    current_length = len(sentence)
                else:
                    current_chunk.append(sentence)
                    current_length += len(sentence)
            
            if current_chunk:
                chunks.append({
                    'text': ' '.join(current_chunk),
                    'section': f"Paragraph {len(chunks) + 1}",
                    'chunk_type': 'paragraph',
                    'metadata': {
                        'document_type': doc_type
                    }
                })
        
        return chunks

    def _split_long_section(self, text: str) -> List[str]:
        """Split long sections into smaller chunks"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) > 800 and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)
            else:
                current_chunk.append(sentence)
                current_length += len(sentence)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

class LegalQueryProcessor:
    def __init__(self):
        self.legal_keywords = {
            'obligation': ['shall', 'must', 'required', 'obligation', 'duty'],
            'prohibition': ['shall not', 'prohibited', 'forbidden', 'may not'],
            'permission': ['may', 'permitted', 'allowed', 'can'],
            'definition': ['means', 'defined as', 'refers to', 'includes'],
            'condition': ['if', 'unless', 'provided that', 'subject to'],
            'temporal': ['upon', 'after', 'before', 'during', 'within']
        }

    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze the legal intent of the query"""
        query_lower = query.lower()
        
        intent_scores = {}
        for intent, keywords in self.legal_keywords.items():
            score = sum(query_lower.count(keyword) for keyword in keywords)
            intent_scores[intent] = score
        
        primary_intent = max(intent_scores, key=intent_scores.get)
        
        return {
            'primary_intent': primary_intent,
            'intent_scores': intent_scores,
            'query_type': self._classify_query_type(query_lower)
        }

    def _classify_query_type(self, query: str) -> str:
        """Classify the type of legal query"""
        if any(word in query for word in ['what is', 'define', 'definition']):
            return 'definition'
        elif any(word in query for word in ['can i', 'am i allowed', 'is it permitted']):
            return 'permission'
        elif any(word in query for word in ['must i', 'required to', 'obligation']):
            return 'obligation'
        elif any(word in query for word in ['conflict', 'contradiction', 'inconsistent']):
            return 'conflict_analysis'
        else:
            return 'general_inquiry'

    def detect_conflicts(self, citations: List[Citation]) -> Optional[ConflictAnalysis]:
        """Detect conflicts between different citations"""
        if len(citations) < 2:
            return None
        
        # Group citations by document type
        doc_groups = defaultdict(list)
        for citation in citations:
            doc_groups[citation.document_type].append(citation)
        
        # Simple conflict detection based on contradictory keywords
        conflict_keywords = [
            ('shall', 'shall not'),
            ('required', 'prohibited'),
            ('must', 'may not'),
            ('permitted', 'forbidden')
        ]
        
        conflicting_citations = []
        for i, citation1 in enumerate(citations):
            for j, citation2 in enumerate(citations[i+1:], i+1):
                if self._check_textual_conflict(citation1.text_snippet, citation2.text_snippet, conflict_keywords):
                    conflicting_citations.extend([citation1, citation2])
        
        if conflicting_citations:
            return ConflictAnalysis(
                conflicting_citations=list(set(conflicting_citations)),
                analysis="Potential conflicts detected based on contradictory language patterns.",
                confidence=0.7
            )
        
        return None

    def _check_textual_conflict(self, text1: str, text2: str, conflict_keywords: List[tuple]) -> bool:
        """Check if two text snippets contain conflicting information"""
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        for pos_keyword, neg_keyword in conflict_keywords:
            if (pos_keyword in text1_lower and neg_keyword in text2_lower) or \
               (neg_keyword in text1_lower and pos_keyword in text2_lower):
                return True
        
        return False

# Initialize processors
doc_processor = LegalDocumentProcessor()
query_processor = LegalQueryProcessor()

async def initialize_system():
    """Initialize the embedding model and vector database"""
    global embedding_model, chroma_client, collection
    
    try:
        # Initialize embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Embedding model loaded successfully")
        
        # Initialize ChromaDB
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Create or get collection
        try:
            collection = chroma_client.get_collection("legal_documents")
        except:
            collection = chroma_client.create_collection(
                name="legal_documents",
                metadata={"description": "Legal documents collection for RAG system"}
            )
        
        logger.info("ChromaDB initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing system: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    await initialize_system()

def extract_text_from_file(file_path: str, filename: str) -> str:
    """Extract text from uploaded file"""
    try:
        if filename.endswith('.pdf'):
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        elif filename.endswith('.docx'):
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        elif filename.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        else:
            raise ValueError(f"Unsupported file format: {filename}")
    except Exception as e:
        logger.error(f"Error extracting text from {filename}: {str(e)}")
        raise

@app.post("/upload-document", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a legal document"""
    try:
        # Generate unique document ID
        doc_id = str(uuid.uuid4())
        
        # Save uploaded file temporarily
        temp_path = f"temp_{doc_id}_{file.filename}"
        async with aiofiles.open(temp_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Extract text from file
        text = extract_text_from_file(temp_path, file.filename)
        
        # Detect document type
        doc_type = doc_processor.detect_document_type(text)
        
        # Create intelligent chunks
        chunks = doc_processor.intelligent_chunking(text, doc_type)
        
        # Generate embeddings and store in vector database
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = embedding_model.encode(chunk_texts).tolist()
        
        # Prepare metadata
        metadatas = []
        ids = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            ids.append(chunk_id)
            metadatas.append({
                'document_id': doc_id,
                'filename': file.filename,
                'document_type': doc_type,
                'section': chunk['section'],
                'chunk_type': chunk['chunk_type'],
                'chunk_index': i,
                'upload_date': datetime.now().isoformat(),
                **chunk.get('metadata', {})
            })
        
        # Add to ChromaDB
        collection.add(
            embeddings=embeddings,
            documents=chunk_texts,
            metadatas=metadatas,
            ids=ids
        )
        
        # Clean up temporary file
        os.remove(temp_path)
        
        logger.info(f"Document {file.filename} processed successfully with {len(chunks)} chunks")
        
        return DocumentUploadResponse(
            document_id=doc_id,
            filename=file.filename,
            document_type=doc_type,
            chunks_created=len(chunks),
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        # Clean up temporary file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the legal documents"""
    start_time = datetime.now()
    
    try:
        # Analyze query intent
        query_analysis = query_processor.analyze_query_intent(request.query)
        
        # Generate query embedding
        query_embedding = embedding_model.encode([request.query]).tolist()[0]
        
        # Build filter conditions
        where_conditions = {}
        if request.document_ids:
            where_conditions['document_id'] = {'$in': request.document_ids}
        
        # Perform vector search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=request.max_results,
            where=where_conditions if where_conditions else None
        )
        
        # Process results and create citations
        citations = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            # Convert distance to similarity score
            similarity = 1 - distance
            
            if similarity >= request.similarity_threshold:
                citations.append(Citation(
                    document_id=metadata['document_id'],
                    document_name=metadata['filename'],
                    document_type=metadata['document_type'],
                    section=metadata['section'],
                    page_number=metadata.get('page_number'),
                    relevance_score=similarity,
                    text_snippet=doc[:500] + "..." if len(doc) > 500 else doc
                ))
        
        # Generate contextual answer
        if citations:
            context = "\n\n".join([f"From {cite.document_name} ({cite.section}):\n{cite.text_snippet}" 
                                 for cite in citations[:5]])
            
            answer = f"Based on the legal documents, here's what I found regarding your query:\n\n"
            
            # Provide a structured answer based on query intent
            if query_analysis['query_type'] == 'definition':
                answer += "**Definition/Explanation:**\n"
            elif query_analysis['query_type'] == 'obligation':
                answer += "**Legal Obligations:**\n"
            elif query_analysis['query_type'] == 'permission':
                answer += "**Permissions/Rights:**\n"
            
            # Add relevant excerpts with proper attribution
            for cite in citations[:3]:  # Top 3 most relevant
                answer += f"\n• **{cite.document_name} - {cite.section}:** {cite.text_snippet[:200]}...\n"
            
            if len(citations) > 3:
                answer += f"\n*And {len(citations) - 3} additional relevant sections found.*"
                
        else:
            answer = "I couldn't find any relevant information in the uploaded legal documents for your query. Please try rephrasing your question or uploading more relevant documents."
        
        # Detect conflicts
        conflicts = query_processor.detect_conflicts(citations)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Count unique documents searched
        unique_docs = len(set(cite.document_id for cite in citations))
        
        return QueryResponse(
            answer=answer,
            citations=citations,
            conflicts=conflicts,
            processing_time=processing_time,
            total_documents_searched=unique_docs
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", response_model=List[DocumentInfo])
async def get_documents():
    """Get list of all uploaded documents"""
    try:
        # Get all unique documents from the collection
        results = collection.get()
        
        # Group by document_id to get unique documents
        documents = {}
        for metadata in results['metadatas']:
            doc_id = metadata['document_id']
            if doc_id not in documents:
                documents[doc_id] = {
                    'document_id': doc_id,
                    'filename': metadata['filename'],
                    'document_type': metadata['document_type'],
                    'upload_date': metadata['upload_date'],
                    'chunk_count': 1,
                    'status': 'active'
                }
            else:
                documents[doc_id]['chunk_count'] += 1
        
        return list(documents.values())
        
    except Exception as e:
        logger.error(f"Error getting documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and all its chunks"""
    try:
        # Get all chunk IDs for this document
        results = collection.get(
            where={'document_id': document_id}
        )
        
        if not results['ids']:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete all chunks for this document
        collection.delete(ids=results['ids'])
        
        logger.info(f"Document {document_id} deleted successfully")
        return {"message": "Document deleted successfully", "document_id": document_id}
        
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)