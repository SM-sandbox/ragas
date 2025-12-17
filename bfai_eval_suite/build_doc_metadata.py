#!/usr/bin/env python3
"""
Document Metadata Extraction Pipeline

Extracts structured metadata from PDF documents using Gemini 2.5 Flash with HIGH reasoning.
Produces JSON files conforming to SolarElectricalDocMetadata schema.

Configuration:
- Model: gemini-2.5-flash
- Thinking: HIGH (24,576 tokens)
- Max Output: 65,536 tokens
- Temperature: 0.0 (deterministic)
"""

import os
import sys
import json
import re
import time
import hashlib
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF
import jsonschema
from google import genai
from google.genai import types

# OCR support
try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Paths
    "input_dir": "corpus_pdfs",
    "output_dir": "doc_metadata",
    "inputs_first10_dir": "doc_metadata/inputs_first10",
    "json_dir": "doc_metadata/json",
    "logs_dir": "doc_metadata/logs",
    "manifests_dir": "doc_metadata/manifests",
    
    # GCP / Vertex AI
    "project": "bf-rag-sandbox-scott",
    "location": "us-central1",
    
    # Model Configuration - Gemini 2.5 Flash with HIGH thinking
    "model": "gemini-2.5-flash",
    "temperature": 0.0,
    "max_output_tokens": 65536,
    "thinking_budget": 24576,  # HIGH thinking
    
    # Processing
    "max_workers": 4,
    "max_retries": 5,
    "base_delay": 2.0,
    "min_text_chars": 1500,
    "pages_to_extract": 10,
    "toc_search_pages": 25,
    
    # Failure threshold
    "max_failure_rate": 0.40,  # Allow up to 40% failures before abort
}

# =============================================================================
# JSON SCHEMA
# =============================================================================

METADATA_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "SolarElectricalDocMetadata",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "doc_id",
        "doc_type",
        "doc_title",
        "publisher_or_manufacturer",
        "canonical_anchor_phrase",
        "publication",
        "standards",
        "equipment",
        "topics",
        "extraction_notes",
        "source"
    ],
    "properties": {
        "doc_id": {"type": "string", "minLength": 1},
        "doc_type": {
            "type": "string",
            "enum": [
                "manufacturer_manual",
                "installation_guide",
                "commissioning_guide",
                "maintenance_manual",
                "troubleshooting_guide",
                "datasheet",
                "safety_standard",
                "electrical_code",
                "regulatory_guidance",
                "training_material",
                "other"
            ]
        },
        "doc_title": {"type": "string", "minLength": 1},
        "subtitle": {"type": ["string", "null"]},
        "publisher_or_manufacturer": {"type": "string", "minLength": 1},
        "canonical_anchor_phrase": {"type": "string", "minLength": 10},
        "publication": {
            "type": "object",
            "additionalProperties": False,
            "required": ["revision", "edition_year", "publication_date_text"],
            "properties": {
                "revision": {"type": ["string", "null"]},
                "edition_year": {"type": ["integer", "null"], "minimum": 1900, "maximum": 2100},
                "publication_date_text": {"type": ["string", "null"]}
            }
        },
        "standards": {
            "type": "object",
            "additionalProperties": False,
            "required": ["standard_body", "standard_ids", "jurisdiction_or_region"],
            "properties": {
                "standard_body": {"type": "array", "items": {"type": "string"}},
                "standard_ids": {"type": "array", "items": {"type": "string"}},
                "jurisdiction_or_region": {"type": "array", "items": {"type": "string"}}
            }
        },
        "equipment": {
            "type": "object",
            "additionalProperties": False,
            "required": ["equipment_family", "models", "product_names"],
            "properties": {
                "equipment_family": {"type": "array", "items": {"type": "string"}},
                "models": {"type": "array", "items": {"type": "string"}},
                "product_names": {"type": "array", "items": {"type": "string"}}
            }
        },
        "topics": {
            "type": "object",
            "additionalProperties": False,
            "required": ["major_topics", "safety_topics", "procedural_domains", "keywords"],
            "properties": {
                "major_topics": {"type": "array", "items": {"type": "string"}, "minItems": 5, "maxItems": 25},
                "safety_topics": {"type": "array", "items": {"type": "string"}},
                "procedural_domains": {"type": "array", "items": {"type": "string"}},
                "keywords": {"type": "array", "items": {"type": "string"}, "minItems": 10, "maxItems": 40},
                "toc_outline": {"type": ["array", "null"], "items": {"type": "string"}}
            }
        },
        "extraction_notes": {
            "type": "object",
            "additionalProperties": False,
            "required": ["confidence", "missing_fields", "ambiguities"],
            "properties": {
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "missing_fields": {"type": "array", "items": {"type": "string"}},
                "ambiguities": {"type": "array", "items": {"type": "string"}}
            }
        },
        "source": {
            "type": "object",
            "additionalProperties": False,
            "required": ["filename", "uri", "page_range_used"],
            "properties": {
                "filename": {"type": "string"},
                "uri": {"type": ["string", "null"]},
                "page_range_used": {"type": "string"}
            }
        }
    }
}

# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are a precise information extraction engine. Output must be valid JSON that conforms to the provided JSON Schema. Do not include any additional keys. Do not include markdown. If a field is not present, use null or an empty array, and list the missing field name in extraction_notes.missing_fields. Never guess model numbers or edition years, only extract what is stated."""

USER_PROMPT_TEMPLATE = """Task: Extract document metadata from the provided text (pages 1–10). Output a single JSON object that strictly conforms to the JSON Schema.
Rules:
- Output JSON only.
- Do not invent values.
- Prefer exact strings as shown in the text.
- For standard_ids, include both common and formal identifiers when present (example: 'NEC' and 'NFPA 70').
- Build canonical_anchor_phrase using available identity signals:
  - If manufacturer manual: '<Manufacturer> <Model or ProductName> <doc_type friendly name>, <revision if any> (<edition_year if any>)'
  - If standard: '<Standard ID> <edition_year if any> (<jurisdiction if any>)'
- major_topics and keywords should reflect the table of contents and repeated technical concepts.
- CRITICAL ARRAY LIMITS - you MUST respect these:
  - major_topics: minimum 5, MAXIMUM 25 items
  - keywords: minimum 10, MAXIMUM 40 items
  - If you have more items, select the most important ones to stay within limits.

Inputs:
doc_id: {doc_id}
filename: {filename}
uri: {uri}
page_range_used: '{page_range}'

Text (pages 1–10):
{text}

JSON Schema:
{schema}"""

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ProcessingResult:
    """Result of processing a single document."""
    doc_id: str
    filename: str
    status: str  # success, failed, skipped
    confidence: Optional[str] = None
    missing_fields_count: int = 0
    models_found_count: int = 0
    standard_ids_count: int = 0
    output_json_path: Optional[str] = None
    output_text_path: Optional[str] = None
    error_message: Optional[str] = None
    processing_time_seconds: float = 0.0
    text_length: int = 0
    pages_extracted: int = 0

# =============================================================================
# UTILITIES
# =============================================================================

def setup_logging(logs_dir: str) -> logging.Logger:
    """Setup logging to file and console."""
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(logs_dir) / f"extraction_{timestamp}.log"
    
    logger = logging.getLogger("doc_metadata")
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def generate_doc_id(filename: str) -> str:
    """Generate a stable doc_id from filename."""
    # Remove extension
    name = Path(filename).stem
    # Create safe slug
    slug = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    slug = re.sub(r'_+', '_', slug).strip('_').lower()
    # Add short hash for uniqueness
    hash_suffix = hashlib.md5(filename.encode()).hexdigest()[:6]
    return f"{slug[:50]}_{hash_suffix}"


def retry_with_backoff(func, max_retries: int, base_delay: float, logger: logging.Logger):
    """Execute function with retry and exponential backoff."""
    import random
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()
            
            if "429" in str(e) or "resource exhausted" in error_str or "quota" in error_str:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"  Rate limited, retry {attempt+1}/{max_retries} in {delay:.1f}s...")
                time.sleep(delay)
            elif attempt < max_retries - 1:
                delay = base_delay + random.uniform(0, 0.5)
                logger.warning(f"  Error: {e}, retry {attempt+1}/{max_retries} in {delay:.1f}s...")
                time.sleep(delay)
            else:
                raise
    
    raise last_exception


# =============================================================================
# PDF EXTRACTION
# =============================================================================

def extract_text_pymupdf(pdf_path: str, max_pages: int = 10) -> Tuple[str, int]:
    """Extract text from PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    pages_to_read = min(max_pages, len(doc))
    
    text_parts = []
    for page_num in range(pages_to_read):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():
            text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
    
    doc.close()
    return "\n\n".join(text_parts), pages_to_read


def extract_text_ocr(pdf_path: str, max_pages: int = 10, logger: logging.Logger = None) -> Tuple[str, int]:
    """Extract text from PDF using Tesseract OCR (for image-heavy PDFs)."""
    if not OCR_AVAILABLE:
        if logger:
            logger.warning("  OCR not available (pytesseract/pdf2image not installed)")
        return "", 0
    
    try:
        # Convert PDF pages to images
        images = convert_from_path(pdf_path, first_page=1, last_page=max_pages, dpi=200)
        
        text_parts = []
        for i, image in enumerate(images):
            # Run OCR on each page image
            page_text = pytesseract.image_to_string(image)
            if page_text.strip():
                text_parts.append(f"--- Page {i + 1} (OCR) ---\n{page_text}")
        
        if logger:
            logger.info(f"  OCR extracted {len(''.join(text_parts))} chars from {len(images)} pages")
        
        return "\n\n".join(text_parts), len(images)
    except Exception as e:
        if logger:
            logger.warning(f"  OCR extraction failed: {e}")
        return "", 0


def find_toc_page(pdf_path: str, search_pages: int = 25) -> Optional[int]:
    """Find a Table of Contents page within first N pages."""
    doc = fitz.open(pdf_path)
    pages_to_search = min(search_pages, len(doc))
    
    toc_patterns = [
        r'table\s+of\s+contents',
        r'contents',
        r'index',
    ]
    
    for page_num in range(pages_to_search):
        page = doc[page_num]
        text = page.get_text().lower()
        for pattern in toc_patterns:
            if re.search(pattern, text):
                doc.close()
                return page_num
    
    doc.close()
    return None


def extract_text_with_fallback(
    pdf_path: str, 
    min_chars: int, 
    pages_to_extract: int,
    toc_search_pages: int,
    logger: logging.Logger
) -> Tuple[str, str]:
    """
    Extract text with fallback strategies.
    Returns (text, page_range_description)
    """
    # Primary extraction
    text, pages_read = extract_text_pymupdf(pdf_path, pages_to_extract)
    
    if len(text) >= min_chars:
        return text, f"1-{pages_read}"
    
    logger.warning(f"  Text too short ({len(text)} chars), trying fallback...")
    
    # Fallback: Try to find TOC and include it
    toc_page = find_toc_page(pdf_path, toc_search_pages)
    
    if toc_page is not None and toc_page >= pages_to_extract:
        # Extract page 1 + TOC page
        doc = fitz.open(pdf_path)
        
        text_parts = []
        # Page 1
        text_parts.append(f"--- Page 1 ---\n{doc[0].get_text()}")
        # TOC page
        text_parts.append(f"--- Page {toc_page + 1} (Table of Contents) ---\n{doc[toc_page].get_text()}")
        
        doc.close()
        
        combined_text = "\n\n".join(text_parts)
        if len(combined_text) > len(text):
            return combined_text, f"1, {toc_page + 1} (TOC)"
    
    # If still short, try pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            text_parts = []
            for i, page in enumerate(pdf.pages[:pages_to_extract]):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(f"--- Page {i + 1} ---\n{page_text}")
            
            pdfplumber_text = "\n\n".join(text_parts)
            if len(pdfplumber_text) > len(text):
                logger.info(f"  pdfplumber extracted more text ({len(pdfplumber_text)} chars)")
                text = pdfplumber_text
                pages_read = min(pages_to_extract, len(pdf.pages))
    except Exception as e:
        logger.warning(f"  pdfplumber fallback failed: {e}")
    
    # Final fallback: OCR for image-heavy PDFs
    if len(text) < min_chars and OCR_AVAILABLE:
        logger.info(f"  Trying OCR extraction for image-heavy PDF...")
        ocr_text, ocr_pages = extract_text_ocr(pdf_path, pages_to_extract, logger)
        if len(ocr_text) > len(text):
            logger.info(f"  OCR extracted more text ({len(ocr_text)} chars)")
            return ocr_text, f"1-{ocr_pages} (OCR)"
    
    return text, f"1-{pages_read}"


# =============================================================================
# LLM EXTRACTION
# =============================================================================

class MetadataExtractor:
    """Extracts metadata using Gemini 2.5 Flash with HIGH thinking."""
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Initialize Vertex AI client
        self.client = genai.Client(
            vertexai=True,
            project=config["project"],
            location=config["location"]
        )
        
        self.model = config["model"]
        self.logger.info(f"Initialized MetadataExtractor with {self.model}")
    
    def extract(
        self, 
        doc_id: str, 
        filename: str, 
        text: str, 
        page_range: str,
        uri: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract metadata from document text."""
        
        # Build prompt
        user_prompt = USER_PROMPT_TEMPLATE.format(
            doc_id=doc_id,
            filename=filename,
            uri=uri or "null",
            page_range=page_range,
            text=text[:50000],  # Truncate if very long
            schema=json.dumps(METADATA_SCHEMA, indent=2)
        )
        
        # Configure generation with HIGH thinking
        gen_config = types.GenerateContentConfig(
            temperature=self.config["temperature"],
            max_output_tokens=self.config["max_output_tokens"],
            thinking_config=types.ThinkingConfig(
                thinking_budget=self.config["thinking_budget"]
            ),
            response_mime_type="application/json",
        )
        
        def do_generate():
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    types.Content(role="user", parts=[
                        types.Part(text=SYSTEM_PROMPT + "\n\n" + user_prompt)
                    ])
                ],
                config=gen_config
            )
            return response
        
        # Call with retry
        response = retry_with_backoff(
            do_generate,
            self.config["max_retries"],
            self.config["base_delay"],
            self.logger
        )
        
        # Parse response - handle potential None
        response_text = response.text
        if response_text is None:
            raise ValueError("Model returned empty response")
        response_text = response_text.strip()
        
        # Clean up if wrapped in markdown
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].strip() == "```":
                lines = lines[:-1]
            response_text = "\n".join(lines)
        
        # Parse and post-process to fix common issues
        metadata = json.loads(response_text)
        
        # Fix: Truncate arrays that exceed limits
        if "topics" in metadata:
            if "major_topics" in metadata["topics"] and len(metadata["topics"]["major_topics"]) > 25:
                metadata["topics"]["major_topics"] = metadata["topics"]["major_topics"][:25]
            if "keywords" in metadata["topics"] and len(metadata["topics"]["keywords"]) > 40:
                metadata["topics"]["keywords"] = metadata["topics"]["keywords"][:40]
        
        # Fix: Convert None to empty string for required string fields
        required_strings = ["doc_id", "doc_type", "doc_title", "publisher_or_manufacturer", "canonical_anchor_phrase"]
        for field in required_strings:
            if metadata.get(field) is None:
                metadata[field] = "Unknown" if field != "canonical_anchor_phrase" else "Unknown Document"
        
        # Fix source.filename if None
        if metadata.get("source", {}).get("filename") is None:
            metadata["source"]["filename"] = self.config.get("current_filename", "unknown.pdf")
        
        return metadata


# =============================================================================
# VALIDATION
# =============================================================================

def validate_metadata(metadata: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate metadata against schema."""
    errors = []
    
    try:
        jsonschema.validate(instance=metadata, schema=METADATA_SCHEMA)
        return True, []
    except jsonschema.ValidationError as e:
        errors.append(f"Schema validation error: {e.message}")
        return False, errors
    except Exception as e:
        errors.append(f"Validation error: {str(e)}")
        return False, errors


# =============================================================================
# MAIN PROCESSING
# =============================================================================

class DocumentMetadataBuilder:
    """Main class for building document metadata."""
    
    def __init__(self, config: dict, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        
        # Setup directories
        for dir_key in ["output_dir", "inputs_first10_dir", "json_dir", "logs_dir", "manifests_dir"]:
            Path(config[dir_key]).mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging(config["logs_dir"])
        self.logger.info("=" * 70)
        self.logger.info("DOCUMENT METADATA EXTRACTION PIPELINE")
        self.logger.info("=" * 70)
        self.logger.info(f"Model: {config['model']}")
        self.logger.info(f"Thinking Budget: {config['thinking_budget']} tokens (HIGH)")
        self.logger.info(f"Max Output Tokens: {config['max_output_tokens']}")
        self.logger.info(f"Dry Run: {dry_run}")
        
        # Initialize extractor
        self.extractor = MetadataExtractor(config, self.logger)
        
        # Results tracking
        self.results: List[ProcessingResult] = []
        
        # JSONL log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.jsonl_log_path = Path(config["logs_dir"]) / f"processing_{timestamp}.jsonl"
    
    def get_pdf_files(self) -> List[Path]:
        """Get list of PDF files to process."""
        input_dir = Path(self.config["input_dir"])
        pdfs = sorted(input_dir.glob("*.pdf"))
        return pdfs
    
    def should_skip(self, doc_id: str) -> bool:
        """Check if document already has valid metadata."""
        json_path = Path(self.config["json_dir"]) / f"{doc_id}.json"
        
        if not json_path.exists():
            return False
        
        try:
            with open(json_path) as f:
                existing = json.load(f)
            valid, _ = validate_metadata(existing)
            if valid:
                self.logger.info(f"  Skipping {doc_id} - valid metadata exists")
                return True
        except:
            pass
        
        return False
    
    def process_single_document(self, pdf_path: Path) -> ProcessingResult:
        """Process a single PDF document."""
        start_time = time.time()
        filename = pdf_path.name
        doc_id = generate_doc_id(filename)
        
        self.logger.info(f"\nProcessing: {filename}")
        self.logger.info(f"  doc_id: {doc_id}")
        
        result = ProcessingResult(
            doc_id=doc_id,
            filename=filename,
            status="pending"
        )
        
        try:
            # Check if should skip
            if self.should_skip(doc_id):
                result.status = "skipped"
                result.processing_time_seconds = time.time() - start_time
                return result
            
            # Extract text
            self.logger.info(f"  Extracting text...")
            text, page_range = extract_text_with_fallback(
                str(pdf_path),
                self.config["min_text_chars"],
                self.config["pages_to_extract"],
                self.config["toc_search_pages"],
                self.logger
            )
            
            result.text_length = len(text)
            result.pages_extracted = int(page_range.split("-")[-1].split(",")[0].strip()) if "-" in page_range else 1
            
            self.logger.info(f"  Extracted {len(text)} chars from pages {page_range}")
            
            # Save extracted text
            text_path = Path(self.config["inputs_first10_dir"]) / f"{doc_id}.txt"
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text)
            result.output_text_path = str(text_path)
            
            # Extract metadata via LLM
            self.logger.info(f"  Calling Gemini 2.5 Flash (HIGH thinking)...")
            uri = f"gs://brightfoxai-documents/BRIGHTFOXAI/EVALV3/{filename}"
            
            metadata = self.extractor.extract(
                doc_id=doc_id,
                filename=filename,
                text=text,
                page_range=page_range,
                uri=uri
            )
            
            # Validate
            self.logger.info(f"  Validating schema...")
            valid, errors = validate_metadata(metadata)
            
            if not valid:
                self.logger.error(f"  Validation failed: {errors}")
                result.status = "failed"
                result.error_message = "; ".join(errors)
                result.processing_time_seconds = time.time() - start_time
                return result
            
            # Save JSON
            json_path = Path(self.config["json_dir"]) / f"{doc_id}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            
            result.output_json_path = str(json_path)
            result.status = "success"
            result.confidence = metadata.get("extraction_notes", {}).get("confidence", "unknown")
            result.missing_fields_count = len(metadata.get("extraction_notes", {}).get("missing_fields", []))
            result.models_found_count = len(metadata.get("equipment", {}).get("models", []))
            result.standard_ids_count = len(metadata.get("standards", {}).get("standard_ids", []))
            
            self.logger.info(f"  ✓ Success - confidence: {result.confidence}, models: {result.models_found_count}, standards: {result.standard_ids_count}")
            
        except Exception as e:
            self.logger.error(f"  ✗ Failed: {e}")
            result.status = "failed"
            result.error_message = str(e)
        
        result.processing_time_seconds = time.time() - start_time
        return result
    
    def log_result(self, result: ProcessingResult):
        """Log result to JSONL file."""
        with open(self.jsonl_log_path, "a") as f:
            f.write(json.dumps(asdict(result)) + "\n")
    
    def run(self, max_docs: Optional[int] = None):
        """Run the full extraction pipeline."""
        pdfs = self.get_pdf_files()
        
        if max_docs:
            pdfs = pdfs[:max_docs]
        
        self.logger.info(f"\nFound {len(pdfs)} PDFs to process")
        
        if self.dry_run:
            self.logger.info("DRY RUN MODE - processing only 2 documents")
            pdfs = pdfs[:2]
        
        total = len(pdfs)
        failed_count = 0
        
        for i, pdf_path in enumerate(pdfs):
            self.logger.info(f"\n[{i+1}/{total}] ", )
            
            result = self.process_single_document(pdf_path)
            self.results.append(result)
            self.log_result(result)
            
            if result.status == "failed":
                failed_count += 1
            
            # Check failure rate
            if total > 4 and failed_count / (i + 1) > self.config["max_failure_rate"]:
                self.logger.error(f"\n!!! ABORTING: Failure rate {failed_count}/{i+1} exceeds {self.config['max_failure_rate']*100}%")
                break
            
            # Rate limiting between docs
            if i < total - 1:
                time.sleep(0.5)
        
        # Generate manifest and summary
        self.generate_manifest()
        self.print_summary()
    
    def generate_manifest(self):
        """Generate manifest file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSONL manifest
        manifest_jsonl = Path(self.config["manifests_dir"]) / f"manifest_{timestamp}.jsonl"
        with open(manifest_jsonl, "w") as f:
            for result in self.results:
                f.write(json.dumps(asdict(result)) + "\n")
        
        # CSV manifest
        manifest_csv = Path(self.config["manifests_dir"]) / f"manifest_{timestamp}.csv"
        with open(manifest_csv, "w") as f:
            headers = ["doc_id", "filename", "status", "confidence", "missing_fields_count", 
                      "models_found_count", "standard_ids_count", "output_json_path", "error_message"]
            f.write(",".join(headers) + "\n")
            for result in self.results:
                row = [
                    result.doc_id,
                    f'"{result.filename}"',
                    result.status,
                    result.confidence or "",
                    str(result.missing_fields_count),
                    str(result.models_found_count),
                    str(result.standard_ids_count),
                    result.output_json_path or "",
                    f'"{result.error_message}"' if result.error_message else ""
                ]
                f.write(",".join(row) + "\n")
        
        self.logger.info(f"\nManifest saved to: {manifest_jsonl}")
        self.logger.info(f"Manifest CSV saved to: {manifest_csv}")
    
    def print_summary(self):
        """Print final summary."""
        total = len(self.results)
        succeeded = sum(1 for r in self.results if r.status == "success")
        failed = sum(1 for r in self.results if r.status == "failed")
        skipped = sum(1 for r in self.results if r.status == "skipped")
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("EXTRACTION COMPLETE")
        self.logger.info("=" * 70)
        self.logger.info(f"Total PDFs:     {total}")
        self.logger.info(f"Succeeded:      {succeeded}")
        self.logger.info(f"Failed:         {failed}")
        self.logger.info(f"Skipped:        {skipped}")
        self.logger.info(f"Success Rate:   {succeeded/total*100:.1f}%" if total > 0 else "N/A")
        
        if failed > 0:
            self.logger.info("\nFailure Reasons:")
            failure_reasons = {}
            for r in self.results:
                if r.status == "failed" and r.error_message:
                    reason = r.error_message[:100]
                    failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            
            for reason, count in sorted(failure_reasons.items(), key=lambda x: -x[1]):
                self.logger.info(f"  [{count}] {reason}")
        
        # Confidence distribution
        confidence_counts = {"high": 0, "medium": 0, "low": 0}
        for r in self.results:
            if r.confidence in confidence_counts:
                confidence_counts[r.confidence] += 1
        
        self.logger.info(f"\nConfidence Distribution:")
        for conf, count in confidence_counts.items():
            self.logger.info(f"  {conf}: {count}")
        
        self.logger.info("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Extract document metadata from PDFs")
    parser.add_argument("--dry-run", action="store_true", help="Process only 2 documents for testing")
    parser.add_argument("--max-docs", type=int, default=None, help="Maximum documents to process")
    parser.add_argument("--input-dir", type=str, default=None, help="Override input directory")
    args = parser.parse_args()
    
    config = CONFIG.copy()
    if args.input_dir:
        config["input_dir"] = args.input_dir
    
    builder = DocumentMetadataBuilder(config, dry_run=args.dry_run)
    builder.run(max_docs=args.max_docs)


if __name__ == "__main__":
    main()
