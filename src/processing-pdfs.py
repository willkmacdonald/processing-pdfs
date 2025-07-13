#!/usr/bin/env python3
"""
PDF Pre-Processor

A high-performance PDF document processor that extracts structured content
using Docling with configurable quality/performance trade-offs.
"""

import os
import json
import logging
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import traceback
import re

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
except ImportError as e:
    print(f"Error importing Docling: {e}")
    print("Please install Docling or run in the provided container")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

class ProcessingProfile:
    """Processing profiles for different quality/speed trade-offs."""
    
    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"
    
    @classmethod
    def get_profile_settings(cls, profile: str) -> Dict[str, Any]:
        """Get pipeline settings for a processing profile."""
        profiles = {
            cls.FAST: {
                'do_table_structure': False,
                'do_picture_classification': False,
                'do_picture_description': False,
                'do_formula_enrichment': False,
                'do_code_enrichment': False,
                'generate_page_images': False,
                'images_scale': 1.0,
                'description': 'Fast processing - text extraction only'
            },
            cls.BALANCED: {
                'do_table_structure': True,
                'do_picture_classification': False,
                'do_picture_description': False,
                'do_formula_enrichment': False,
                'do_code_enrichment': False,
                'generate_page_images': False,
                'images_scale': 1.0,
                'description': 'Balanced processing - text + basic structure'
            },
            cls.QUALITY: {
                'do_table_structure': True,
                'do_picture_classification': False,
                'do_picture_description': False,
                'do_formula_enrichment': False,  # Disabled for medtech
                'do_code_enrichment': False,     # Disabled for medtech
                'generate_page_images': False,
                'images_scale': 2.0,
                'description': 'High quality - medtech optimized (no formulas/code)'
            }
        }
        return profiles.get(profile, profiles[cls.BALANCED])

class PDFPreProcessor:
    """Main PDF preprocessing class with configurable quality and parallelization."""
    
    def __init__(self, 
                 input_dir: str = "./input", 
                 output_dir: str = "./output",
                 profile: str = ProcessingProfile.BALANCED,
                 debug: bool = False):
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.profile = profile
        self.debug = debug
        
        # Ensure directories exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize converter
        self._setup_converter()
        
        # Statistics
        self.stats = {
            "processed": 0,
            "failed": 0,
            "total": 0,
            "total_time": 0.0,
            "avg_time_per_file": 0.0,
            "profile_used": profile
        }
        
        if self.debug:
            logger.setLevel(logging.DEBUG)
    
    def _setup_converter(self):
        """Initialize Docling converter with profile settings."""
        try:
            logger.info(f"Initializing converter with '{self.profile}' profile...")
            
            profile_settings = ProcessingProfile.get_profile_settings(self.profile)
            logger.info(f"Profile: {profile_settings['description']}")
            
            # Create PDF format options
            pdf_options = PdfFormatOption()
            
            # Configure pipeline if available
            if hasattr(pdf_options, 'pipeline_options') and pdf_options.pipeline_options:
                pipeline = pdf_options.pipeline_options
                
                # Apply profile settings
                for setting, value in profile_settings.items():
                    if setting != 'description' and hasattr(pipeline, setting):
                        setattr(pipeline, setting, value)
                        logger.debug(f"Set {setting} = {value}")
            
            # Create converter
            self.converter = DocumentConverter(
                format_options={InputFormat.PDF: pdf_options}
            )
            
            logger.info("Converter initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to create enhanced converter: {e}")
            logger.info("Falling back to basic converter...")
            self.converter = DocumentConverter()
    
    def process_directory(self) -> Dict[str, Any]:
        """Process all PDF files in the input directory."""
        pdf_files = list(self.input_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.input_dir}")
            return self.stats
        
        self.stats["total"] = len(pdf_files)
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        start_time = time.time()
        
        # Process files sequentially
        for i, pdf_file in enumerate(pdf_files, 1):
            try:
                logger.info(f"Processing {i}/{len(pdf_files)}: {pdf_file.name}")
                self._process_single_pdf(pdf_file)
                self.stats["processed"] += 1
                logger.info(f"✓ Completed: {pdf_file.name}")
            except Exception as e:
                self.stats["failed"] += 1
                logger.error(f"✗ Failed: {pdf_file.name} - {e}")
                if self.debug:
                    traceback.print_exc()
        
        self.stats["total_time"] = time.time() - start_time
        if self.stats["processed"] > 0:
            self.stats["avg_time_per_file"] = self.stats["total_time"] / self.stats["processed"]
        
        self._print_summary()
        return self.stats
    
    def _create_markdown_from_docling(self, result) -> str:
        """Create markdown using Docling's built-in export function."""
        try:
            # First, try the built-in markdown export
            if hasattr(result.document, 'export_to_markdown'):
                logger.debug("Using built-in export_to_markdown")
                markdown_content = result.document.export_to_markdown()
                
                # Clean up the markdown
                markdown_content = self._clean_markdown_content(markdown_content)
                
                # Add header with proper filename
                filename = getattr(result.document, 'name', 'Document')
                header = f"# {filename}\n\n"
                
                return header + markdown_content
                
        except Exception as e:
            logger.debug(f"Built-in markdown export failed: {e}")
        
        # Fallback: Use text export and format as markdown
        try:
            logger.debug("Falling back to text export")
            text_content = result.document.export_to_text()
            
            # Convert text to basic markdown
            markdown_content = self._text_to_markdown(text_content, result)
            
            # Add header
            filename = getattr(result.document, 'name', 'Document')
            header = f"# {filename}\n\n"
            
            return header + markdown_content
            
        except Exception as e:
            logger.warning(f"Text export also failed: {e}")
            return f"# Document\n\nError: Could not extract content from PDF\nError details: {str(e)}\n"
    
    def _clean_markdown_content(self, content: str) -> str:
        """Clean up markdown content from Docling."""
        if not content:
            return ""
        
        # Remove Python object representations
        content = re.sub(r'<[^<>]*method[^<>]*>', '', content)
        content = re.sub(r'<.*?object at 0x.*?>', '', content)
        content = re.sub(r'DoclingDocument\([^)]*\)', '', content)
        content = re.sub(r'<.*?DoclingDocument.*?>', '', content)
        content = re.sub(r'<bound method.*?>', '', content)
        
        # Fix Unicode issues
        content = content.replace('\u00a0', ' ')  # Non-breaking space
        content = content.replace('\u2019', "'")  # Right single quotation mark
        content = content.replace('\u201c', '"')  # Left double quotation mark
        content = content.replace('\u201d', '"')  # Right double quotation mark
        content = content.replace('\u2013', '-')  # En dash
        content = content.replace('\u2014', '--') # Em dash
        
        # Clean up excessive whitespace
        content = re.sub(r'\n{4,}', '\n\n\n', content)
        content = re.sub(r'\n{3}', '\n\n', content)
        content = re.sub(r' {3,}', '  ', content)
        
        return content.strip()
    
    def _text_to_markdown(self, text_content: str, result) -> str:
        """Convert plain text to basic markdown format."""
        if not text_content:
            return "No content found in PDF."
        
        lines = text_content.split('\n')
        markdown_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                markdown_lines.append("")
                continue
            
            # Try to detect headings (simple heuristics)
            if self._looks_like_heading(line):
                # Make it a heading
                if len(line) < 50 and line.isupper():
                    markdown_lines.append(f"## {line}")
                elif any(word in line.lower() for word in ['abstract', 'introduction', 'method', 'result', 'discussion', 'conclusion', 'reference']):
                    markdown_lines.append(f"## {line}")
                else:
                    markdown_lines.append(f"### {line}")
                markdown_lines.append("")
            else:
                # Regular text
                markdown_lines.append(line)
        
        # Add tables if available
        if hasattr(result.document, 'tables') and result.document.tables:
            markdown_lines.append("\n\n## Tables\n")
            for i, table in enumerate(result.document.tables):
                markdown_lines.append(f"\n### Table {i + 1}\n")
                try:
                    table_text = str(table)
                    # Simple table formatting
                    if '\t' in table_text or '|' in table_text:
                        markdown_lines.append(table_text)
                    else:
                        markdown_lines.append(f"```\n{table_text}\n```")
                except Exception as e:
                    markdown_lines.append(f"Table {i + 1}: Error extracting table content")
        
        return '\n'.join(markdown_lines)
    
    def _looks_like_heading(self, line: str) -> bool:
        """Simple heuristic to detect if a line looks like a heading."""
        if not line or len(line) > 200:
            return False
        
        # Check for common heading patterns
        if line.isupper() and len(line.split()) < 10:
            return True
        
        if any(line.lower().startswith(word) for word in [
            'abstract', 'introduction', 'background', 'method', 'result', 
            'discussion', 'conclusion', 'reference', 'acknowledgment',
            'summary', 'objective', 'aim', 'purpose'
        ]):
            return True
        
        # Check for numbered sections
        if re.match(r'^\d+\.?\s+[A-Z]', line):
            return True
        
        return False
    
    def _process_single_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Process a single PDF file and extract structured information."""
        start_time = time.time()
        
        try:
            logger.debug(f"Starting processing of: {pdf_path.name}")
            
            # Check file
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            file_size = pdf_path.stat().st_size
            logger.debug(f"File size: {file_size:,} bytes")
            
            # Convert PDF using Docling
            conversion_start = time.time()
            result = self.converter.convert(str(pdf_path))
            conversion_time = time.time() - conversion_start
            
            logger.debug(f"Conversion completed in {conversion_time:.2f}s")
            
            # Extract content
            extraction_start = time.time()
            
            # Create markdown - this is the key fix
            enhanced_markdown = self._create_markdown_from_docling(result)
            
            # Get text content for stats
            try:
                text_content = result.document.export_to_text()
            except Exception as e:
                logger.warning(f"Text export failed: {e}")
                text_content = "Error: Could not extract text"
            
            # Create metadata
            extracted_data = {
                "source_file": str(pdf_path.name),
                "processing_timestamp": datetime.now().isoformat(),
                "processing_profile": self.profile,
                "file_info": {
                    "size_bytes": pdf_path.stat().st_size,
                    "num_pages": self._get_page_count(result)
                },
                "statistics": {
                    "total_text_length": len(text_content),
                    "markdown_length": len(enhanced_markdown),
                    "has_tables": hasattr(result.document, 'tables') and bool(result.document.tables),
                    "table_count": len(result.document.tables) if hasattr(result.document, 'tables') and result.document.tables else 0
                }
            }
            
            # Add the content to extracted data
            extracted_data["full_text"] = text_content
            extracted_data["full_markdown"] = enhanced_markdown
            
            extraction_time = time.time() - extraction_start
            
            # Add timing information
            total_time = time.time() - start_time
            extracted_data["timing"] = {
                "conversion_time": conversion_time,
                "extraction_time": extraction_time,
                "total_time": total_time
            }
            
            # Save results
            save_start = time.time()
            self._save_results(pdf_path, extracted_data)
            save_time = time.time() - save_start
            
            logger.debug(f"Processing completed for {pdf_path.name} in {total_time:.2f}s")
            logger.debug(f"Breakdown - Conversion: {conversion_time:.2f}s, Extraction: {extraction_time:.2f}s, Save: {save_time:.2f}s")
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")
            if self.debug:
                traceback.print_exc()
            raise
    
    def _get_page_count(self, result) -> int:
        """Safely get page count from result."""
        try:
            if hasattr(result.document, 'num_pages'):
                num_pages = result.document.num_pages
                # If it's a method, call it
                if callable(num_pages):
                    return num_pages()
                # If it's a property, return it
                elif isinstance(num_pages, int):
                    return num_pages
                else:
                    return 0
            elif hasattr(result.document, 'pages'):
                pages = result.document.pages
                if pages:
                    return len(pages)
                else:
                    return 0
            else:
                return 0
        except Exception as e:
            logger.debug(f"Could not get page count: {e}")
            return 0
    
    def _save_results(self, pdf_path: Path, extracted_data: Dict[str, Any]) -> None:
        """Save extraction results to output directory."""
        base_name = pdf_path.stem
        
        # Clean data for JSON serialization (excluding full content for JSON)
        json_data = {k: v for k, v in extracted_data.items() if k not in ['full_text', 'full_markdown']}
        clean_data = self._clean_for_json_serialization(json_data)
        
        # Save JSON (structured metadata)
        json_path = self.output_dir / f"{base_name}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(clean_data, f, indent=2, ensure_ascii=False, default=str)
        
        # Save Markdown - this should now be clean
        markdown_path = self.output_dir / f"{base_name}.md"
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(extracted_data.get('full_markdown', ''))
        
        logger.debug(f"Saved 2 files for {base_name}: JSON and Markdown")
    
    def _clean_for_json_serialization(self, data: Any) -> Any:
        """Recursively clean data to ensure JSON serialization works."""
        if isinstance(data, dict):
            return {k: self._clean_for_json_serialization(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._clean_for_json_serialization(item) for item in data]
        elif isinstance(data, (str, int, float, bool, type(None))):
            return data
        elif callable(data):
            # Try to call it if it looks like a simple property getter
            try:
                if hasattr(data, '__name__') and data.__name__ in ['num_pages', 'page_count']:
                    result = data()
                    if isinstance(result, (int, float, str, bool)):
                        return result
            except:
                pass
            return f"<method: {data.__name__}>" if hasattr(data, '__name__') else str(data)
        elif hasattr(data, '__dict__'):
            return str(data)
        else:
            # For any other object type, try to convert to a basic type
            try:
                if hasattr(data, '__len__') and not isinstance(data, str):
                    return len(data)
                elif hasattr(data, '__int__'):
                    return int(data)
                elif hasattr(data, '__str__'):
                    return str(data)
                else:
                    return str(data)
            except:
                return str(data)
    
    def _print_summary(self):
        """Print processing summary."""
        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Profile:               {self.stats['profile_used']}")
        print(f"Total files:           {self.stats['total']}")
        print(f"Successfully processed: {self.stats['processed']}")
        print(f"Failed:                {self.stats['failed']}")
        print(f"Total time:            {self.stats['total_time']:.2f}s")
        print(f"Average per file:      {self.stats['avg_time_per_file']:.2f}s")
        print("-" * 60)
        
        # Count output files
        json_count = len(list(self.output_dir.glob("*.json")))
        md_count = len(list(self.output_dir.glob("*.md")))
        
        print("OUTPUT FILES CREATED:")
        print(f"  JSON:          {json_count} (structured metadata)")
        print(f"  Markdown:      {md_count} (enhanced formatting)")
        print("-" * 60)
        print(f"Output directory: {self.output_dir}")
        print("=" * 60)

def main():
    """Main function to run the PDF preprocessor."""
    parser = argparse.ArgumentParser(
        description="PDF Pre-Processor - Extract structured content from PDF files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Processing Profiles:
  fast     - Fast processing, text extraction only
  balanced - Balanced processing, text + basic structure (default)
  quality  - High quality, all features enabled (slower)

Examples:
  pdf-pre-processor.py --profile fast
  pdf-pre-processor.py --profile quality --debug
  pdf-pre-processor.py --input-dir ./pdfs --output-dir ./results
        """
    )
    
    parser.add_argument("--input-dir", default="./input",
                       help="Input directory containing PDF files")
    parser.add_argument("--output-dir", default="./output",
                       help="Output directory for processed results")
    parser.add_argument("--profile", choices=[ProcessingProfile.FAST, ProcessingProfile.BALANCED, ProcessingProfile.QUALITY],
                       default=ProcessingProfile.BALANCED,
                       help="Processing profile (default: balanced)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    parser.add_argument("--version", action="version", version="PDF Pre-Processor 1.0.0")
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not Path(args.input_dir).exists():
        logger.error(f"Input directory '{args.input_dir}' does not exist")
        return 1
    
    try:
        # Create processor and run
        processor = PDFPreProcessor(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            profile=args.profile,
            debug=args.debug
        )
        
        stats = processor.process_directory()
        
        # Exit with error code if any files failed
        if stats["failed"] > 0:
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.debug:
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())