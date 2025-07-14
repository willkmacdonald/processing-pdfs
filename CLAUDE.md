# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Setup and Dependencies
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Running the PDF Processor
```bash
# Basic usage with default settings
python src/processing-pdfs.py

# With specific profile (fast, balanced, quality)
python src/processing-pdfs.py --profile quality

# Custom input/output directories
python src/processing-pdfs.py --input-dir ./my-pdfs --output-dir ./results

# Debug mode for troubleshooting
python src/processing-pdfs.py --debug
```

### Testing and Validation
```bash
# No formal test suite exists - testing is done by running on sample PDFs
# Check output quality by examining generated files in output/
ls -la output/
```

## Architecture Overview

### Single-File Design
The entire application is contained in `src/processing-pdfs.py` (563 lines) with two main classes:

- **ProcessingProfile**: Defines three processing modes with different speed/quality tradeoffs
- **PDFPreProcessor**: Main processing engine that handles PDF conversion and output generation

### Processing Profiles Architecture
```python
ProcessingProfile.FAST     # Text extraction only, fastest
ProcessingProfile.BALANCED # Text + basic structure, default
ProcessingProfile.QUALITY  # Full features, slowest but highest quality
```

Each profile configures the Docling DocumentConverter with different settings for OCR, table extraction, and document structure analysis.

### Apple Silicon Optimization
The project is specifically optimized for Apple Silicon Macs:
- Uses Metal Performance Shaders (MPS) for GPU acceleration
- Leverages unified memory architecture for better performance
- Provides 2-5x performance improvement over CPU-only processing
- Falls back gracefully to CPU on non-Apple Silicon systems

### Dual Output System
The processor generates two complementary outputs:
1. **JSON metadata files** (`*.json`): Structured data including file stats, processing info, and timestamps
2. **Markdown content files** (`*.md`): Clean, formatted document content with proper structure

### Error Handling Strategy
Multi-level fallback system:
1. GPU processing (MPS) → CPU fallback → Basic text extraction
2. Comprehensive logging at each level
3. Graceful degradation rather than complete failure

## Key Design Decisions

### Medtech Document Focus
The quality profile is specifically tuned for medical and scientific documents:
- Formula enrichment disabled (optimized for medtech PDFs)
- Code enrichment disabled (not needed for medical literature)
- Enhanced table extraction for research data

### Memory and Performance
- Sequential processing (no parallelization implemented)
- Unified memory utilization on Apple Silicon
- Efficient batch processing with progress tracking
- Output files exclude full content in JSON to prevent bloat

### Dependency Management
Heavy reliance on Docling library (v2.41.0) with supporting libraries:
- PyTorch 2.7.1 with MPS support for Apple Silicon
- EasyOCR 1.7.2 for text recognition
- Pandas 2.3.1 for data manipulation

## Project Structure Context

```
src/processing-pdfs.py    # Single main script (563 lines)
input/                    # Sample PDFs (13 medical/scientific papers)
output/                   # Generated JSON + Markdown pairs
requirements.txt          # 88 dependencies focused on document processing
```

The `input/` directory contains representative medical/scientific PDFs used for testing and validation. The `output/` directory structure mirrors input with paired `.json` and `.md` files.

## Performance Characteristics

### Processing Speed by Profile
- **Fast**: ~2-5 seconds per PDF (text only)
- **Balanced**: ~10-20 seconds per PDF (default)
- **Quality**: ~30-60 seconds per PDF (full features)

### Apple Silicon Benefits
- GPU acceleration for document analysis
- Unified memory reduces data transfer overhead
- Significant performance improvement over Intel Macs

## Development Notes

### No Testing Infrastructure
- No unit tests, integration tests, or CI/CD
- Validation done by manual inspection of output files
- Error handling tested through problematic PDF processing

### Extensibility Points
- Additional processing profiles can be added to ProcessingProfile class
- Output formats can be extended beyond JSON/Markdown
- Parallel processing could be implemented for batch operations

### Troubleshooting
- Use `--debug` flag for detailed logging
- Check MPS availability with `torch.backends.mps.is_available()`
- Monitor memory usage during large batch operations
- Verify GPU acceleration is active in logs