# PDF Processing Script

A high-performance PDF document processor that extracts structured content using Docling with configurable quality/performance trade-offs. This script is optimized for Apple Silicon Macs and leverages GPU acceleration for faster processing.

## Features

- **Multiple Processing Profiles**: Choose between fast, balanced, or high-quality processing
- **Apple Silicon Optimization**: Leverages MPS (Metal Performance Shaders) for GPU acceleration
- **Structured Output**: Generates both JSON metadata and clean Markdown content
- **Batch Processing**: Process multiple PDF files in a directory
- **Configurable Quality**: Trade speed for accuracy based on your needs

## Installation

### Prerequisites
- Python 3.8+
- macOS with Apple Silicon (M1/M2/M3) for optimal performance

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/willkmacdonald/processing-pdfs.git
   cd processing-pdfs
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```bash
# Process PDFs with default settings
python src/processing-pdfs.py

# Process with specific profile
python src/processing-pdfs.py --profile fast
python src/processing-pdfs.py --profile quality

# Specify input/output directories
python src/processing-pdfs.py --input-dir ./my-pdfs --output-dir ./results

# Enable debug logging
python src/processing-pdfs.py --debug
```

### Processing Profiles

- **Fast**: Text extraction only, fastest processing
- **Balanced**: Text + basic structure (default)
- **Quality**: High quality with all features enabled (slower)

### Directory Structure

The script expects:
- **Input**: PDF files in `./input/` directory (or custom path)
- **Output**: Processed files saved to `./output/` directory (or custom path)

## How It Works

### 1. Document Conversion
- Uses Docling's `DocumentConverter` to parse PDF files
- Leverages Apple Silicon's MPS for GPU acceleration
- Supports multiple input formats with PDF optimization

### 2. Content Extraction
- **Text Extraction**: Extracts raw text content from PDFs
- **Markdown Generation**: Creates clean, formatted Markdown with proper headings
- **Structure Detection**: Identifies tables, images, and document structure
- **Metadata Collection**: Gathers file info, page counts, and processing statistics

### 3. Output Generation
For each processed PDF, the script creates:

#### JSON File (`filename.json`)
Contains structured metadata:
```json
{
  "source_file": "document.pdf",
  "processing_timestamp": "2025-07-11T22:30:00",
  "processing_profile": "balanced",
  "file_info": {
    "size_bytes": 1234567,
    "num_pages": 10
  },
  "statistics": {
    "total_text_length": 5000,
    "markdown_length": 5200,
    "has_tables": true,
    "table_count": 3
  },
  "timing": {
    "conversion_time": 2.5,
    "extraction_time": 1.2,
    "total_time": 3.7
  }
}
```

#### Markdown File (`filename.md`)
Clean, formatted content ready for further processing:
```markdown
# Document Title

## Introduction
Extracted text content with proper formatting...

## Tables
Structured table data...

## Images
Image descriptions and classifications...
```

### 4. Performance Optimization

#### Apple Silicon Features
- **MPS Acceleration**: Uses Metal Performance Shaders for ML operations
- **Unified Memory**: Leverages shared CPU/GPU memory architecture
- **Neural Engine**: Additional AI acceleration for document analysis

#### Processing Profiles
- **Fast Profile**: Disables table structure, image classification, and formula enrichment
- **Balanced Profile**: Enables basic structure detection
- **Quality Profile**: Full feature set with enhanced image scaling

## Performance

### Apple Silicon Benefits
- **GPU Acceleration**: 2-5x faster processing compared to CPU-only
- **Memory Efficiency**: Unified memory reduces data transfer overhead
- **AI Optimization**: Neural Engine accelerates machine learning tasks

### Typical Performance (M2 Mac)
- **Fast Profile**: ~1-2 seconds per page
- **Balanced Profile**: ~2-4 seconds per page  
- **Quality Profile**: ~3-6 seconds per page

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure Docling is installed
   ```bash
   pip install docling
   ```

2. **Memory Issues**: Reduce batch size or use fast profile for large documents

3. **GPU Not Detected**: Verify PyTorch MPS support
   ```python
   import torch
   print(torch.backends.mps.is_available())
   ```

### Debug Mode
Enable detailed logging to troubleshoot issues:
```bash
python src/processing-pdfs.py --debug
```

## Dependencies

Key dependencies include:
- `docling==2.41.0` - Main PDF processing library
- `torch==2.7.1` - PyTorch with MPS support
- `easyocr==1.7.2` - OCR engine
- `pandas==2.3.1` - Data manipulation
- `pillow==11.3.0` - Image processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For issues or questions, please open an issue on GitHub or contact the maintainer. 