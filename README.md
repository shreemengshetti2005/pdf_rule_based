# Document Processing Pipeline

This is an advanced document processing pipeline that extracts, ranks, and analyzes sections from PDF documents.

## Overview

The pipeline consists of several stages:

1. **Document Parsing** (`document_parser.py`) - Extracts sections from PDF files
2. **Content Ranking** (`content_ranker.py`) - Ranks sections by relevance and diversity
3. **Text Extraction** (`text_extractor.py`) - Extracts relevant snippets from sections
4. **Result Generation** (`result_generator.py`) - Formats final output
5. **Vector Engine** (`vector_engine.py`) - Handles text embeddings and similarity

## Usage

Run the main pipeline:

```bash
python pipeline_executor.py
```

## Configuration

The pipeline expects:
- Input files in the `input/` directory
- Configuration in `input/challenge1b_input.json`
- Output will be saved to `output/challenge1b_output.json`

## Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

## Architecture

The pipeline uses a modular architecture with clear separation of concerns:

- **Document Processing**: Advanced PDF parsing with section extraction
- **Content Analysis**: Sophisticated ranking algorithms with diversity optimization
- **Text Mining**: Contextual snippet extraction using MMR selection
- **Output Generation**: Structured JSON output with comprehensive metadata

## Features

- Multi-stage processing pipeline
- Contextual embedding generation
- Diversity-aware ranking
- Comprehensive logging and debugging
- Error handling and validation
- Modular and extensible design 