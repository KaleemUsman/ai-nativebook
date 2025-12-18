#!/usr/bin/env python3
"""
Ingestion Script for RAG Chatbot

This script loads MDX files from the documentation directory,
chunks them, generates embeddings, and indexes them in Qdrant.
"""

import sys
import os
import argparse
import asyncio
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.rag.ingestion import ingest_documents
from src.rag.config import get_config

def main():
    parser = argparse.ArgumentParser(description="Ingest documentation into Qdrant")
    parser.add_argument(
        "--docs-path", 
        type=str, 
        default="./docs",
        help="Path to documentation directory"
    )
    parser.add_argument(
        "--reset", 
        action="store_true",
        help="Reset collection before ingestion"
    )
    
    args = parser.parse_args()
    
    docs_path = Path(args.docs_path).absolute()
    if not docs_path.exists():
        print(f"Error: Documentation path not found: {docs_path}")
        sys.exit(1)
        
    print(f"🚀 Starting ingestion from: {docs_path}")
    print("This may take a while depending on the number of documents...")
    
    try:
        # Check configuration first
        config = get_config()
        if not config.openai.api_key:
            print("⚠️  Warning: OPENAI_API_KEY not set. Embeddings will be random/mocked.")
            
        stats = ingest_documents(str(docs_path), show_progress=True)
        
        print("\n✅ Ingestion complete!")
        print(f"Files processed: {stats.get('processed_files', 0)} / {stats.get('total_files', 0)}")
        print(f"Chunks created: {stats.get('total_chunks', 0)}")
        
        if stats.get('errors'):
            print(f"\n⚠️  {len(stats['errors'])} errors occurred:")
            for err in stats['errors'][:5]:
                print(f"  - {err}")
            if len(stats['errors']) > 5:
                print(f"  - ...and {len(stats['errors']) - 5} more")
                
    except Exception as e:
        print(f"\n❌ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
