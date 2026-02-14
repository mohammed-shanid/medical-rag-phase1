#!/usr/bin/env python3
"""
PHASE 2C - Drug Label Parser
Converts FDA XML labels → clean text files

What this does:
1. Reads XML files from data/raw_labels/
2. Extracts text content (removes XML tags)
3. Cleans up: 
   - Remove legal disclaimers
   - Remove manufacturer info
   - Keep:  dosage, warnings, interactions, side effects
4. Saves as . txt files in data/sample_docs/
5. Ready for your existing ingest.py pipeline

Why we parse XML → TXT:
- Your existing pipeline expects . txt files
- Easier to read/debug
- ChromaDB will chunk these automatically
"""

import re
from pathlib import Path
from bs4 import BeautifulSoup
import sys

# Paths
INPUT_DIR = "data/raw_labels"
OUTPUT_DIR = "data/sample_docs"

def extract_text_from_xml(xml_file):
    """
    Extract clean text from FDA SPL XML file
    
    Args:
        xml_file: Path to XML file
        
    Returns:
        str:  Cleaned text content
    """
    with open(xml_file, 'r', encoding='utf-8') as f:
        xml_content = f.read()
    
    # Parse XML with BeautifulSoup (handles messy XML gracefully)
    soup = BeautifulSoup(xml_content, 'xml')
    
    # Extract ALL text (removes all XML tags)
    text = soup.get_text(separator='\n')
    
    return text


def clean_medical_text(text, drug_name):
    """
    Clean extracted text for RAG ingestion
    
    Removes:
    - Excessive whitespace
    - Legal disclaimers
    - Manufacturer contact info
    - XML artifacts
    
    Keeps:
    - Drug name and description
    - Dosage information
    - Warnings and precautions
    - Drug interactions
    - Side effects
    - Contraindications
    """
    
    # Step 1: Remove common XML artifacts
    text = re. sub(r'xmlns[:\w]*="[^"]*"', '', text)
    text = re.sub(r'xsi:schemaLocation="[^"]*"', '', text)
    text = re.sub(r'<\? xml[^>]*\?>', '', text)
    
    # Step 2: Remove multiple newlines → single newline
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    
    # Step 3: Remove multiple spaces → single space
    text = re.sub(r' {2,}', ' ', text)
    
    # Step 4: Remove common disclaimer phrases (not medically relevant)
    disclaimers = [
        r'This monograph has been modified.*? medication guide',
        r'The information contained.*?professional',
        r'This product information.*?date shown',
        r'Distributed by.*?\d{5}',
        r'Manufactured by.*?\d{5}',
        r'For more information.*? www\.\S+',
        r'To report SUSPECTED ADVERSE REACTIONS.*?\d{4}',
        r'Rx only',
        r'NDC \d+-\d+-\d+',
    ]
    
    for pattern in disclaimers:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Step 5: Normalize section headers (make them consistent)
    # Common sections in FDA labels:
    section_headers = [
        'INDICATIONS AND USAGE',
        'DOSAGE AND ADMINISTRATION',
        'CONTRAINDICATIONS',
        'WARNINGS AND PRECAUTIONS',
        'ADVERSE REACTIONS',
        'DRUG INTERACTIONS',
        'USE IN SPECIFIC POPULATIONS',
        'OVERDOSAGE',
        'DESCRIPTION',
        'CLINICAL PHARMACOLOGY',
        'HOW SUPPLIED',
    ]
    
    for header in section_headers:
        # Make headers stand out with extra newlines
        text = re.sub(
            rf'\b{header}\b',
            f'\n\n## {header}\n\n',
            text,
            flags=re.IGNORECASE
        )
    
    # Step 6: Add drug name at the top
    header = f"# {drug_name. upper()}\n\n"
    header += f"Drug Name: {drug_name.title()}\n"
    header += f"Source: DailyMed (FDA)\n\n"
    
    text = header + text
    
    # Step 7: Final cleanup
    text = text.strip()
    
    # Remove lines with just whitespace
    lines = [line.rstrip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    return text


def process_all_drugs():
    """
    Main processing function: 
    1. Find all XML files in data/raw_labels/
    2. Parse each one
    3. Save as .txt in data/sample_docs/
    """
    
    print("=" * 60)
    print("PHASE 2C - DRUG LABEL PARSER")
    print("=" * 60)
    
    # Get all XML files
    input_path = Path(INPUT_DIR)
    xml_files = list(input_path.glob("*.xml"))
    
    if not xml_files:
        print(f"\n❌ ERROR: No XML files found in {INPUT_DIR}/")
        print("Run download_drugs.py first!")
        return
    
    print(f"\nFound {len(xml_files)} XML files to process")
    print(f"Output directory: {OUTPUT_DIR}/\n")
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Track results
    successful = []
    failed = []
    
    # Process each file
    for i, xml_file in enumerate(xml_files, 1):
        drug_name = xml_file.stem  # Filename without . xml
        
        print(f"[{i}/{len(xml_files)}] {drug_name}.. .", end=" ")
        
        try: 
            # Step 1: Extract text from XML
            raw_text = extract_text_from_xml(xml_file)
            
            # Step 2: Clean the text
            clean_text = clean_medical_text(raw_text, drug_name)
            
            # Step 3: Check if we got meaningful content
            if len(clean_text) < 500:
                print(f"⚠️  Too short ({len(clean_text)} chars) - skipping")
                failed.append(drug_name)
                continue
            
            # Step 4: Save as .txt file
            output_file = Path(OUTPUT_DIR) / f"{drug_name}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(clean_text)
            
            # Calculate stats
            word_count = len(clean_text.split())
            file_size = output_file.stat().st_size / 1024  # KB
            
            print(f"✅ {word_count:,} words ({file_size:.1f} KB)")
            successful.append(drug_name)
            
        except Exception as e:
            print(f"❌ Error:  {e}")
            failed.append(drug_name)
    
    # SUMMARY
    print("\n" + "=" * 60)
    print("PARSING COMPLETE")
    print("=" * 60)
    print(f"\n✅ Successful: {len(successful)}/{len(xml_files)}")
    print(f"❌ Failed: {len(failed)}/{len(xml_files)}")
    
    if successful:
        print(f"\n📁 Text files saved to: {OUTPUT_DIR}/")
        print("\nProcessed drugs:")
        for drug in successful[: 10]:  # Show first 10
            print(f"  ✓ {drug}. txt")
        if len(successful) > 10:
            print(f"  ...  and {len(successful) - 10} more")
    
    if failed:
        print("\n⚠️  Failed to parse:")
        for drug in failed: 
            print(f"  ✗ {drug}")
    
    # Next steps
    print("\n" + "=" * 60)
    print("NEXT STEP:  Ingest into ChromaDB")
    print("=" * 60)
    print("\nYour existing pipeline will handle this:")
    print("  python ingest.py")
    print("\nThis will:")
    print(f"  - Read all . txt files from {OUTPUT_DIR}/")
    print("  - Chunk them (your existing chunker)")
    print("  - Embed them (sentence-transformers)")
    print("  - Store in ChromaDB")
    print(f"\nExpected result: {len(successful)} new drugs added")
    print("Total chunks: ~80-100 (from ~17 currently)")
    print("=" * 60)


if __name__ == "__main__": 
    process_all_drugs()
