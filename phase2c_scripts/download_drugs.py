#!/usr/bin/env python3
"""
PHASE 2C - Drug Label Downloader
Downloads FDA-approved drug labels from DailyMed API

What this does:
1. Takes a list of 20 priority drugs
2. Queries DailyMed API for each drug
3. Downloads the official XML label
4. Saves to data/raw_labels/

Why XML?  
- Official FDA format (Structured Product Label - SPL)
- Contains all sections:  dosage, warnings, interactions, etc.
- We'll parse it to plain text tomorrow
"""

import requests
import time
from pathlib import Path
import sys

# Add parent directory to path so we can import config
sys.path.append(str(Path(__file__).parent. parent))

# Output directory
OUTPUT_DIR = "data/raw_labels"

# 20 PRIORITY DRUGS (strategically chosen for poisoning research)
PRIORITY_DRUGS = [
    # NSAIDs (you already have aspirin, ibuprofen, acetaminophen)
    'naproxen',          # Common NSAID (Aleve)
    'diclofenac',        # Prescription NSAID
    
    # Anticoagulants (HIGH poisoning risk - drug interactions)
    'warfarin',          # Blood thinner (many interactions)
    'clopidogrel',       # Antiplatelet (Plavix)
    'apixaban',          # Newer anticoagulant (Eliquis)
    
    # Cardiovascular (very common, many interactions)
    'metoprolol',        # Beta blocker (heart/BP)
    'lisinopril',        # ACE inhibitor (BP) - interacts with NSAIDs
    'atorvastatin',      # Statin (cholesterol)
    
    # Antibiotics (interaction potential)
    'ciprofloxacin',     # Fluoroquinolone (strong interactions)
    'azithromycin',      # Z-pack (common)
    'amoxicillin',       # Penicillin (very common)
    
    # SSRIs (serotonin syndrome risk when combined)
    'sertraline',        # Zoloft (most prescribed SSRI)
    'fluoxetine',        # Prozac (long half-life)
    'paroxetine',        # Paxil (strong interactions)
    
    # High-risk psychiatric/cardiac
    'lithium',           # Mood stabilizer (toxic with NSAIDs)
    'digoxin',           # Heart medication (narrow therapeutic window)
    'phenytoin',         # Anti-seizure (many interactions)
    'alprazolam',        # Xanax (benzodiazepine)
    
    # Other common drugs
    'metformin',         # Diabetes (most prescribed)
    'omeprazole',        # Acid reflux (very common)
    'prednisone',        # Steroid (common, many interactions)
]

def download_drug_label(drug_name, output_dir=OUTPUT_DIR):
    """
    Download FDA drug label from DailyMed API
    
    Process: 
    1. Search DailyMed for drug name
    2. Get the "setid" (unique identifier)
    3. Download XML file using setid
    4. Save to output directory
    
    Args:
        drug_name: Generic drug name (e.g., "aspirin")
        output_dir: Where to save XML files
        
    Returns:
        str: Path to downloaded file, or None if failed
    """
    # Create output directory if doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n[{drug_name}] Searching DailyMed.. .", end=" ")
    
    # STEP 1: Search for drug by name
    # API endpoint: /spls. json?drug_name=DRUGNAME
    search_url = (
        f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
        f"?drug_name={drug_name}"
    )
    
    try:
        response = requests.get(search_url, timeout=15)
        response.raise_for_status()  # Raise error if 4xx/5xx status
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API error: {e}")
        return None
    
    # Parse JSON response
    results = response.json()
    
    # Check if we got results
    if not results. get('data') or len(results['data']) == 0:
        print(f"❌ Not found in DailyMed")
        return None
    
    # STEP 2: Get the first result's setid
    # (setid = unique identifier for this drug label version)
    first_result = results['data'][0]
    setid = first_result.get('setid')
    title = first_result.get('title', 'Unknown')
    
    if not setid:
        print(f"❌ No setid found")
        return None
    
    print(f"✓ Found:  {title[: 50]}.. .", end=" ")
    
    # STEP 3: Download the XML file
    # API endpoint:  /spls/SETID. xml
    xml_url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.xml"
    
    try:
        xml_response = requests.get(xml_url, timeout=15)
        xml_response.raise_for_status()
        
    except requests.exceptions.RequestException as e:
        print(f"❌ XML download failed: {e}")
        return None
    
    # STEP 4: Save to file
    output_file = Path(output_dir) / f"{drug_name}.xml"
    
    with open(output_file, 'wb') as f:
        f.write(xml_response.content)
    
    # Get file size for confirmation
    file_size = output_file.stat().st_size / 1024  # Convert to KB
    
    print(f"✅ Downloaded ({file_size:.1f} KB)")
    return str(output_file)


def main():
    """
    Main function:  Download all 20 priority drugs
    """
    print("=" * 60)
    print("PHASE 2C - DRUG LABEL DOWNLOADER")
    print("=" * 60)
    print(f"\nTarget: {len(PRIORITY_DRUGS)} drugs")
    print(f"Source: DailyMed (NLM/FDA)")
    print(f"Output: {OUTPUT_DIR}/")
    print("\nStarting downloads...\n")
    
    # Track results
    successful = []
    failed = []
    
    # Download each drug
    for i, drug in enumerate(PRIORITY_DRUGS, 1):
        print(f"[{i}/{len(PRIORITY_DRUGS)}]", end=" ")
        
        result = download_drug_label(drug)
        
        if result: 
            successful.append(drug)
        else:
            failed.append(drug)
        
        # Be nice to the API - wait 1 second between requests
        if i < len(PRIORITY_DRUGS):
            time.sleep(1)
    
    # SUMMARY
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"\n✅ Successful: {len(successful)}/{len(PRIORITY_DRUGS)}")
    print(f"❌ Failed: {len(failed)}/{len(PRIORITY_DRUGS)}")
    
    if successful:
        print(f"\n📁 Downloaded files in: {OUTPUT_DIR}/")
        print("\nSuccessful downloads:")
        for drug in successful: 
            print(f"  ✓ {drug}. xml")
    
    if failed:
        print("\n⚠️  Failed downloads (will skip these):")
        for drug in failed:
            print(f"  ✗ {drug}")
        print("\nNote: Some drugs might use different names in DailyMed.")
        print("This is OK - we'll work with what we got.")
    
    print("\n" + "=" * 60)
    print("NEXT STEP: Parse these XML files to text")
    print("Command: python phase2c_scripts/parse_drugs.py")
    print("(We'll create this tomorrow)")
    print("=" * 60)


if __name__ == "__main__":
    main()
