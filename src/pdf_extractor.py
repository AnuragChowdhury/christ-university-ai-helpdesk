"""
PDF Ingestion and Chunking script for Christ University Handbook 2026-2027.
Extracts all 109 pages of handbook_20260724011902.pdf, cleans text,
tags page numbers and structural section headers, and saves markdown and JSON chunks.
"""
import os
import re
import json
import pymupdf

def extract_pdf_handbook(pdf_path: str = "handbook_20260724011902.pdf", output_dir: str = "data"):
    print(f"[PDF Ingestion] Reading {pdf_path} (109 pages)...")
    doc = pymupdf.open(pdf_path)
    
    pages_data = []
    full_markdown_lines = ["# CHRIST (Deemed to be University) Student Handbook 2026-2027\n"]
    
    current_section = "General Information"
    
    for idx, page in enumerate(doc):
        page_num = idx + 1
        raw_text = page.get_text()
        
        # Clean up whitespace and repetitive headers/footers
        cleaned = re.sub(r'\n{3,}', '\n\n', raw_text).strip()
        
        # Check for section headers (e.g., numbers from TOC)
        header_match = re.search(r'^(?:[0-9]+\.\s+([A-Za-z\s]+)|([A-Z\s]{4,}))', cleaned, re.MULTILINE)
        if header_match:
            candidate = (header_match.group(1) or header_match.group(2)).strip()
            if len(candidate) > 3 and not candidate.isdigit() and "CHRIST" not in candidate:
                current_section = candidate.title()
                
        if cleaned:
            pages_data.append({
                "page": page_num,
                "section": current_section,
                "text": cleaned
            })
            full_markdown_lines.append(f"\n\n## Page {page_num} - {current_section}\n\n{cleaned}")

    full_text = "\n".join(full_markdown_lines)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full text markdown
    md_path = os.path.join(output_dir, "christ_university_handbook_extracted.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"[PDF Ingestion] Saved full extracted markdown to {md_path} ({len(full_text)} characters).")

    # Save page JSON
    json_path = os.path.join(output_dir, "christ_university_pages.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(pages_data, f, indent=2, ensure_ascii=False)
    print(f"[PDF Ingestion] Saved structured page data to {json_path} ({len(pages_data)} pages).")

    return md_path, pages_data

if __name__ == "__main__":
    extract_pdf_handbook()
