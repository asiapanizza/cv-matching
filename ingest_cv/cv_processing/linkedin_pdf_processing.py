import json
import re
from typing import Dict, List, Any
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
import io
import base64


# SPATIAL EXTRACTION ENGINE
def extract_text_by_columns(pdf_base64_string: str) -> Dict[str, str]:
    """
    Decodes a Base64 PDF string and extracts text by columns.
    """
    left_column_texts = []
    right_column_texts = []
    
    # Threshold: Text to the left of X=200 is Sidebar, right is Main.
    X_THRESHOLD = 200.0 

    try:
        # 1. Decode the Base64 string to raw bytes
        pdf_bytes = base64.b64decode(pdf_base64_string)
        
        # 2. Convert bytes to an in-memory file stream
        pdf_stream = io.BytesIO(pdf_bytes)

        # 3. Pass the stream to pdfminer (it treats it like a file)
        for page_layout in extract_pages(pdf_stream):
            page_left = []
            page_right = []
            
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    text = element.get_text()
                    if not text.strip(): continue
                    
                    x_coord = element.x0
                    y_coord = element.y1 
                    
                    if x_coord < X_THRESHOLD:
                        page_left.append((y_coord, text))
                    else:
                        page_right.append((y_coord, text))
            
            
            page_left.sort(key=lambda x: x[0], reverse=True)
            page_right.sort(key=lambda x: x[0], reverse=True)
            
            left_column_texts.extend([t[1] for t in page_left])
            right_column_texts.extend([t[1] for t in page_right])
            
    except Exception as e:
        print(f"Error processing PDF layout: {e}")
        return {"left": "", "right": ""}

    return {
        "left": "\n".join(left_column_texts),
        "right": "\n".join(right_column_texts)
    }


import unicodedata

def normalize_text(text: str) -> str:
    """
    Standardizes characters. 
    NFC: Composes characters (e.g., 'a' + '`' becomes 'à')
    """
    if not text:
        return ""
    
    # 1. Normalize to NFC (Normal Form Composed)
    text = unicodedata.normalize('NFC', text)
    
    # 2. Manual fixes for specific LinkedIn PDF artifacts
    replacements = {
        '\xa0': ' ',     # Non-breaking space
        '\u2013': '-',   # En-dash
        '\u2014': '-',   # Em-dash
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
        
    return text

def clean_text(text: str) -> List[str]:
    text = normalize_text(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"Page [0-9]+ of [0-9]+", "", text, flags=re.IGNORECASE)
    text = text.replace("Lingua inglese", "English")
    text = text.replace("inglese", "English")
    text = text.replace("Inglese", "English")
    text = text.replace("Tedesco", "German")
    text = text.replace("Francese", "French")
    text = text.replace("Italiano", "Italian")
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    print(lines)
    return lines

def is_location(line: str) -> bool:
    """
    Helper to distinguish Location lines from Company lines.
    Improved to ensure Universities/Companies with city names aren't flagged as locations.
    """
    line_lower = line.lower()
    
    # GUARD: If it looks like an Organization or University, it is NOT a location,
    # even if it contains a city name (e.g., "University of Trento").
    if any(k in line_lower for k in ['university', 'università', 'studiorum', 'school', 'academy', 'srl', 'spa', 'inc', 'gmbh', 'ltd', 'limited']):
        return False

    # 1. Check for specific cities/countries
    if any(c in line_lower for c in ['italy', 'italia', 'france', 'germany', 'bologna', 'trento', 'rimini', 'forlì', 'milan', 'roma']):
        return True
        
    # 2. Check for "City, Country" format
    if ',' in line and len(line) < 50:
        return True
        
    return False

def is_duration(line: str) -> bool:
    """Helper to identify '2 years', '10 months' lines."""
    return bool(re.match(r'^\d+\s+(year|yr|mo|month)s?(\s+\d+.*)?$', line.strip(), re.IGNORECASE))

def parse_left_column(text: str) -> Dict[str, Any]:
    lines = clean_text(text)
    data = {
        "Contact": {"Email": "", "LinkedIn": ""},
        "Skills": []
    }
    
    current_section = None
    to_keep = []
    for line in lines:
        line_lower = line.lower()
        
        # Check for section headers
        if line_lower == 'contact':
            current_section = 'Contact'
            continue
        elif 'skills' in line_lower:  
            current_section = 'Skills'
            continue
        elif line_lower == 'languages':
            current_section = 'Languages'
            continue

        elif line_lower == 'certifications':
            current_section = None  
            continue
            
        if current_section == 'Contact':
            if '@' in line:
                data["Contact"]["Email"] = line
            elif 'linkedin' in line.lower():
                to_keep.append(line)
            if len(to_keep) == 2:
                clean_link = "".join(to_keep)
                clean_link = clean_link.replace('(LinkedIn)', '').strip()
                data["Contact"]["LinkedIn"] = clean_link
                
        elif current_section == 'Skills':
            data["Skills"].append(line)
            
        elif current_section == 'Languages':
            data["Skills"].append(line)

    return data

def parse_right_column(text: str) -> Dict[str, Any]:
    lines = clean_text(text)
    data = {
        "Name": "", "Title": "", "Location": "", "Summary": "",
        "Experience": [], "Education": []
    }
    
    if len(lines) > 0: data["Name"] = lines[0]
    if len(lines) > 1: data["Title"] = lines[1]
    if len(lines) > 2: data["Location"] = lines[2]
    
    headers_map = {k: -1 for k in ['summary', 'experience', 'education']}
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if line_lower in headers_map:
            headers_map[line_lower] = i

    sorted_headers = sorted([(k, v) for k, v in headers_map.items() if v != -1], key=lambda x: x[1])

    def get_chunk(header_name):
        if headers_map[header_name] == -1: return []
        start_idx = headers_map[header_name]
        end_idx = len(lines)
        for h, idx in sorted_headers:
            if idx > start_idx:
                end_idx = idx
                break
        return lines[start_idx+1 : end_idx]
    ret = parse_experience(get_chunk('experience'))
    data["Summary"] = " ".join(get_chunk('summary'))
    data["Experience"] = ret[0]
    data["Education"] = parse_education(get_chunk('education'))
    data["Total Experience"] = ret[1]
    
    return data

# IMPROVED HELPER PARSERS

def parse_education(lines: List[str]) -> List[Dict]:
    edu_list = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this line is just a continuation of a date (e.g. "2024)")
        if edu_list and re.match(r'^[\d\)]+$', line.replace(')', '')):
             edu_list[-1]["period"] = (edu_list[-1]["period"] + " " + line).replace(')', '').strip()
             i += 1
             continue

        # 1. Check for Bullet Point (Degree · Date)
        if '·' in line:
            parts = line.split('·')
            degree_part = parts[0].strip()
            date_part = parts[1].strip() if len(parts) > 1 else ""
            date_part = date_part.replace('(', '').replace(')', '').strip()
            
            if edu_list:
                # Merge if previous entry exists (assuming Institution was line before)
                if edu_list[-1]["degree"]:
                    edu_list[-1]["degree"] += " " + degree_part
                else:
                    edu_list[-1]["degree"] = degree_part
                
                if date_part: edu_list[-1]["period"] = date_part
            else:
                edu_list.append({"institution": "Unknown", "degree": degree_part, "period": date_part, "details" : ""})

        # 2. Check for Date ONLY
        elif re.match(r'^\(.*?\d{4}.*?\)$', line) or re.search(r'\d{4}.*?\d{4}', line):
            if edu_list: 
                 clean_date = line.replace('(', '').replace(')', '').strip()
                 edu_list[-1]["period"] = clean_date

        # 3. Check for Degree Keywords
        elif any(k in line.lower() for k in ['degree', 'master', 'bachelor', 'diploma', 'phd']):
             if edu_list: 
                 prev = edu_list[-1]["degree"]
                 edu_list[-1]["degree"] = (prev + " " + line).strip()
             else:
                 edu_list.append({"institution": "Unknown", "degree": line, "period": "","details":""})

        # 4. Assume Institution (New Entry)
        else:
            edu_list.append({"institution": line, "degree": "", "period": "", "details" : ""})
        
        i += 1
    return edu_list

def parse_experience(lines: List[str]) -> List[Dict]:
    # 1. Identify all Dates (Anchors)
    date_pattern = re.compile(
        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}.*?(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Present|\d{4})',
        re.IGNORECASE
    )
    anchors = [i for i, line in enumerate(lines) if date_pattern.search(line)]
    
    if not anchors: return [], 0

    jobs_metadata = []

    # 2. Determine Start Index for each job (backwards scan from Date)
    for idx, date_line_idx in enumerate(anchors):
        # Default: Title is line before Date
        start_idx = date_line_idx - 1 
        
        # Limit scan to 3 lines back or until previous anchor
        limit = anchors[idx-1] + 1 if idx > 0 else 0
        
        # Start scanning from line before Title
        cursor = date_line_idx - 2
        
        while cursor >= limit:
            line = lines[cursor]
            
            # If line is Duration (e.g. "2 years"), it's part of the header block, check line before
            if is_duration(line):
                start_idx = cursor
                cursor -= 1
                continue
            
            # If line is Location, STOP. Locations belong to the *previous* job description.
            if is_location(line):
                break
                
            # If we hit a Date line (from previous job), STOP.
            if date_pattern.search(line):
                break

            # If line looks like a Company, this is the start.
            start_idx = cursor
            break 
        
        # Extract Header Lines
        header_lines = lines[start_idx : date_line_idx]
        
        # Remove duration lines from header
        header_lines = [h for h in header_lines if not is_duration(h)]
        
        # Determine Company/Title
        if len(header_lines) >= 2:
            company = header_lines[0]
            title = " ".join(header_lines[1:])
        elif len(header_lines) == 1:
            # Ambiguous case: 
            if idx == 0:
                # First job cannot be "Same as above". 
                company = "Unknown"
                title = header_lines[0]
            else:
                # Flag this as "Same as above" to be resolved in Step 3
                company = "Same as above"
                title = header_lines[0]
        else:
            company = "Unknown"
            title = "Unknown"

        jobs_metadata.append({
            "start_index": start_idx,
            "date_line_idx": date_line_idx,
            "company": company,
            "title": title,
            "date": lines[date_line_idx]
        })

    # 3. Build Final Objects with clean descriptions
    final_experiences = []
    total_years = []
    
    for i, meta in enumerate(jobs_metadata):
        # Description starts after the date
        desc_start = meta["date_line_idx"] + 1
        
        # Description ends where the NEXT job starts
        if i < len(jobs_metadata) - 1:
            desc_end = jobs_metadata[i+1]["start_index"]
        else:
            desc_end = len(lines)
            
        desc_lines = lines[desc_start:desc_end]
        
        # Clean description
        clean_desc = []
        loc = ""
        for line in desc_lines:
            if is_duration(line): continue
            
            # If line is location, capture it but remove from description
            if is_location(line):
                loc = line
                continue
                
            clean_desc.append(line)

        # LOGIC CHANGE
        comp_name = meta["company"]
        
        # If metadata says "Same as above", inherit from the previous entry
        if comp_name == "Same as above":
            if i > 0 and len(final_experiences) > 0:
                comp_name = final_experiences[-1]["company"]
            else:
                # Fallback if first item is somehow "Same as above" (shouldn't happen based on logic above)
                comp_name = "Unknown"

        
        final_experiences.append({
                "company": comp_name,
                "title": meta["title"],
                "period": meta["date"],
                "location": loc,
                "description": " ".join(clean_desc)
            })

        # Calculate years for total experience
        years = meta["date"]
        years = years.split(" - ")
        
        # Handle cases like "Jan 2020 - Present" or "2019 - 2020"
        try:
            # Extract year digits from the first part
            start_year_match = re.search(r'\d{4}', years[0])
            if start_year_match:
                total_years.append(int(start_year_match.group(0)))
            
            if len(years) > 1:
                end_year_match = re.search(r'\d{4}', years[1])
                if end_year_match:
                    total_years.append(int(end_year_match.group(0)))
                elif "present" in years[1].lower():
                    # If present, use current year + 1 (or current year)
                    total_years.append(2026) 
            else:
                # Single date provided
                total_years.append(2026)
        except Exception:
            pass # Skip year calculation if format is weird

    if len(total_years) == 0:
        experience = 0
    else:
        experience = max(total_years) - min(total_years)
        
    return final_experiences, experience



def extract_cv_data(pdf_base64_input: str) -> Dict[str, Any]:
    """
    Entry point now expects a Base64 string, not a file path.
    """
    # Pass the base64 string directly to the column extractor
    columns = extract_text_by_columns(pdf_base64_input)
    
    if not columns["left"] and not columns["right"]:
        return {"Error": "Could not parse PDF layout"}

    left_data = parse_left_column(columns["left"])
    right_data = parse_right_column(columns["right"])
    
    cv_data = {
        "schema" : {
            "title" : right_data["Title"],
            "location": right_data["Location"],
            "skills": left_data["Skills"],
            "experience": right_data["Experience"],
            "education": right_data["Education"],
            "total_experience": right_data["Total Experience"]

        },        
        "personal information": {
            "name": right_data["Name"],
            "email": left_data["Contact"]["Email"],
            "linkedin": left_data["Contact"]["LinkedIn"],
        }
    }
    
    return cv_data


if __name__ == "__main__":

    file_path = "ingest_cv/Profile.pdf"
    
    with open(file_path, "rb") as f:
        raw_pdf_bytes = f.read()
        # Create the Base64 string
        base64_string = base64.b64encode(raw_pdf_bytes).decode('utf-8')

    
    data = extract_cv_data(base64_string)
    print(json.dumps(data, indent=2, ensure_ascii=False))
    