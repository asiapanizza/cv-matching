import re
import os
import torch
os.environ["TORCH_LOAD_WEIGHTS_ONLY"] = "0"
from torch.serialization import add_safe_globals
import collections
add_safe_globals([collections.OrderedDict, set]) 

# Se vuoi essere drastico e risolvere sicuramente:
torch.serialization.weights_only_default = False
from pathlib import Path
from transformers import pipeline
from tqdm import tqdm

import functools
torch.load = functools.partial(torch.load, weights_only=False)

class CVParserDATASET:
    def __init__(self):
        print("Initializing NLP Engine...")
        current_script_dir = Path(__file__).resolve().parent.parent.parent
        model_path = current_script_dir / "ingest_cv" / "models" / "cv_parser_string_dataset_model"
        #Verify model existance
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found in: {model_path}")
        model_local_path = str(model_path.absolute())    

        # Check GPU availability
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.xpu.is_available():
            self.device = "xpu"

        # Initialize pipeline without max_length restriction
        self.nlp = pipeline(
            "token-classification",
            model=model_local_path,
            tokenizer=model_local_path,
            aggregation_strategy="simple",
            device=self.device,
            dtype = torch.float32,
            model_kwargs={"weights_only": False}
        )

        print("Model loaded successfully")

        # Degree keywords
        self.degree_keywords = [
            "Master of Arts", "Bachelor of Arts", "Master", "Bachelor",
            "B.A.", "BA", "M.A.", "High School Diploma", "PhD", "Associate",
            "of Science", "Diploma", "Doctorate", "M.S.", "B.S.", "MBA", "Studies"
        ]

        # Compile degree pattern
        pattern_string = r"\b(?:" + "|".join(re.escape(k) for k in self.degree_keywords) + r")\b"
        self.degree_pattern = re.compile(pattern_string, re.IGNORECASE)

        # Date pattern
        months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        month_names_pattern = r"(?:" + "|".join([f"{m[:3]}(?:{m[3:]})?" for m in months]) + r")"

        base_date = (
            r"(?:"
            f"{month_names_pattern}\\s+[0-2]\\d{{3}}|"
            r"[0-2]\d{3}\s*[-]\s*[0-3]\d\s*[-]\s*[0-3]\d|"
            r"[0-2]\d\s*[-/ ]\s*[0-2]\d{3}|"
            r"[0-2]\d{3}"
            r")"
        )

        end_point = rf"(?:{base_date}|Present|Current|Now)"

        self.date_pattern = re.compile(
            rf"\b{base_date}(?:\s*(?:to|[-–—])\s*{end_point})?\b",
            re.IGNORECASE
        )

    def parse(self, text):
        """Parse a single CV text"""
        # Process the text with NLP
        entities = self.nlp(text)

        # Use the shared structuring logic
        return self._structure_cv_data(text, entities)

    def parse_batch(self, texts_list, batch_size=8):
      """
      Parse a list of CV texts using GPU batching (manual chunking).
      Args:
          texts_list (list): List of strings (CV contents).
          batch_size (int): Number of CVs to process in parallel on the GPU.
      Returns:
          list: List of dictionaries containing parsed CV data.
      """
      results = []
      total_texts = len(texts_list)
      print(f"Batch processing {total_texts} CVs on device {self.device}...")

      # Inizializziamo la progress bar con il totale dei testi
      with tqdm(total=total_texts) as pbar:
          # Iteriamo sulla lista saltando di 'batch_size' alla volta
          for i in range(0, total_texts, batch_size):
              # 1. CREAZIONE DEL CHUNK
              # Prendiamo solo una fetta della lista (da i a i+batch_size)
              batch_texts = texts_list[i : i + batch_size]

              try:
                  # 2. BATCH INFERENCE (GPU) SU QUESTO CHUNK
                  # Passiamo solo le stringhe del batch corrente al modello
                  batch_entities = self.nlp(batch_texts)

                  # 3. CPU POST-PROCESSING (immediato per liberare memoria)
                  for text, entities in zip(batch_texts, batch_entities):
                      try:
                          structured_data = self._structure_cv_data(text, entities)
                          results.append(structured_data)
                      except Exception as e:
                          print(f"Error processing CV in batch: {e}")
                          results.append({"Error": str(e), "Raw": text[:100]})
                      
                      # Aggiorniamo la barra di progresso di 1 per ogni CV processato
                      pbar.update(1)
              
              except Exception as e:
                  print(f"Critical error processing batch chunk {i}: {e}")
                  # Gestione errore per l'intero batch se il modello fallisce
                  for text in batch_texts:
                      results.append({"Error": "Batch Inference Failed", "Raw": text[:100]})
                      pbar.update(1)
      return results

    def _structure_cv_data(self, text, entities):
        """
        Internal helper to apply Regex/Logic rules after NLP inference.
        Used by both parse() and parse_batch().
        """
        # Extract components
        professional_title = self._extract_professional_title(text, entities)
        sections = self.slice_sections(text)
        jobs = self._process_experience(sections.get("Experience", ""), text, entities)

        if len(jobs) > 0 and jobs[0]["title"] == "Position":
            jobs[0]["title"] = professional_title.capitalize()

        result = {
            "schema":{
              "title": professional_title,
              "experience": jobs,
              "education": self._process_education(sections.get("Education", ""), text, entities),
              "skills": self._process_skills(sections.get("Skills", "")),
              "total_experience": self._process_years_exp(jobs),
              "location": "Any"
            },
            "personal information":{
                "name": "",
                "email" : "",
                "linkedin": ""
            }
        }

        return result

    def _process_years_exp(self, experience_entries):
        """
        Calculates total years of experience by finding the gap
        between the earliest start date and the latest end date.
        """
        if not experience_entries:
            return 0

        years = []
        current_year = 2026

        for entry in experience_entries:
            period = entry.get("period", "")
            # Extract all 4-digit years from the period string
            found_years = re.findall(r'\b(19\d{2}|20\d{2})\b', period)

            # Check for "Current" or "Present"
            is_current = any(word in period.lower() for word in ["present", "current", "now", "today"])

            if found_years:
                # Convert string years to integers
                int_years = [int(y) for y in found_years]
                years.extend(int_years)

            if is_current:
                years.append(current_year)

        if not years:
            return None

        # Calculate span
        earliest = min(years)
        latest = max(years)

        total_years = latest - earliest

        # Return 0 if the math results in a negative (data error)
        # or at least 1 if they started/ended in the same year
        return max(0, total_years)

    def _extract_professional_title(self, text, all_entities):
        """Extract the main professional title from the beginning of the CV using NLP"""
        # Look at the first 600 characters where titles are usually located
        header_section = text[:600]

        # Method 1: Use NLP to find POSITION entities near the very start
        position_candidates = []
        for ent in all_entities:
            if ent['start'] < 300 and ent['entity_group'] in ['POSITION', 'JOB_TITLE', 'TITLE']:
                title = ent['word'].replace("##", "").strip()
                # Prioritize titles found very early in the document
                score = 300 - ent['start']  # Earlier = higher score
                position_candidates.append((score, title, ent['start']))

        # Sort by score (position in document)
        position_candidates.sort(reverse=True, key=lambda x: x[0])

        # Return the first valid candidate
        for score, title, start in position_candidates:
            if len(title) > 3 and len(title) < 80:
                # Avoid common false positives
                title_lower = title.lower()
                if 'summary' not in title_lower and 'experience' not in title_lower and 'education' not in title_lower:
                    return title

        # Method 2: Look for all-caps title at the beginning
        lines = header_section.split('\n') if '\n' in header_section else [header_section[:150]]

        for line in lines[:5]:
            line = line.strip()
            if not line or len(line) < 3:
                continue

            # Check for all-caps titles (common format)
            if line.isupper() and 5 < len(line) < 60:
                line_lower = line.lower()
                # Avoid section headers
                if 'summary' not in line_lower and 'experience' not in line_lower and 'skills' not in line_lower:
                    return line.title()

        # Method 3: Extract from before "Summary" or "Experience" section
        summary_pattern = r'^(.+?)(?:Summary|Experience|Skills|Accomplishments)'
        match = re.search(summary_pattern, header_section, re.IGNORECASE | re.DOTALL)

        if match:
            potential_title = match.group(1).strip()
            potential_title = re.sub(r'\s+', ' ', potential_title)

            if 5 < len(potential_title) < 80:
                # Clean up
                potential_title = potential_title.replace('\n', ' ').strip()
                return potential_title

        # Method 4: Use any POSITION entity found in header
        for score, title, start in position_candidates:
            if len(title) > 3:
                return title

        return "Professional"  # Default fallback

    def slice_sections(self, text):
        """Extract major sections from CV using Regex patterns for space-delimited CVs"""

        # These patterns look for section headers followed by multiple spaces
        # The key is matching the header itself, not requiring spaces before it
        patterns = {
            "Experience": r'\s\s+(?:(?:Military\s+)?(?:Work\s+)?(?:Relevant\s+)?(?:Professional\s+)?(?:Production\s+)?(?:Visual\s+Merchandise\s+)?(?:Teaching\s+)?Experience|(?:Work\s+)?History|Career\s+Accomplishments|PROFESSIONAL\s+EXPERIENCE)\s\s+',
            "Education": r'\s\s(?:(?:General\s+)?Education(?:\s+and\s+Training)?(?:\s+and\s+Credentials)?(?:\s+and\s+Professional\s+Training)?(?:\s+and\s+Certifications)?|Educational\s+Background|EDUCATION|Academic\s+Background|Certifications)\s\s+',
            "Skills": r'\s\s+(?:Summary\s+of\s+)?(?:Technical\s+)?(?:Key\s+)?(?:Additional\s+)?(?:Computer\s+)?Skills(?:\s+and\s+(?:Qualifications|Expertise|Competencies|Accomplishments))?\s\s+'
        }

        matches = []

        for label, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                matches.append({
                    "label": label,
                    "start": match.start(),
                    "end": match.end()
                })

        matches.sort(key=lambda x: x["start"])

        # Slice the text
        sliced = {}
        for i in range(len(matches)):
            current_section = matches[i]
            section_name = current_section["label"]

            start_pos = current_section["end"]

            if i + 1 < len(matches):
                end_pos = matches[i+1]["start"]
            else:
                end_pos = len(text)

            section_content = text[start_pos:end_pos].strip()
            sliced[section_name] = section_content

        return sliced

    def _process_experience(self, section_text, full_text, all_entities):
        """Extract work experience entries"""

        # Helper: Merge NLP tokens
        def merge_entities(entities):
            if not entities: return []
            sorted_ents = sorted(entities, key=lambda x: x['start'])
            merged = []
            current = sorted_ents[0].copy()
            current['word'] = current['word'].replace("##", "")

            for i in range(1, len(sorted_ents)):
                next_ent = sorted_ents[i]
                next_word = next_ent['word'].replace("##", "")
                if (next_ent['start'] - current['end'] <= 2) and (next_ent['entity_group'] == current['entity_group']):
                    current['end'] = next_ent['end']
                    current['word'] += " " + next_word if not next_ent['word'].startswith("##") else next_word
                    current['word'] = current['word'].strip()
                else:
                    merged.append(current)
                    current = next_ent.copy()
                    current['word'] = next_word
            merged.append(current)
            return merged

        # Prepare Entities
        section_start_index = full_text.find(section_text)
        relevant_ents = [e.copy() for e in all_entities if e['entity_group'] in ['POSITION', 'JOB_TITLE', 'TITLE', 'COMPANY', 'ORG']]

        section_ents = []
        for ent in relevant_ents:
            if section_start_index <= ent['start'] < section_start_index + len(section_text):
                ent['start'] -= section_start_index
                ent['end'] -= section_start_index
                section_ents.append(ent)

        merged_ents = merge_entities(section_ents)

        # Find and VALIDATE Dates
        raw_date_matches = list(re.finditer(self.date_pattern, section_text))
        date_matches = []

        for match in raw_date_matches:
            txt = match.group(0).strip()

            # Filter A: Year Range
            if re.match(r'^\d{4}$', txt):
                year = int(txt)
                if year < 1970 or year > 2030: continue

            # Look at text surrounding the date
            post_chars = section_text[match.end():match.end()+25]
            pre_chars = section_text[max(0, match.start()-10):match.start()]

            # Filter B: Lowercase Check
            if re.match(r'^\s+[a-z]', post_chars):
                if not re.match(r'^\s+(to|present|current|now|until)', post_chars, re.IGNORECASE):
                    continue

            # Filter C: Preposition Check
            if re.search(r'\b(in|during|around|since)\s+$', pre_chars, re.IGNORECASE):
                if not re.search(r'(\n|\r)\s*(in|during|around|since)\s+$', pre_chars, re.IGNORECASE):
                    continue

            # Filter D: Narrative Verb Check
            narrative_verbs = r'^\s+(participated|delivered|presented|completed|awarded|selected|graduated|received|managed|led|worked|assisted|created|developed|organized)'
            if re.match(narrative_verbs, post_chars, re.IGNORECASE):
                continue

            date_matches.append(match)

        if not date_matches: return []

        # PASS 1: Identify Title/Company
        job_structures = []

        for i, match in enumerate(date_matches):
            d_start, d_end = match.start(), match.end()
            prev_job_end = date_matches[i-1].end() if i > 0 else 0
            next_job_start = date_matches[i+1].start() if i+1 < len(date_matches) else len(section_text)

            found_title_obj = None
            found_company_obj = None
            candidates = []

            window_start = max(prev_job_end, d_start - 150)
            window_end = min(next_job_start, d_end + 150)

            # Strategy A: NLP Candidates
            for ent in merged_ents:
                if ent['start'] >= window_start and ent['end'] <= window_end:
                    if ent['end'] <= d_start:   side = 'before'; dist = d_start - ent['end']
                    elif ent['start'] >= d_end: side = 'after';  dist = ent['start'] - d_end
                    else:                       side = 'overlap'; dist = 0
                    candidates.append({'ent': ent, 'dist': dist, 'side': side, 'type': 'nlp'})

            # Strategy B: Regex Candidates
            pre_text = section_text[window_start:d_start]
            chunks = re.split(r'(\n|\r|\s{2,}|\s[|•]\s)', pre_text)
            valid_chunks = [c for c in chunks if c and c.strip()]

            if valid_chunks:
                pre_candidate = valid_chunks[-1]
                pre_clean = re.sub(r'\s*[,|\-–]\s*$', '', pre_candidate).strip()
                pre_clean = re.sub(r'^(Experience|Work History|Professional Experience|History)\s+', '', pre_clean, flags=re.IGNORECASE).strip()

                is_valid = True
                if pre_clean.lower() in ["experience", "work history", "summary", "skills"]: is_valid = False
                if len(pre_clean) < 3 or len(pre_clean) > 80: is_valid = False

                if is_valid:
                    idx = pre_text.rfind(pre_candidate)
                    if idx != -1:
                        inner_idx = pre_candidate.find(pre_clean)
                        if inner_idx != -1:
                            final_start = window_start + idx + inner_idx
                            dist = d_start - (final_start + len(pre_clean))

                            if dist < 15:
                                candidates.append({
                                    'ent': {'word': pre_clean, 'start': final_start, 'end': final_start + len(pre_clean), 'entity_group': 'TITLE'},
                                    'dist': dist, 'side': 'before', 'type': 'regex'
                                })

                                if len(valid_chunks) > 1:
                                    title_chunk_start = idx
                                    text_before_title = pre_text[:title_chunk_start]

                                    if '\n' not in text_before_title.strip():
                                        comp_chunk = valid_chunks[0]
                                        comp_clean = re.sub(r'\s*[,|\-–]\s*$', '', comp_chunk).strip()

                                        if len(comp_clean) > 2 and comp_clean.lower() not in ["experience", "work history"]:
                                            c_idx = pre_text.find(comp_chunk)
                                            if c_idx != -1:
                                                candidates.append({
                                                    'ent': {
                                                        'word': comp_clean,
                                                        'start': window_start + c_idx,
                                                        'end': window_start + c_idx + len(comp_clean),
                                                        'entity_group': 'COMPANY'
                                                    },
                                                    'dist': 0, 'side': 'before', 'type': 'regex_structure'
                                                })

            post_text = section_text[d_end:window_end]
            post_match = re.match(r'^\s*(.*?)(?=\n|\s{2,}|－|—|\||Company)', post_text, re.IGNORECASE)

            if post_match:
                post_clean = post_match.group(1).strip()
                post_clean = re.sub(r'^[,|\-–]\s*', '', post_clean).strip()
                if 3 < len(post_clean) < 80:
                    dist = post_match.start(1)
                    if dist < 12:
                        candidates.append({
                            'ent': {'word': post_clean, 'start': d_end + post_match.start(1), 'end': d_end + post_match.end(1), 'entity_group': 'TITLE'},
                            'dist': dist, 'side': 'after', 'type': 'regex'
                        })

            candidates.sort(key=lambda x: (0 if x.get('type') == 'regex_structure' else 1, x['dist'], 0 if x['type'] == 'regex' else 1))

            for cand in candidates:
                if cand['ent']['entity_group'] in ['POSITION', 'JOB_TITLE', 'TITLE']:
                    if not found_title_obj: found_title_obj = cand['ent']

            for cand in candidates:
                if cand['ent']['entity_group'] in ['COMPANY', 'ORG']:
                    if not found_company_obj: found_company_obj = cand['ent']

            if not found_title_obj and found_company_obj and found_company_obj['start'] > d_end:
                sandwich = section_text[d_end:found_company_obj['start']].strip()
                sandwich = re.sub(r'^[,|\-–]\s*', '', sandwich)
                sandwich = re.sub(r'\s*[,|\-–]\s*$', '', sandwich).strip()
                if 3 < len(sandwich) < 80:
                    s_start = section_text.find(sandwich, d_end)
                    if s_start != -1:
                        found_title_obj = { 'word': sandwich, 'start': s_start, 'end': s_start + len(sandwich) }

            job_structures.append({
                'date_match': match,
                'title_obj': found_title_obj,
                'company_obj': found_company_obj
            })

        # PASS 2: Extract Content
        jobs = []
        for i, job in enumerate(job_structures):
            current_date_match = job['date_match']
            title_obj = job['title_obj']
            company_obj = job['company_obj']

            title_text = title_obj['word'] if title_obj else "Position"
            company_text = company_obj['word'] if company_obj else "Company"

            desc_start_candidates = [current_date_match.end()]
            if title_obj and title_obj['start'] > current_date_match.end():
                desc_start_candidates.append(title_obj['end'])
            if company_obj and company_obj['start'] > current_date_match.end():
                if company_obj['start'] - current_date_match.end() < 50:
                    desc_start_candidates.append(company_obj['end'])

            desc_start = max(desc_start_candidates)

            desc_end = len(section_text)
            if i + 1 < len(job_structures):
                next_job = job_structures[i+1]
                next_date_start = next_job['date_match'].start()

                if next_job['title_obj']:
                    t_start = next_job['title_obj']['start']
                    if t_start > desc_start and t_start < next_date_start:
                        desc_end = t_start
                    else:
                        desc_end = next_date_start
                else:
                    desc_end = next_date_start

            raw_desc = section_text[desc_start:desc_end].strip()

            raw_desc = re.sub(r'^[\s,.\-–|]+', '', raw_desc).strip()
            if company_text != "Company" and raw_desc.startswith(company_text):
                raw_desc = raw_desc[len(company_text):].strip()
            raw_desc = re.sub(r'^[A-Za-z\s]+\s*,\s*[A-Z]{2}[A-Za-z\s]*', '', raw_desc).strip()
            raw_desc = re.sub(r'^[\s,.\-–|]+', '', raw_desc).strip()

            loc_search_start = section_start_index + desc_start
            location = self._extract_location(raw_desc, loc_search_start, all_entities)
            cleaned_desc = self._clean_description(raw_desc, title_text, location)

            jobs.append({
                "title": title_text,
                "company": "",
                "period": current_date_match.group(0),
                "location": "",
                "description": ""
            })

        return jobs

    def _process_education(self, section_text, full_text, all_entities):
        """Extract and clean education entries with overlap prevention"""
        date_matches = list(re.finditer(self.date_pattern, section_text))
        section_start = full_text.find(section_text)

        education_entries = []
        used_indices = set()

        for i, match in enumerate(date_matches):
            d_start, d_end = match.start(), match.end()
            date_text = match.group(0)

            prev_date_end = date_matches[i-1].end() if i > 0 else 0
            next_date_start = date_matches[i+1].start() if i+1 < len(date_matches) else len(section_text)

            chunk_start = max(prev_date_end, d_start - 150)
            chunk_end = next_date_start

            content_chunk = section_text[chunk_start:chunk_end].strip()
            content_chunk_lower = content_chunk.lower()

            has_degree = self.degree_pattern.search(content_chunk)
            edu_kw_count = sum(1 for kw in ["university", "college", "school", "major", "gpa", "studies", "degree", "diploma"] if kw in content_chunk_lower)

            is_evaluation = any(kw in content_chunk_lower for kw in ["performance", "evaluation", "merit", "proposal", "hiring"])

            if not (has_degree or edu_kw_count >= 1) or is_evaluation:
                continue

            found_institution = None
            found_degree = has_degree.group(0) if has_degree else None

            full_chunk_start = section_start + chunk_start
            full_chunk_end = section_start + chunk_end

            for ent in all_entities:
                if full_chunk_start <= ent['start'] <= full_chunk_end:
                    if ent['entity_group'] in ['ORG', 'SCHOOL', 'UNIVERSITY'] and not found_institution:
                        found_institution = ent['word'].replace("##", "").strip()
                    if not found_degree and ent['entity_group'] in ['DEGREE', 'QUALIFICATION']:
                        found_degree = ent['word'].replace("##", "").strip()

            details = content_chunk
            if found_institution:
                details = details.replace(found_institution, "")
            if found_degree:
                details = re.sub(re.escape(found_degree), "", details, flags=re.IGNORECASE)

            details = details.replace(date_text, "").strip()
            details = re.sub(r'[:\-\s,·|]+', ' ', details).strip()

            if found_institution or found_degree:
                education_entries.append({
                    "period": date_text,
                    "institution": found_institution or "",
                    "degree": found_degree or "",
                    "details": ""
                })

        unique_entries = []
        seen = set()
        for entry in education_entries:
            identifier = f"{entry['institution']}-{entry['degree']}".lower()
            if identifier not in seen:
                unique_entries.append(entry)
                seen.add(identifier)

        return unique_entries

    def _process_skills(self, section):
        skill_list = []

        if "," in section:
            section = section.replace(",", ";")
        if "  " in section:
            section = section.replace("  ", ";")
        section = section.split(";")
        for el in section:
            el = el.rstrip()
            el = el.lstrip()
            if len(el) < 25:
                skill_list.append(el.lower())
        return skill_list

    def _extract_location(self, text, text_position, all_entities):
        """Extract location information using NLP + regex"""
        location = {"city": "Unknown", "state": "Unknown", "raw": "Unknown"}

        location_entities = []
        window_start = text_position
        window_end = text_position + len(text)

        for ent in all_entities:
            if window_start <= ent['start'] <= window_end:
                if ent['entity_group'] in ['LOCATION', 'GPE', 'CITY', 'STATE', 'PLACE']:
                    location_entities.append(ent['word'].replace("##", "").strip())

        if location_entities and len(location_entities) >= 2:
            location['city'] = location_entities[0]
            location['state'] = location_entities[1]
            location['raw'] = f"{location['city']}, {location['state']}"
            return location
        elif location_entities and len(location_entities) == 1:
            location['raw'] = location_entities[0]
            return location

        match = re.search(r'([A-Z][a-zA-Z\s]+?)\s+,\s+([A-Z][a-zA-Z\s]+?)(?:\s+|,|\.|\d)', text)

        if match:
            city = match.group(1).strip()
            state = match.group(2).strip()

            if city not in ['Company Name', 'United'] and state not in ['United States']:
                location['city'] = city
                location['state'] = state
                location['raw'] = f"{city}, {state}"
                return location

        match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*,\s*([A-Z]{2})\b', text)
        if match:
            location['city'] = match.group(1).strip()
            location['state'] = match.group(2).strip()
            location['raw'] = f"{location['city']}, {location['state']}"
            return location

        match = re.search(r"Company Name\s*[－\-]\s*([^,]+)\s*,\s*([^.\d]+?)(?:\s|$)", text)
        if match:
            city = match.group(1).strip()
            state = match.group(2).strip()

            if city not in ['City'] and state not in ['State']:
                location['city'] = city
                location['state'] = state
                location['raw'] = f"{city}, {state}"

        return location

    def _clean_description(self, text, title, location):
        """Clean description text by removing redundant information"""
        cleaned = text

        if title and title != "Position":
            cleaned = cleaned.replace(title, "")

        if location['raw'] != "Unknown":
            patterns_to_remove = [
                f"Company Name － {location['city']} , {location['state']}",
                f"Company Name - {location['city']} , {location['state']}",
                f"Company Name {location['city']} , {location['state']}",
                f"{location['city']} , {location['state']}",
            ]
            for pattern in patterns_to_remove:
                cleaned = cleaned.replace(pattern, "")

        cleaned = cleaned.replace("Company Name － City , State", "")
        cleaned = cleaned.replace("Company Name - City , State", "")
        cleaned = cleaned.replace("Company Name City , State", "")
        cleaned = cleaned.replace("Company Name", "")

        cleaned = cleaned.strip()
        cleaned = re.sub(r'^\W+', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)

        if len(cleaned) > 250:
            cleaned = cleaned[:250] + "..."

        return cleaned