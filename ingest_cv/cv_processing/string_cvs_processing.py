import pandas as pd
import torch
from transformers import pipeline
from gliner import GLiNER
import re
from pathlib import Path

class CVParserNLP:
    def __init__(self, batch_size=8):
        print("Initializing NLP Engine...")

        self.batch_size = batch_size
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.xpu.is_available():
            self.device = "xpu"
        print(f"Using device: {self.device}")
        current_script_dir = Path(__file__).resolve().parent.parent.parent
        model_path_1 = current_script_dir / "ingest_cv" / "models" / "cv_parser_cvs_model_1"
        #Verify model existance
        if not model_path_1.exists():
            raise FileNotFoundError(f"Model not found in: {model_path_1}")
        model_local_path_1 = str(model_path_1.absolute()) 
        if self.device == "cuda" or self.device == "xpu":
            self.classifier = pipeline(
                "zero-shot-classification",
                model=model_local_path_1,
                device=0,
                torch_dtype=torch.float16
            )
        else:
            self.classifier = pipeline(
                "zero-shot-classification",
                model=model_local_path_1
            )
        
        model_path_2 = current_script_dir / "ingest_cv" / "models" / "cv_parser_cvs_model_2"
        #Verify model existance
        if not model_path_2.exists():
            raise FileNotFoundError(f"Model not found in: {model_path_2}")
        model_local_path_2 = str(model_path_2.absolute()) 
        self.extractor = GLiNER.from_pretrained(model_local_path_2)
        self.extractor.to(self.device)
        if self.device == "cuda" or self.device == "xpu":
            self.extractor.half()

        self.contact_labels = ["person", "email", "phone number", "profile"]
        self.experience_labels = ["job title", "position", "company name", "organization", "employment date", "start date", "end date", "department", "location", "responsibility", "achievement"]
        self.education_labels = ["university", "college", "school name", "degree", "major", "field of study", "graduation date", "GPA", "academic achievement", "thesis title"]
        self.skills_labels = ["programming language", "framework", "tool", "software", "technical skill", "soft skill", "certification", "proficiency level"]

        print("Models loaded successfully")

        self.degree_keywords = [
            "Master of Arts", "Bachelor of Arts", "Master", "Bachelor",
            "B.A.", "BA", "M.A.", "High School Diploma", "PhD", "Associate",
            "of Science", "Diploma", "Doctorate", "M.S.", "B.S.", "MBA", "Studies",
            "Certification", "Certificate", "Licensure", "Credential", "Certified"
        ]
        pattern_string = r"\b(?:" + "|".join(re.escape(k) for k in self.degree_keywords) + r")\b"
        self.degree_pattern = re.compile(pattern_string, re.IGNORECASE)

        months = ["January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December"]
        month_names_pattern = r"(?:" + "|".join([f"{m[:3]}(?:{m[3:]})?" for m in months]) + r")"

        base_date = (
            r"(?:"
            f"{month_names_pattern}\\s+[0-2]\\d{{3}}|"
            r"[0-2]\d{3}\s*[-]\s*[0-3]\d\s*[-]\s*[0-3]\d|"
            r"[0-2]\d\s*[-/ ]\s*[0-2]\d{3}|"
            r"[0-2]\d{3}"
            r")"
        )
        end_point = rf"(?:{base_date}|Present|Current|Now|Today)"
        self.date_pattern = re.compile(
            rf"\b{base_date}(?:\s*(?:to|[-–—])\s*{end_point})?\b",
            re.IGNORECASE
        )

        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.linkedin_pattern = re.compile(r'(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9_-]+/?', re.IGNORECASE)

    def _clean_field(self, text):
        if not text:
            return ""
        text = text.replace('|', ' ')
        text = re.sub(r'\s+-\s+', ' ', text)
        text = text.strip(" .,;|-•*:\n\t")
        return re.sub(r'\s+', ' ', text).strip()

    def _chunk_text(self, text, max_length=200, overlap=50):
        words = text.split()
        chunks = []
        positions = []
        start = 0
        while start < len(words):
            end = min(start + max_length, len(words))
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            char_start = len(' '.join(words[:start]))
            if start > 0:
                char_start += 1
            positions.append(char_start)
            start += max_length - overlap
        return chunks, positions

    def _extract_entities_for_section(self, text, section_type):
        label_map = {
            "contact": self.contact_labels,
            "experience": self.experience_labels,
            "education": self.education_labels,
            "skills": self.skills_labels
        }
        labels = label_map.get(section_type, self.contact_labels)
        chunks, positions = self._chunk_text(text)
        all_entities = []
        seen_entities = set()

        for chunk, char_offset in zip(chunks, positions):
            chunk_entities = self.extractor.predict_entities(chunk, labels)
            for ent in chunk_entities:
                ent['start'] += char_offset
                ent['end'] += char_offset
                ent['entity_group'] = ent['label'].upper().replace(" ", "_")
                ent['word'] = ent['text']
                entity_id = f"{ent['word']}_{ent['start']}_{ent['end']}"
                if entity_id not in seen_entities:
                    seen_entities.add(entity_id)
                    all_entities.append(ent)
        return all_entities

    def parse(self, text):
        if not isinstance(text, str):
            text = str(text)
        if not text or text.strip() == "" or text == "nan":
            return self._empty_result()
        return self._parse_single(text)

    def _parse_single(self, text, all_entities=None):
        sections = self._process_cv(text)
        print(f"DEBUG: Sections found: {list(sections.keys())}")

        header_text = sections.get("Header", text[:500])
        contact_info = self._extract_contact_info(header_text)

        header_entities = self._extract_entities_for_section(header_text, "contact")
        professional_title = self._extract_professional_title(header_text, header_entities)

        experience_section = sections.get("Experience", "")
        jobs = self._process_experience(experience_section)

        if professional_title == "Professional" and jobs and jobs[0].get("title"):
            professional_title = jobs[0]["title"]
        professional_title = self._clean_field(professional_title)

        education_section = sections.get("Education", "")
        certifications_section = sections.get("Certifications", "")

        real_education = self._process_education_section(education_section)
        certifications = self._process_certifications_section(certifications_section)

        combined_education = real_education + certifications

        skills_section = sections.get("Skills", "")
        languages_section = sections.get("Languages", "")
        combined_skills_text = skills_section
        if languages_section:
            combined_skills_text += "\n" + languages_section

        skills = self._process_skills(combined_skills_text)

        if jobs and (jobs[0]["title"] == "Position" or not jobs[0]["title"]):
            jobs[0]["title"] = professional_title.capitalize()

        return {
            "personal information": {
                "name": contact_info["Name"],
                "email": contact_info["Email"],
                "linkedin": contact_info["Linkedin"]
            },
            "title": professional_title,
            "experience": jobs,
            "education": combined_education,
            "skills": skills,
            "total_experience": self._process_years_exp(jobs),
            "location": contact_info["Location"] or "Any"
        }

    def _process_certifications_section(self, section_text):
        if not section_text:
            return []

        date_matches = list(re.finditer(self.date_pattern, section_text))
        certs = []

        stopwords = [
            "I", "We", "He", "She", "They", "It", "My",
            "Additionally", "Also", "Furthermore", "Moreover", "However",
            "In", "On", "At", "During", "Since", "While", "Following"
        ]

        if not date_matches:
            lines = [l.strip() for l in section_text.split('\n') if l.strip()]
            for line in lines:
                if line and line[0].isupper():
                    first_word = line.split()[0].rstrip(".,;")
                    if first_word not in stopwords and len(line) > 5:
                        certs.append({
                            "period": "Unknown", "institution": "", "degree": self._clean_field(line), "details": ""
                        })
            return certs

        for i, match in enumerate(date_matches):
            d_start = match.start()
            date_text = match.group(0)
            prev_date_end = date_matches[i-1].end() if i > 0 else 0

            before_date = section_text[max(prev_date_end, d_start-200):d_start].strip()
            lines = [l.strip() for l in before_date.split('\n') if l.strip()]

            if not lines: continue
            target_text = lines[-1]

            cert_name = ""

            narrative_match = re.search(r'(?:hold|held|earned|obtained|received|completed|awarded)\s+(?:a\s+|an\s+|the\s+)?(.*?)(\s+(?:credential|certification|certificate|diploma|licensure)|$|in\s+)', target_text, re.IGNORECASE)

            if narrative_match:
                potential_name = narrative_match.group(1).strip()
                if 3 < len(potential_name) < 80:
                     cert_name = potential_name

            if not cert_name:
                start_cap = re.match(r'^([A-Z0-9][a-zA-Z0-9\s\(\)\-]+)', target_text)
                if start_cap:
                    potential = start_cap.group(1).strip()
                    first_word = potential.split()[0]
                    if first_word not in stopwords:
                         potential = re.sub(r'\s+(?:and|from|with|at|by)$', '', potential, flags=re.IGNORECASE)
                         if len(potential) > 4:
                             cert_name = potential

                if not cert_name:
                    cap_sequences = re.findall(r'([A-Z][a-zA-Z0-9\s\(\)\-]+)', target_text)
                    longest_seq = ""
                    for seq in cap_sequences:
                        seq = seq.strip()
                        if seq.split()[0] not in stopwords and len(seq) > 4:
                            if len(seq) > len(longest_seq):
                                longest_seq = seq

                    if longest_seq:
                         cert_name = longest_seq

            if cert_name:
                certs.append({
                    "period": date_text,
                    "institution": "",
                    "degree": self._clean_field(cert_name),
                    "details": ""
                })

        return certs

    def _process_education_section(self, section_text):
        if not section_text:
            return []

        edu_entities = self._extract_entities_for_section(section_text, "education")
        date_matches = list(re.finditer(self.date_pattern, section_text))
        education_entries = []

        institution_keywords = ['university', 'college', 'institute', 'school', 'academy', 'università']

        for i, match in enumerate(date_matches):
            d_start = match.start()
            date_text = match.group(0)
            prev_date_end = date_matches[i-1].end() if i > 0 else 0

            before_date = section_text[max(prev_date_end, d_start-300):d_start].strip()

            found_institution = None
            found_degree = None

            lines_before = [l.strip() for l in before_date.split('\n') if l.strip()]

            for line in reversed(lines_before[-4:]):
                line_lower = line.lower()
                if line_lower in ['graduated', 'completed', 'diploma obtained']:
                    continue

                if not found_degree:
                    has_degree = self.degree_pattern.search(line)
                    if has_degree:
                        found_degree = line
                        if any(kw in line_lower for kw in institution_keywords):
                            if "|" in line:
                                parts = line.split("|")
                                found_degree = parts[0].strip()
                                found_institution = parts[1].strip()
                            elif " at " in line_lower:
                                parts = re.split(r'\sat\s', line, flags=re.IGNORECASE)
                                found_degree = parts[0].strip()
                                found_institution = parts[-1].strip()
                        continue

                if not found_institution:
                    if any(kw in line_lower for kw in institution_keywords):
                        found_institution = line
                        continue

            if not found_degree or not found_institution:
                chunk_start = max(prev_date_end, d_start - 300)
                chunk_end = min(d_start + 50, len(section_text))
                for ent in edu_entities:
                    if chunk_start <= ent['start'] <= chunk_end:
                        if not found_degree and ent['entity_group'] in ['DEGREE', 'QUALIFICATION']:
                            found_degree = ent['word']
                        if not found_institution and ent['entity_group'] in ['UNIVERSITY', 'COLLEGE', 'SCHOOL_NAME']:
                            found_institution = ent['word']

            if found_institution and found_degree:
                if found_institution.lower() in found_degree.lower():
                    found_institution = ""
                elif found_degree.lower() in found_institution.lower():
                    found_degree = found_institution
                    found_institution = ""

            if not found_degree and lines_before:
                clean_lines = [l for l in reversed(lines_before[-3:]) if l.lower() not in ['graduated', 'completed']]
                if clean_lines:
                    found_degree = clean_lines[0]

            if found_degree:
                education_entries.append({
                    "period": date_text,
                    "institution": self._clean_field(found_institution or ""),
                    "degree": self._clean_field(found_degree),
                    "details": ""
                })

        return education_entries

    def _extract_contact_info(self, header_text):
        contact_entities = self._extract_entities_for_section(header_text, "contact")
        email_match = self.email_pattern.search(header_text)
        email = email_match.group(0) if email_match else ""
        for ent in contact_entities:
            if ent['entity_group'] == 'EMAIL':
                email = ent['word']
                break

        linkedin_match = self.linkedin_pattern.search(header_text)
        linkedin = linkedin_match.group(0) if linkedin_match else ""
        for ent in contact_entities:
            if ent['entity_group'] == 'PROFILE':
                linkedin = ent['word']
                break

        location = ""
        lines = [line.strip() for line in header_text.split('\n') if line.strip()]
        bad_indicators = ['personal information', 'curriculum vitae', 'resume', 'contact', 'email', 'phone', 'www.', 'http', '@', 'name:']
        loc_line_pattern = re.compile(r'^([A-Z][a-zA-Z\.\s-]+,\s*[A-Z][a-zA-Z\.\s-]+)$')

        for line in lines:
            line_lower = line.lower()
            if any(ind in line_lower for ind in bad_indicators) or any(c.isdigit() for c in line):
                continue
            if loc_line_pattern.match(line):
                if not any(x in line_lower for x in ['university', 'college', 'limited', 'gmbh', 'inc.', 'ltd']):
                    location = line
                    break

        if not location:
            location_entities = self.extractor.predict_entities(header_text[:500], ["city", "location", "address", "place"])
            for ent in location_entities:
                if ent['label'].lower() in ['city', 'location', 'place']:
                    potential_loc = ent['text'].strip()
                    if ',' in potential_loc or len(potential_loc.split()) <= 4:
                        location = potential_loc
                        break

        name = ""
        person_candidates = [e for e in contact_entities if e['entity_group'] == 'PERSON']
        excluded_terms = ['personal information', 'contact', 'curriculum vitae', 'resume', 'cv', 'profile', 'summary', 'objective']
        person_candidates = [p for p in person_candidates if p['word'].lower() not in excluded_terms]
        person_candidates.sort(key=lambda x: (x['start'], -len(x['word'].split())))

        if person_candidates:
            name = person_candidates[0]['word']
        else:
            name_pattern = re.compile(r'(?:Full\s+)?Name\s*:\s*([A-Z][a-zA-Z\s\.]+?)(?:\n|$)', re.IGNORECASE)
            name_match = name_pattern.search(header_text)
            if name_match:
                name = name_match.group(1).strip()
            else:
                for line in lines[:5]:
                    line_lower = line.lower()
                    if (len(line) < 50 and
                        all(term not in line_lower for term in excluded_terms) and
                        not line.endswith(':') and not '@' in line and not any(char.isdigit() for char in line)):
                        name_match = re.match(r'^([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?(?:\s+[A-Z][a-z]+){0,2})$', line)
                        if name_match:
                            name = name_match.group(1).strip()
                            break

        return {"Name": self._clean_field(name), "Email": email, "Linkedin": linkedin, "Location": self._clean_field(location)}

    def _empty_result(self):
        return {
            "personal information": {"name": "Unknown", "email": "Unknown", "linkedin": "Unknown"},
            "title": "Unknown", "experience": [], "education": [], "skills": [], "total_experience": 0, "location": ""
        }

    def _process_years_exp(self, experience_entries):
        if not experience_entries:
            return 0
        years = []
        current_year = 2026
        for entry in experience_entries:
            period = entry.get("period", "")
            found_years = re.findall(r'\b(19\d{2}|20\d{2})\b', period)
            is_current = any(word in period.lower() for word in ["present", "current", "now", "today"])
            if found_years:
                years.extend([int(y) for y in found_years])
            if is_current:
                years.append(current_year)
        if not years:
            return 0
        return max(0, max(years) - min(years))

    def _extract_professional_title(self, text, all_entities):
        header_section = text[:600]
        excluded_terms = ['personal information', 'contact', 'curriculum vitae', 'resume', 'cv', 'profile', 'summary', 'experience', 'education', 'skills', 'objective', 'professional summary']
        position_candidates = []
        for ent in all_entities:
            if ent['start'] < 300 and ent['entity_group'] in ['POSITION', 'JOB_TITLE', 'TITLE']:
                title = ent['word'].strip()
                if any(term in title.lower() for term in excluded_terms):
                    continue
                score = 300 - ent['start']
                position_candidates.append((score, title, ent['start']))

        position_candidates.sort(reverse=True, key=lambda x: x[0])
        for score, title, start in position_candidates:
            if 3 < len(title) < 80:
                return title

        lines = header_section.split('\n')
        for i, line in enumerate(lines[:8]):
            line = line.strip()
            if not line or any(term in line.lower() for term in excluded_terms):
                continue
            if 10 < len(line) < 80:
                title_indicators = ['manager', 'director', 'engineer', 'developer', 'analyst', 'specialist', 'coordinator', 'lead', 'senior', 'junior', 'consultant', 'architect']
                if any(indicator in line.lower() for indicator in title_indicators):
                    return line
        return "Professional"

    def _process_cv(self, raw_text):
        raw_text = re.sub(r'\bcurriculum vitae\b', '', raw_text, flags=re.IGNORECASE)
        raw_text = re.sub(r'\bresume\b', '', raw_text, flags=re.IGNORECASE)
        raw_text = re.sub(r'\bcv\b', "", raw_text, flags=re.IGNORECASE)

        delimiters = r'\n\s*[-=*]*\s*(WORK\s+EXPERIENCE|EXPERIENCE|PROFESSIONAL\s+EXPERIENCE|EDUCATION|ACADEMIC\s+BACKGROUND|SKILLS(?:\s+AND\s+COMPETENCIES)?|TECHNICAL\s+SKILLS|CORE\s+SKILLS|COMPETENCIES|CERTIFICATIONS?|CERTIFICATIONS?\s+AND\s+TRAINING|SUMMARY|PROFESSIONAL\s+SUMMARY|OBJECTIVE|LANGUAGES?|INTERESTS?|REFERENCES?)\s*[-=*]*\s*\n'

        parts = re.split(delimiters, raw_text, flags=re.IGNORECASE)
        structured_cv = {"Header": parts[0].strip()}

        category_map = {
            "professional work experience and employment history": "Experience",
            "academic education and university degrees": "Education",
            "technical skills and competencies": "Skills",
            "personal contact information and details": "Contact",
            "certifications and licenses": "Certifications",
            "language proficiency and communication": "Languages"
        }

        direct_map = {
            "EXPERIENCE": "Experience", "WORK EXPERIENCE": "Experience", "PROFESSIONAL EXPERIENCE": "Experience",
            "EDUCATION": "Education", "ACADEMIC BACKGROUND": "Education",
            "SKILLS": "Skills", "TECHNICAL SKILLS": "Skills", "CORE SKILLS": "Skills", "COMPETENCIES": "Skills",
            "SKILLS AND COMPETENCIES": "Skills",
            "CERTIFICATIONS": "Certifications", "CERTIFICATION": "Certifications", "CERTIFICATIONS AND TRAINING": "Certifications",
            "SUMMARY": "Summary", "PROFESSIONAL SUMMARY": "Summary", "OBJECTIVE": "Summary",
            "LANGUAGES": "Languages", "LANGUAGE": "Languages",
            "INTERESTS": "Interests", "INTEREST": "Interests",
            "REFERENCES": "References", "REFERENCE": "References"
        }

        for i in range(1, len(parts), 2):
            if i >= len(parts):
                break
            section_header = parts[i].strip().upper() if i < len(parts) else ""
            content = parts[i+1].strip() if (i+1) < len(parts) else ""
            if not content:
                continue
            if section_header in direct_map:
                final_key = direct_map[section_header]
                structured_cv[final_key] = content
            else:
                try:
                    prediction = self.classifier(content[:400], list(category_map.keys()), hypothesis_template="This section contains information about {}.")
                    best_label = prediction['labels'][0]
                    if prediction["scores"][0] > 0.3:
                        final_key = category_map[best_label]
                        structured_cv[final_key] = content
                except Exception:
                    continue
        return structured_cv

    def _process_experience(self, section_text):
        if not section_text:
            return []

        exp_entities = self._extract_entities_for_section(section_text, "experience")
        date_matches = list(re.finditer(self.date_pattern, section_text))
        jobs = []

        for i, match in enumerate(date_matches):
            d_start = match.start()
            d_end = match.end()
            prev_date_end = date_matches[i-1].end() if i > 0 else 0
            next_date_start = date_matches[i+1].start() if i+1 < len(date_matches) else len(section_text)

            before_date = section_text[max(prev_date_end, d_start-300):d_start].strip()
            title_text = ""
            company_text = ""

            pipe_pattern = r'([^\n\|]+)\s*\|\s*([^\n]+)'
            pipe_match = re.search(pipe_pattern, before_date)

            if pipe_match:
                potential_title = pipe_match.group(1).strip()
                potential_company = pipe_match.group(2).strip()
                potential_title = re.sub(r'^[\s•\-\*]+', '', potential_title).strip()
                potential_company = re.sub(r'^[\s•\-\*]+', '', potential_company).strip()
                if 2 < len(potential_title) < 100:
                    title_text = potential_title
                if 2 < len(potential_company) < 100:
                    company_text = potential_company

            if not title_text or not company_text:
                lines_before = [l.strip() for l in before_date.split('\n') if l.strip()]
                for line in reversed(lines_before[-3:]):
                    comma_parts = [p.strip() for p in line.split(',')]
                    if len(comma_parts) >= 2:
                        if not title_text and 3 < len(comma_parts[0]) < 100 and not re.match(r'^[0-9\-/]+$', comma_parts[0]):
                            title_text = comma_parts[0]
                        if not company_text and 2 < len(comma_parts[1]) < 100 and not re.match(r'^[A-Z][a-z]+,?\s*[A-Z][a-z]+$', comma_parts[1]):
                            company_text = comma_parts[1]
                        if title_text and company_text:
                            break

            if not title_text or not company_text:
                candidates = sorted([e for e in exp_entities if e['entity_group'] in ['JOB_TITLE', 'POSITION', 'COMPANY_NAME', 'ORGANIZATION']], key=lambda x: abs(x['start'] - d_start))
                for cand in candidates:
                    if cand['entity_group'] in ['JOB_TITLE', 'POSITION'] and not title_text and abs(cand['start'] - d_start) < 250:
                        title_text = cand['word']
                    if cand['entity_group'] in ['COMPANY_NAME', 'ORGANIZATION'] and not company_text and abs(cand['start'] - d_start) < 250:
                        company_text = cand['word']

            description = section_text[d_end:next_date_start].strip()
            first_period = description.find('.')
            if first_period != -1:
                description = description[:first_period + 1].strip()

            if "education" not in description.lower()[:100]:
                jobs.append({
                    "title": self._clean_field(title_text),
                    "company": self._clean_field(company_text),
                    "period": match.group(0),
                    "description": description
                })
        return jobs

    # ### FIX: UPDATED SKILLS PROCESSING FOR SHORT WORDS AND NOISE ###
    def _process_skills(self, section_text):
        if not section_text:
            return []

        skill_entities = self._extract_entities_for_section(section_text, "skills")
        skill_list = []

        # Whitelist for short skills (1-2 chars) that are usually valid
        short_skills = {'c', 'r', 'go', 'c#', 'f#', 'c++', 'vb', 'js', 'aws', 'git', 'npm', 'pip', 'bi', 'db', 'ai', 'ml', 'qa', 'ui', 'ux', 'seo', 'sem', 'os', 'ip', 'ci', 'cd'}

        # 1. Add GLiNER findings first
        for ent in skill_entities:
            if ent['entity_group'] in ['PROGRAMMING_LANGUAGE', 'FRAMEWORK', 'TOOL', 'SOFTWARE', 'TECHNICAL_SKILL', 'SOFT_SKILL', 'LANGUAGE']:
                skill_name = self._clean_field(ent['word']).lower()
                if skill_name:
                     skill_list.append(skill_name)

        # 2. Extract Languages
        language_pattern = r'\b(fluent|native|proficient|conversational|intermediate|basic|advanced)\s+in\s+([A-Z][a-z]+)\b'
        language_matches = re.finditer(language_pattern, section_text, re.IGNORECASE)
        for match in language_matches:
            proficiency = match.group(1).lower()
            language = match.group(2).lower()
            skill_phrase = f"{proficiency} in {language}"
            if skill_phrase not in skill_list:
                skill_list.append(skill_phrase)

        clean_text = section_text

        # 3. Expanded Cleaning of Noise Phrases
        clean_patterns = [
           r'my technical repertoire includes',
           r'my technical competencies include',
           r'my skills include',
           r'i am (?:highly )?skilled in',
           r'proficiency in',
           r'proficient in',
           r'include',
           r'including',
           r'such as',
           r'like',
           r'knowledge covers',
           r'expertise lies in',
           r'experience with',
           r'programming languages?',
           r'frameworks?',
           r'libraries',
           r'tools?',
           r'infrastructure',
           r'regarding',
           r'advanced proficiency in'
        ]
        pattern_combined = r'(?:' + '|'.join(clean_patterns) + r')\s*'
        clean_text = re.sub(pattern_combined, '', clean_text, flags=re.IGNORECASE)

        # Fix dot-names before split
        clean_text = clean_text.replace("Node.js", "NodeJS").replace(".NET", "DotNet").replace("Vue.js", "VueJS")

        # Standardize separators
        clean_text = clean_text.replace(" and ", ",")
        clean_text = re.sub(r'\n\s*[-•*]\s*', ',', clean_text)

        # Split
        parts = re.split(r'[,;.]|\s+as well as\s+', clean_text)

        for part in parts:
            part = self._clean_field(part)
            if not part: continue

            part_lower = part.lower()

            # Logic for short words
            if len(part) < 3:
                if part_lower not in short_skills and not part.isupper():
                    continue

            # Stopword checks
            if any(lang_skill in part_lower for lang_skill in skill_list if 'in' in lang_skill): continue
            word_count = len(part.split())
            if word_count > 8: continue

            sentence_indicators = [' i ', ' my ', ' am ', ' is ', ' are ', ' was ', ' were ', ' have ', ' has ']
            if any(indicator in ' ' + part_lower + ' ' for indicator in sentence_indicators): continue

            # Final Cleanup
            part = re.sub(r'^(?:and|or|also|with)\s+', '', part, flags=re.IGNORECASE)

            if len(part) > 0 and part.lower() not in skill_list:
                skill_list.append(part.lower())

        return skill_list