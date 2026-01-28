#!/usr/bin/env python3
"""
Job Data Analyzer
Analyzes job data from JSON files to find the most searched jobs and skills
"""

import json
from collections import Counter
from typing import List, Dict, Any


# ============================================================================
# CONFIGURATION - EDIT THESE VALUES
# ============================================================================

# Input file path - CHANGE THIS to your JSON file path
INPUT_FILE = "cleaned_jobs.json"

# Number of top items to display (e.g., top 10, top 20, etc.)
TOP_N = 10

# Output file path - Set to None if you don't want to save results
# Set to a filename like "report.json" to save the analysis
OUTPUT_FILE = "job_report.json"  # Change to "report.json" to save results

# ============================================================================


def load_job_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load job data from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing job data
        
    Returns:
        List of job dictionaries
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both single dict and list of dicts
        if isinstance(data, dict):
            return [data]
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        print(f"Please check the file path and make sure the file exists.")
        return []
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' is not valid JSON.")
        return []


def analyze_jobs(jobs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze job data to find most common job titles and skills.
    
    Args:
        jobs: List of job dictionaries
        
    Returns:
        Dictionary containing analysis results
    """
    # Counters for different attributes
    job_titles = Counter()
    skills = Counter()
    locations = Counter()
    companies = Counter()
    categories = Counter()
    
    # Process each job
    for job in jobs:
        # Count job titles
        if 'title' in job and job['title']:
            job_titles[job['title']] += 1
        
        # Count skills
        if 'skills' in job and job['skills']:
            for skill in job['skills']:
                if skill:  # Ensure skill is not empty
                    skills[skill] += 1
        
        # Count locations
        if 'location' in job and job['location']:
            locations[job['location']] += 1
        
        # Count companies
        if 'company' in job and job['company']:
            companies[job['company']] += 1
        
        # Count categories
        if 'category' in job and job['category']:
            categories[job['category']] += 1
    
    return {
        'total_jobs': len(jobs),
        'job_titles': job_titles,
        'skills': skills,
        'locations': locations,
        'companies': companies,
        'categories': categories
    }


def display_top_items(counter: Counter, title: str, top_n: int = 10) -> None:
    """
    Display the top N items from a Counter object.
    
    Args:
        counter: Counter object to display
        title: Title for the display
        top_n: Number of top items to show
    """
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    
    if not counter:
        print("No data available")
        return
    
    total = sum(counter.values())
    for rank, (item, count) in enumerate(counter.most_common(top_n), 1):
        percentage = (count / total) * 100
        print(f"{rank:2}. {item:50} | Count: {count:5} | {percentage:5.1f}%")
    
    if len(counter) > top_n:
        print(f"\n... and {len(counter) - top_n} more")


def generate_report(analysis: Dict[str, Any], jobs: List[Dict[str, Any]], top_n: int = 10) -> None:
    """
    Generate a comprehensive report of the analysis.
    
    Args:
        analysis: Analysis results dictionary
        jobs: Original list of jobs for additional calculations
        top_n: Number of top items to show for each category
    """
    print("\n" + "="*70)
    print(" "*20 + "JOB DATA ANALYSIS REPORT")
    print("="*70)
    print(f"\nTotal Jobs Analyzed: {analysis['total_jobs']}")
    
    # Display top job titles
    display_top_items(
        analysis['job_titles'], 
        f"TOP {top_n} MOST SEARCHED JOB TITLES", 
        top_n
    )
    
    # Display top skills
    display_top_items(
        analysis['skills'], 
        f"TOP {top_n} MOST SEARCHED SKILLS", 
        top_n
    )
    
    # Display top locations
    display_top_items(
        analysis['locations'], 
        f"TOP {top_n} LOCATIONS", 
        top_n
    )
    
    # Display top companies
    display_top_items(
        analysis['companies'], 
        f"TOP {top_n} COMPANIES", 
        top_n
    )
    
    # Display categories
    display_top_items(
        analysis['categories'], 
        "CATEGORIES", 
        top_n
    )
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    print(f"Unique Job Titles: {len(analysis['job_titles'])}")
    print(f"Unique Skills: {len(analysis['skills'])}")
    print(f"Unique Locations: {len(analysis['locations'])}")
    print(f"Unique Companies: {len(analysis['companies'])}")
    
    jobs_with_skills = sum(1 for job in jobs if job.get('skills') and len(job.get('skills', [])) > 0)
    print(f"Jobs with Skills Listed: {jobs_with_skills}")
    print(f"Jobs without Skills: {analysis['total_jobs'] - jobs_with_skills}")


def save_report_to_file(analysis: Dict[str, Any], output_file: str, top_n: int = 10) -> None:
    """
    Save analysis report to a JSON file.
    
    Args:
        analysis: Analysis results dictionary
        output_file: Path to output JSON file
        top_n: Number of top items to include
    """
    report = {
        'total_jobs': analysis['total_jobs'],
        'top_job_titles': dict(analysis['job_titles'].most_common(top_n)),
        'top_skills': dict(analysis['skills'].most_common(top_n)),
        'top_locations': dict(analysis['locations'].most_common(top_n)),
        'top_companies': dict(analysis['companies'].most_common(top_n)),
        'categories': dict(analysis['categories']),
        'statistics': {
            'unique_job_titles': len(analysis['job_titles']),
            'unique_skills': len(analysis['skills']),
            'unique_locations': len(analysis['locations']),
            'unique_companies': len(analysis['companies'])
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Report saved to: {output_file}")


def main():
    """Main function to run the job data analyzer."""
    
    print("="*70)
    print(" "*20 + "JOB DATA ANALYZER")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Input File: {INPUT_FILE}")
    print(f"  Top N Items: {TOP_N}")
    print(f"  Output File: {OUTPUT_FILE if OUTPUT_FILE else 'Not saving to file'}")
    print("="*70)
    
    # Load and analyze data
    print(f"\nLoading job data from: {INPUT_FILE}")
    jobs = load_job_data(INPUT_FILE)
    
    if not jobs:
        print("\n❌ No data to analyze. Please check your input file.")
        return
    
    print(f"✓ Successfully loaded {len(jobs)} job(s)")
    print(f"\nAnalyzing data...")
    analysis = analyze_jobs(jobs)
    
    # Generate and display report
    generate_report(analysis, jobs, TOP_N)
    
    # Save to file if requested
    if OUTPUT_FILE:
        save_report_to_file(analysis, OUTPUT_FILE, TOP_N)
    
    print("\n" + "="*70)
    print("✓ Analysis complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()