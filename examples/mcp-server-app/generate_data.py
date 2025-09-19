import json
import random
from datetime import datetime, timedelta
import os
from faker import Faker

# For creating fake names
fake = Faker()

# Data
SKILLS = ["Python", "JavaScript", "React", "Node.js", "SQL", "AWS", "Docker", "Git"]
COMPANIES = [
    "TechCorp", "DataFlow", "CloudTech", "InnovateNow", "DevStudio",
    "CodeCraft Solutions", "ByteForge", "NexGen Systems", "ScalePoint",
    "DevHub Technologies", "CloudPeak", "DataBridge Inc", "TechFlow Labs",
    "BuildRight Software", "StreamLine Tech", "CoreLogic Systems", 
    "PixelPerfect Studios", "RapidScale", "ThinkTech Innovations",
    "BlueChip Software", "GreenField Technologies"
] 
LOCATIONS = ["San Francisco", "New York", "Seattle", "Remote", "Austin"]
JOB_TITLES = ["Software Engineer", "Frontend Developer", "Data Scientist", "DevOps Engineer"]
JOB_DESCRIPTIONS = [
    "Join our {adjective} team to build {project} using {skill}. Great opportunity for growth.",
    "We're looking for a {title} to help us {action} with modern technology stack.",
    "Exciting role working on {project} in a {adjective} environment. {skill} experience required.",
    "Great opportunity for a skilled {title} to {action} using cutting-edge tools.",
    "We need a {adjective} {title} to lead development of our next-generation {project}.",
    "Perfect role for someone passionate about {project} and clean code practices.",
    "Join our engineering team to architect and develop scalable {project} solutions.",
    "Seeking a talented developer to enhance our {project} infrastructure.",
    "Be part of our {adjective} culture while building world-class applications.",
    "Help us revolutionize the industry by creating innovative {project} solutions."
]

CANDIDATE_SUMMARIES = [
    "{experience} years of experience building {project} with modern tech stack.",
    "Passionate {title} specializing in {skill} and best practices.",
    "Creative problem solver with expertise in {skill}. {experience} years in tech.",
    "Full-stack developer with {experience} years creating scalable {project}.",
    "Results-driven {title} focused on {project} and performance optimization.",
    "Senior developer specializing in enterprise {project} solutions.",
    "Detail-oriented engineer with {experience}+ years in software development.",
    "Innovative {title} who enjoys tackling complex technical challenges.",
    "Self-motivated developer with proven track record in {skill} development.",
    "Tech enthusiast with {experience} years transforming ideas into robust applications.",
    "Versatile engineer experienced in full-stack development and {skill}.",
    "Goal-oriented {title} with strong analytical and problem-solving skills.",
    "Collaborative team player with {experience} years delivering quality software.",
    "Forward-thinking developer passionate about clean code and architecture."
]

ADJECTIVES = ["innovative", "fast-paced", "collaborative", "dynamic"]
PROJECTS = ["web applications", "mobile apps", "data pipelines", "cloud infrastructure", "APIs"]
ACTIONS = ["scale our platform", "modernize our systems", "build new features", "optimize performance"]

def generate_job(num: int):
    """Generate one job posting"""
    title = random.choice(JOB_TITLES)
    skills = random.sample(SKILLS, random.randint(3, 5))
    skills = {skill : 1 for skill in skills}
    location = random.choice(LOCATIONS)
    
    # Generate description
    template = random.choice(JOB_DESCRIPTIONS)
    description = template.format(
        title=title.lower(),
        adjective=random.choice(ADJECTIVES),
        project=random.choice(PROJECTS),
        action=random.choice(ACTIONS),
        skill=random.choice(list(skills.keys()))
    )
    
    return {
        "job_id": f"J{num}",
        "title": title,
        "company": random.choice(COMPANIES),
        "location": location,
        "description": description,
        "skills": skills,
        "salary_min": random.randint(80000, 120000),
        "salary_max": random.randint(120000, 180000),
        "remote_ok": True if location == "Remote" else random.choice([True, False]),
        "posted_date": int((datetime.now() - timedelta(days=random.randint(1, 30))).timestamp())
    }

def generate_candidate(num: int):
    """Generate one candidate profile"""
    name = fake.name()
    skills = random.sample(SKILLS, random.randint(2, 6))
    skills = {skill: 1 for skill in skills}
    experience = random.randint(1, 10)
    location = random.choice(LOCATIONS)
    
    # Generate summary
    template = random.choice(CANDIDATE_SUMMARIES)
    summary = template.format(
        experience=experience,
        title=random.choice(JOB_TITLES).lower(),
        project=random.choice(PROJECTS),
        skill=random.choice(list(skills.keys()))
    )
    
    return {
        "candidate_id": f"C{num}",
        "name": name,
        "candidate_summary": summary,
        "skills": skills,
        "experience_years": experience,
        "location": location,
        "desired_salary": random.randint(90000, 150000),
        "open_to_remote": True if location == "Remote" else random.choice([True, False])
    }

def save_vespa_feed(items, doc_type, filename):
    """Save as Vespa JSONL feed format"""
    output_dir = "./dataset"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        output = [
            {
            "put": f"id:{doc_type}:{doc_type}::{item[f'{doc_type}_id']}",
            "fields": item
            } 
            for item in items]
        f.write("\n".join(map(json.dumps, output)))

if __name__ == "__main__":
    # Number of docs to create
    NUM_JOBS = 75
    NUM_CANDIDATES = 30
    # Generate data
    jobs = [generate_job(i) for i in range(1, NUM_JOBS + 1)]
    candidates = [generate_candidate(i) for i in range(1, NUM_CANDIDATES + 1)]
    
    # Save files
    save_vespa_feed(jobs, "job", "jobs.jsonl")
    save_vespa_feed(candidates, "candidate", "candidates.jsonl")
    
    print(f"✅ Generated {len(jobs)} jobs and {len(candidates)} candidates")
    print("📁 Files: jobs.jsonl, candidates.jsonl")