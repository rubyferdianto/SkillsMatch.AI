"""
SkillsMatch.AI Flask Web Application

A modern web interface for the SkillsMatch.AI career matching system
with real-time features, beautiful UI, and comprehensive functionality.
"""

import os
import sys
import json
import asyncio
import contextlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Check conda environment on startup
def check_conda_environment():
    """Check if we're running in the correct conda environment"""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    is_production = os.environ.get('RENDER') or os.environ.get('RAILWAY') or os.environ.get('HEROKU') or os.environ.get('VERCEL')
    
    if is_production:
        print(f"🚀 Running in production environment: {os.environ.get('RENDER_SERVICE_NAME', 'Cloud Platform')}")
        print(f"🐍 Python environment: {conda_env or 'system'}")
    elif conda_env != 'smai':
        print("⚠️  WARNING: Not running in 'smai' conda environment!")
        print(f"📍 Current environment: {conda_env or 'base'}")
        print("🔧 To fix this, activate the environment first:")
        print("   conda activate smai")
        print("   python app.py")
        print("")
    else:
        print(f"✅ Running in correct conda environment: {conda_env}")

# Check environment on import (only in development)
if not os.environ.get('RENDER'):
    check_conda_environment()

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_cors import CORS
from flask_socketio import SocketIO, emit
# import eventlet  # Commented out due to SSL issue

# AI imports
try:
    import openai
    from openai import OpenAI
except ImportError:
    openai = None
    OpenAI = None

# Load environment variables from .env file
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# Vector search service import
try:
    try:
        from services.simple_vector_service import get_vector_service
    except ImportError:
        from web.services.simple_vector_service import get_vector_service
    VECTOR_SEARCH_AVAILABLE = True
    print("✅ Vector search service available")
except ImportError as e:
    print(f"⚠️ Vector search service not available: {e}")
    VECTOR_SEARCH_AVAILABLE = False

# Add paths for imports - handle both development and production
app_dir = Path(__file__).parent
project_root = app_dir.parent

# Debug path information in production
is_production = os.environ.get('RENDER') or os.environ.get('RAILWAY') or os.environ.get('HEROKU')
if is_production:
    print(f"📍 App directory: {app_dir}")
    print(f"📍 Project root: {project_root}")
    print(f"📍 Current working directory: {os.getcwd()}")
    print(f"📍 Python path before: {sys.path[:3]}")

sys.path.insert(0, str(project_root))  # Add project root
sys.path.insert(0, str(app_dir))       # Add web directory  
sys.path.insert(0, os.getcwd())        # Add current working directory

if is_production:
    print(f"📍 Python path after: {sys.path[:5]}")

# Import storage layer
try:
    from web.storage import profile_manager
except ImportError:
    try:
        from storage import profile_manager
    except ImportError:
        # Production fallback - create minimal profile manager
        class MinimalProfileManager:
            def list_profiles(self): return []
            def load_profile(self, profile_id): return None
        profile_manager = MinimalProfileManager()

# Debug: Check if API keys are loaded
openai_key = os.environ.get('OPENAI_API_KEY')
github_token = os.environ.get('GITHUB_TOKEN')

if openai_key:
    print(f"✅ OpenAI API key loaded from .env (length: {len(openai_key)} characters)")
    print("🚀 Using latest OpenAI models: GPT-4o, GPT-4 Turbo (including ChatGPT Pro models)")
    if github_token:
        print("🔄 GitHub token also available as fallback")
elif github_token:
    print(f"✅ GitHub token loaded from .env (length: {len(github_token)} characters)")
    print("🚀 Using GitHub Copilot Pro models: GPT-5, O1, DeepSeek-R1")
else:
    print("❌ No AI API keys found - will use enhanced basic matching")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Global database imports with comprehensive fallback handling
def import_database_modules():
    """Import database modules with comprehensive fallback handling."""
    import_attempts = []
    
    # Check if database directory exists in various locations
    database_paths = [
        os.path.join(os.getcwd(), 'database'),  # Current working directory
        os.path.join(os.getcwd(), 'web', 'database'),  # Web subdirectory in cwd
        os.path.join(os.path.dirname(__file__), 'database'),  # Local to app.py
        os.path.join(os.path.dirname(__file__), '..', 'web', 'database')  # Parent web directory
    ]
    
    if is_production:
        print(f"🔍 Checking database paths:")
        for i, path in enumerate(database_paths, 1):
            exists = os.path.exists(path)
            print(f"   Path {i}: {path} - {'EXISTS' if exists else 'NOT FOUND'}")
    
    # Attempt 1: Relative import from current directory
    try:
        from database.models import UserProfile, Job, UserSkill
        print("✅ Successfully imported database.models using relative paths")
        return UserProfile, Job, UserSkill
    except ImportError as e1:
        import_attempts.append(f"Relative import: {e1}")
    
    # Attempt 2: Web prefix import
    try:
        from web.database.models import UserProfile, Job, UserSkill
        print("✅ Successfully imported database.models using web.database paths")
        return UserProfile, Job, UserSkill
    except ImportError as e2:
        import_attempts.append(f"Web prefix import: {e2}")
    
    # Attempt 3: Direct path manipulation
    for database_path in database_paths:
        if os.path.exists(database_path):
            try:
                parent_path = os.path.dirname(database_path)
                if parent_path not in sys.path:
                    sys.path.insert(0, parent_path)
                
                import database.models as db_models
                UserProfile = db_models.UserProfile
                Job = db_models.Job
                UserSkill = db_models.UserSkill
                print(f"✅ Successfully imported database.models using path manipulation: {parent_path}")
                return UserProfile, Job, UserSkill
            except ImportError as e:
                import_attempts.append(f"Path manipulation ({parent_path}): {e}")
                continue
    
    # If all fails, create placeholders
    if is_production:
        print(f"❌ All database import attempts failed:")
        for attempt in import_attempts:
            print(f"   - {attempt}")
    
    class UserProfile:
        def __init__(self, **kwargs): pass
    class Job:
        def __init__(self, **kwargs): pass
    class UserSkill:
        def __init__(self, **kwargs): pass
    
    print("⚠️ Using placeholder classes - database functionality limited")
    return UserProfile, Job, UserSkill

# Import database modules globally
UserProfile, Job, UserSkill = import_database_modules()

def parse_datetime(date_str):
    """Parse datetime string from database into datetime object"""
    if not date_str:
        return None
    if isinstance(date_str, datetime):
        return date_str
    try:
        # Handle ISO format: 2025-11-02T15:52:15.200314
        return datetime.fromisoformat(date_str.replace('T', ' ').split('.')[0])
    except (ValueError, AttributeError):
        return None

# Try to import SkillMatch modules with graceful error handling
try:
    # First try importing the models and utilities (less likely to have OpenAI issues)
    from skillmatch.models import UserProfile, SkillItem, ExperienceLevel, UserPreferences, PreferenceType
    from skillmatch.utils import DataLoader, SkillMatcher
    
    # Then try importing the agent (more likely to have OpenAI compatibility issues)
    try:
        from skillmatch import SkillMatchAgent
        print("✅ SkillMatch core modules loaded successfully")
    except Exception as agent_error:
        print(f"⚠️  SkillMatch agent not available (OpenAI compatibility issue): {agent_error}")
        SkillMatchAgent = None
    
    SKILLMATCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SkillMatch core modules not available: {e}")
    SkillMatchAgent = None
    UserProfile = None
    SkillItem = None
    ExperienceLevel = None
    UserPreferences = None
    PreferenceType = None
    DataLoader = None
    SkillMatcher = None
    SKILLMATCH_AVAILABLE = False

# Scraping functionality removed - using direct API access instead
SCRAPER_AVAILABLE = False

# AI-Powered Job Matching Function
def ai_enhanced_job_matching(profile_data, jobs_list, vector_resume_text=None):
    """Use AI to analyze comprehensive user profile and match with jobs"""
    # Debug AI availability
    print(f"🔍 AI Debug - openai module: {openai is not None}")
    print(f"🔍 AI Debug - OpenAI class: {OpenAI is not None}")
    print(f"🔍 AI Debug - API key available: {openai_key is not None}")
    print(f"🔍 AI Debug - API key length: {len(openai_key) if openai_key else 0}")
    
    if not openai or not openai_key or not OpenAI:
        print("⚠️ AI not available - falling back to basic matching")
        return None
    
    try:
        client = OpenAI(api_key=openai_key)
        
        # Build comprehensive profile context
        profile_context = {
            'name': profile_data.get('name', 'Professional'),
            'title': profile_data.get('title', ''),
            'location': profile_data.get('location', ''),
            'experience_level': profile_data.get('experience_level', 'entry'),
            'summary': profile_data.get('summary', ''),
            'skills': [],
            'work_experience': profile_data.get('work_experience', []),
            'education': profile_data.get('education', []),
            'preferences': profile_data.get('preferences', {}),
            'goals': profile_data.get('goals', ''),
        }
        
        # Extract skills properly
        if profile_data.get('skills'):
            for skill in profile_data['skills']:
                if isinstance(skill, dict):
                    skill_name = skill.get('skill_name', '')
                    if skill_name:
                        profile_context['skills'].append(skill_name)
                elif isinstance(skill, str):
                    profile_context['skills'].append(skill)
        
        # Add resume context if available
        resume_context = ""
        if vector_resume_text:
            resume_context = f"\n\nResume Content Analysis:\n{vector_resume_text[:1000]}..."
        
        # Create AI prompt for job analysis - analyze more jobs for better matches
        job_summaries = []
        for job in jobs_list[:50]:  # Increased to 50 jobs for better AI analysis
            job_summary = {
                'job_id': job.get('job_id', 'unknown'),
                'title': job.get('job_title', job.get('title', 'Unknown')),
                'category': job.get('category', job.get('job_category', 'General')),
                'description': job.get('job_description', job.get('description', ''))[:300],  # More description context
                'required_skills': job.get('job_skill_set', job.get('required_skills', []))[:10]  # Limit skills for token efficiency
            }
            job_summaries.append(job_summary)
        
        prompt = f"""You are an elite career matching AI with deep expertise in talent acquisition and career development. Conduct a comprehensive analysis of this professional's profile and identify the TOP 5 most strategically aligned opportunities.

PROFESSIONAL PROFILE ANALYSIS:
═══════════════════════════════════════════
👤 Name: {profile_context['name']}
🎯 Current Title: {profile_context['title']}
📍 Location: {profile_context['location']}
📊 Experience Level: {profile_context['experience_level']}
📝 Professional Summary: {profile_context['summary']}

🛠️ Core Skills: {', '.join(profile_context['skills'][:15])}

💼 Work Experience: {len(profile_context['work_experience'])} positions
🎓 Education: {len(profile_context['education'])} entries
🚀 Career Goals: {profile_context['goals']}

{resume_context}

AVAILABLE OPPORTUNITIES:
═══════════════════════════════════════════
{json.dumps(job_summaries, indent=2)}

COMPREHENSIVE MATCHING CRITERIA:
═══════════════════════════════════════════
1. 🎯 SKILLS ALIGNMENT (50%): Exact matches, transferable skills, emerging technologies
2. 🏢 INDUSTRY FIT (25%): Domain expertise, sector experience, market alignment
3. 📚 EXPERIENCE LEVEL (15%): Role seniority, responsibility scope, career progression
4. 📍 LOCATION COMPATIBILITY (5%): Geographic preferences, remote flexibility
5. 🚀 GROWTH POTENTIAL (5%): Learning opportunities, career advancement, skill development

QUALITY STANDARDS:
- Only recommend roles with >60% overall match
- Prioritize authentic skill alignment over superficial keyword matching
- Consider career trajectory and progression logic
- Evaluate market demand and growth potential
- Include realistic skill gap analysis

Return a JSON response with this EXACT structure:
{{
    "top_matches": [
        {{
            "job_id": "job_id_here",
            "match_percentage": 87,
            "comprehensive_score": 0.87,
            "skill_match_score": 0.92,
            "industry_match_score": 0.85,
            "education_match_score": 0.88,
            "location_match_score": 1.0,
            "career_growth_score": 0.83,
            "matched_skills": ["specific", "technical", "skills", "found"],
            "skill_gaps": ["skill", "to", "develop"],
            "recommendation_reason": "Compelling explanation of strategic career fit with specific examples",
            "growth_opportunities": "Detailed career development pathway and learning opportunities"
        }}
    ],
    "analysis_summary": "Executive summary of matching methodology and key insights"
}}"""

        print("🤖 Requesting AI job matching analysis...")
        
        # Try different models
        models_to_try = ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo']
        for model in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert career matching AI that provides precise job matching analysis. Always return valid JSON responses with comprehensive scoring."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=2000,
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                
                ai_analysis = json.loads(response.choices[0].message.content)
                print(f"✅ AI job matching completed using {model}")
                return ai_analysis
                
            except Exception as model_error:
                print(f"⚠️ Model {model} failed: {model_error}")
                # Check for quota exceeded error
                if "quota" in str(model_error).lower() or "insufficient" in str(model_error).lower():
                    print("💡 OpenAI quota exceeded - generating enhanced mock response")
                    return _generate_enhanced_mock_ai_response(profile_data, jobs_list)
                continue
        
        print("❌ All AI models failed for job matching - generating enhanced mock response")
        return _generate_enhanced_mock_ai_response(profile_data, jobs_list)
        
    except Exception as e:
        print(f"Error in AI job matching: {e}")
        return None

def _generate_enhanced_mock_ai_response(profile_data, jobs_list):
    """Generate simplified mock AI response with clear matching logic"""
    try:
        print("🎯 Generating simplified job matching response...")
        print(f"📊 Profile: {profile_data.get('name', 'Unknown')}")
        print(f"📋 Raw skills data: {profile_data.get('skills', [])}")
        
        # Extract and normalize user skills with synonyms
        user_skills = []
        skill_synonyms = {
            'python': ['python', 'py', 'python3', 'django', 'flask', 'fastapi', 'pandas', 'numpy', 'scikit-learn', 'python developer', 'python programming'],
            'sql': ['sql', 'mysql', 'postgresql', 'postgres', 'database', 'db', 'sql server', 'oracle', 'sqlite', 'database management', 'database developer', 'database analyst'],
            'javascript': ['javascript', 'js', 'node', 'nodejs', 'react', 'vue', 'angular', 'typescript', 'jquery'],
            'java': ['java', 'spring', 'springboot', 'hibernate', 'java developer', 'j2ee', 'jsp'],
            'machine learning': ['ml', 'machine learning', 'ai', 'artificial intelligence', 'data science', 'deep learning', 'neural networks'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'cloud', 'cloud computing', 'devops'],
            'data': ['data', 'analytics', 'data analysis', 'tableau', 'powerbi', 'excel', 'data analyst', 'business intelligence', 'bi'],
            'web': ['web', 'html', 'css', 'frontend', 'backend', 'fullstack', 'web development', 'web developer'],
            'it': ['it', 'information technology', 'tech', 'technology', 'software', 'programming', 'developer', 'engineer', 'analyst', 'consultant'],
            'software': ['software', 'software development', 'software engineer', 'programmer', 'coding', 'development']
        }
        
        if profile_data.get('skills'):
            for skill in profile_data['skills']:
                if isinstance(skill, dict):
                    skill_name = skill.get('skill_name', '')
                else:
                    skill_name = str(skill)
                if skill_name:
                    user_skills.append(skill_name.lower().strip())
        
        user_skills = list(set([skill for skill in user_skills if skill]))
        user_location = (profile_data.get('location') or '').lower()
        user_experience = (profile_data.get('experience_level') or 'entry').lower()
        user_title = (profile_data.get('title') or '').lower()
        user_summary = (profile_data.get('summary') or '').lower()
        
        print(f"🔍 Processed user skills: {user_skills}")
        print(f"💼 User title: {user_title}")
        print(f"📍 User location: {user_location}")
        print(f"📊 Experience level: {user_experience}")
        
        # Industry keywords for better matching
        industry_keywords = {
            'technology': ['software', 'tech', 'it', 'information technology', 'developer', 'engineer', 'programming', 'coding', 'python', 'sql', 'database', 'web development', 'software development', 'application development', 'system', 'technical', 'programmer', 'analyst'],
            'data': ['data', 'analytics', 'scientist', 'analysis', 'insights', 'bi', 'business intelligence', 'sql', 'database', 'data engineer', 'data analyst', 'python', 'machine learning', 'big data', 'data mining'],
            'software': ['software', 'application', 'system', 'platform', 'development', 'programming', 'coding', 'developer', 'engineer', 'architect', 'technical'],
            'database': ['database', 'sql', 'mysql', 'postgresql', 'oracle', 'data', 'dba', 'administrator', 'developer'],
            'finance': ['finance', 'financial', 'banking', 'investment', 'trading', 'analyst', 'fintech'],
            'healthcare': ['healthcare', 'medical', 'health', 'clinical', 'pharma', 'biotech'],
            'consulting': ['consulting', 'consultant', 'advisory', 'strategy', 'management', 'business analyst'],
            'marketing': ['marketing', 'digital', 'social media', 'content', 'advertising'],
            'sales': ['sales', 'business development', 'account', 'relationship', 'revenue'],
            'engineering': ['engineer', 'engineering', 'software engineer', 'systems engineer', 'technical', 'development']
        }
        
        # Enhanced job scoring with multiple factors
        scored_jobs = []
        excluded_hr_jobs_ai = []  # Track excluded HR jobs in AI matching
        print(f"🔍 Analyzing {min(100, len(jobs_list))} jobs for matches...")
        
        for job in jobs_list[:100]:  # Analyze more jobs for better matches
            # Extract skills from available text fields since job_skill_set doesn't exist
            job_keywords = (job.get('keywords') or '').lower()
            job_category = (job.get('category') or '').lower()
            job_title = (job.get('job_title') or job.get('title') or '').lower()
            job_description = (job.get('job_description') or '').lower()
            
            # Extract skills from job text using keyword matching
            job_skills = []
            all_job_text = f"{job_keywords} {job_title} {job_category} {job_description}"
            
            # Common technical skills to look for
            common_skills = [
                'python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 'angular', 'vue',
                'node', 'django', 'flask', 'spring', 'mysql', 'postgresql', 'mongodb', 'redis',
                'aws', 'azure', 'docker', 'kubernetes', 'git', 'machine learning', 'ai',
                'data analysis', 'excel', 'tableau', 'powerbi', 'analytics', 'business intelligence',
                'project management', 'agile', 'scrum', 'leadership', 'communication',
                'sales', 'marketing', 'customer service', 'finance', 'accounting'
            ]
            
            for skill in common_skills:
                if skill in all_job_text:
                    job_skills.append(skill)
            
            job_skills_lower = job_skills
            
            # 1. SIMPLE SKILL MATCHING - if user has all required skills = 100%
            matched_skills = []
            
            print(f"🔍 DEBUG: Processing job '{job_title}' with skills: {job_skills_lower}")
            print(f"👤 User skills: {user_skills}")
            
            for job_skill in job_skills_lower:
                job_skill_clean = job_skill.strip().lower()
                
                # Check if user has this required skill
                for user_skill in user_skills:
                    user_skill_clean = user_skill.strip().lower()
                    
                    # Exact match or contains match
                    if (job_skill_clean == user_skill_clean or 
                        job_skill_clean in user_skill_clean or 
                        user_skill_clean in job_skill_clean):
                        matched_skills.append(job_skill)
                        print(f"✅ Matched: '{job_skill}' (user has '{user_skill}')")
                        break
            
            # Calculate simple skill coverage
            skill_coverage = len(matched_skills) / max(len(job_skills_lower), 1) if job_skills_lower else 0
            skill_score = skill_coverage  # Simple ratio: matched/required
            
            print(f"🎯 Skills calculation: {len(matched_skills)}/{len(job_skills_lower)} = {skill_coverage * 100:.1f}%")
            
            # 2. INDUSTRY/ROLE ALIGNMENT (25% weight) with EXCLUSION RULES
            industry_score = 0.1  # Lower base score
            user_context = f"{user_title} {user_summary} {' '.join(user_skills)}"
            job_context = f"{job_title} {job_category} {job_description}"
            
            # HARD EXCLUSION RULES - Skip completely incompatible industries
            exclusion_rules = {
                'it_tech': {
                    'user_indicators': ['python', 'sql', 'developer', 'programmer', 'software', 'database', 'coding', 'tech', 'it', 'engineer'],
                    'excluded_job_types': ['human resource', 'hr specialist', 'recruitment', 'people operations', 'talent acquisition', 'hr manager', 'hr coordinator', 'hr business partner']
                },
                'hr': {
                    'user_indicators': ['human resource', 'hr', 'recruitment', 'talent', 'people'],
                    'excluded_job_types': ['software developer', 'programmer', 'database', 'python developer', 'sql developer', 'data engineer']
                }
            }
            
            # Check for hard exclusions
            should_exclude = False
            exclusion_reason = ""
            for rule_name, rule in exclusion_rules.items():
                user_has_indicators = any(indicator in user_context.lower() for indicator in rule['user_indicators'])
                job_is_excluded_type = any(excluded_type in job_context.lower() for excluded_type in rule['excluded_job_types'])
                
                if user_has_indicators and job_is_excluded_type:
                    # Track excluded HR jobs for detailed logging
                    if rule_name == 'it_tech':
                        excluded_hr_jobs_ai.append({
                            'job_id': job.get('job_id', 'unknown'),
                            'job_title': job.get('job_title', 'Unknown Title'),
                            'category': job.get('category', 'Unknown Category'),
                            'exclusion_rule': rule_name,
                            'hr_keywords_found': [excluded_type for excluded_type in rule['excluded_job_types'] if excluded_type in job_context.lower()]
                        })
                    
                    print(f"🚫 AI EXCLUDING job {job.get('job_id', 'unknown')}: {job_title} - {rule_name} exclusion rule triggered")
                    should_exclude = True
                    exclusion_reason = f"{rule_name} exclusion rule"
                    break
            
            if should_exclude:
                continue  # Skip this job entirely
            
            # POSITIVE INDUSTRY MATCHING
            for industry, keywords in industry_keywords.items():
                user_industry_match = sum(1 for kw in keywords if kw in user_context.lower()) / len(keywords)
                job_industry_match = sum(1 for kw in keywords if kw in job_context.lower()) / len(keywords)
                
                if user_industry_match > 0.3 and job_industry_match > 0.3:
                    # Strong industry alignment bonus
                    industry_score = max(industry_score, min(user_industry_match, job_industry_match) * 0.95)
                elif user_industry_match > 0.1 and job_industry_match > 0.1:
                    # Moderate industry alignment
                    industry_score = max(industry_score, min(user_industry_match, job_industry_match) * 0.6)
            
            # 3. EXPERIENCE LEVEL MATCH (15% weight)
            experience_keywords = {
                'entry': ['junior', 'entry', 'graduate', 'associate', '0-2 years'],
                'mid': ['mid', 'senior', 'experienced', '3-5 years', '2-7 years'],
                'senior': ['senior', 'lead', 'principal', 'manager', '5+ years', '7+ years']
            }
            
            exp_score = 0.6  # Default
            user_exp_keywords = experience_keywords.get(user_experience, [])
            for keyword in user_exp_keywords:
                if keyword in job_title or keyword in job_description:
                    exp_score = 0.9
                    break
            
            # 4. LOCATION COMPATIBILITY (5% weight)
            location_score = 1.0  # Default for Singapore-based system
            if user_location and 'singapore' not in user_location:
                if 'remote' in job_description or 'hybrid' in job_description:
                    location_score = 0.9
                else:
                    location_score = 0.7
            
            # 5. CAREER GROWTH POTENTIAL (5% weight)
            growth_indicators = ['lead', 'senior', 'manager', 'director', 'growth', 'development', 'advancement']
            growth_score = 0.6 + (sum(1 for indicator in growth_indicators if indicator in job_description) * 0.1)
            growth_score = min(growth_score, 1.0)
            
            # COMPREHENSIVE SCORE CALCULATION with INDUSTRY IMPORTANCE
            # Increase industry weight to prevent cross-industry matches
            comprehensive_score = (
                skill_score * 0.45 + 
                industry_score * 0.35 +  # Increased from 0.25 to 0.35
                exp_score * 0.12 + 
                location_score * 0.04 + 
                growth_score * 0.04
            )
            
            match_percentage = min(comprehensive_score * 100, 98)
            
            # SIMPLIFIED QUALITY FILTER: Show all matches with at least 1 skill match
            has_meaningful_skills = len(matched_skills) >= 1
            meets_basic_threshold = match_percentage >= 15  # Very low threshold to show more matches
            
            # Show match if has at least 1 skill match and meets basic threshold
            if has_meaningful_skills and meets_basic_threshold:
                # Generate intelligent skill gaps
                skill_gaps = []
                for skill in job_skills_lower[:6]:
                    if skill not in [m.lower() for m in matched_skills]:
                        skill_gaps.append(skill)
                
                # Generate contextual recommendations
                recommendation_reason = _generate_match_reasoning(
                    match_percentage, matched_skills, industry_score, skill_score, exp_score
                )
                
                growth_opportunities = _generate_growth_opportunities(
                    job_category, skill_gaps, user_experience
                )
                
                # Calculate simple skills-only percentage for consistency
                skills_only_percentage = (len(matched_skills) / len(job_skills_lower)) * 100 if job_skills_lower else 0
                
                scored_jobs.append({
                    'job_id': job['job_id'],
                    'match_percentage': round(match_percentage, 1),
                    'skills_only_percentage': round(skills_only_percentage, 1),  # Add pure skills match
                    'comprehensive_score': round(comprehensive_score, 3),
                    'skill_match_score': round(skill_score, 3),
                    'industry_match_score': round(industry_score, 3),
                    'education_match_score': round(exp_score, 3),
                    'location_match_score': round(location_score, 3),
                    'career_growth_score': round(growth_score, 3),
                    'matched_skills': matched_skills[:8],
                    'skill_gaps': skill_gaps[:5],
                    'recommendation_reason': recommendation_reason,
                    'growth_opportunities': growth_opportunities
                })
        
        print(f"🎯 Total jobs analyzed: {min(100, len(jobs_list))}")
        print(f"📊 Jobs meeting criteria: {len(scored_jobs)}")
        if scored_jobs:
            print(f"🏆 Top job score: {scored_jobs[0]['comprehensive_score']:.3f}")
            print(f"🎯 Top job skills: {scored_jobs[0]['matched_skills'][:3]}")
        
        # Sort by comprehensive score and intelligent ranking
        scored_jobs.sort(key=lambda x: (x['comprehensive_score'], len(x['matched_skills']), x['skill_match_score']), reverse=True)
        
        # Take top 5 but ensure diversity
        top_matches = []
        used_categories = set()
        
        for job in scored_jobs:
            if len(top_matches) >= 5:
                break
            
            job_category = None
            for job_data in jobs_list:
                if job_data['job_id'] == job['job_id']:
                    job_category = (job_data.get('category') or '').lower()
                    break
            
            # Ensure category diversity in top results
            if len(top_matches) < 3 or job_category not in used_categories:
                top_matches.append(job)
                if job_category:
                    used_categories.add(job_category)
        
        # Fill remaining slots if needed
        while len(top_matches) < 5 and len(top_matches) < len(scored_jobs):
            for job in scored_jobs:
                if job not in top_matches:
                    top_matches.append(job)
                    break
        
        # FALLBACK: If no matches found, provide lower-threshold matches
        if not top_matches and scored_jobs:
            print("⚠️ No high-quality matches found, providing best available matches...")
            top_matches = scored_jobs[:5]  # Take top 5 regardless of strict criteria
        elif not top_matches:
            print("⚠️ No matches found at all - returning empty results")
        
        # Create enhanced AI-style response
        mock_response = {
            "top_matches": top_matches,
            "analysis_summary": f"Advanced AI analysis evaluated {len(jobs_list)} opportunities using multi-factor matching (skills, industry alignment, experience level, growth potential). Found {len(top_matches)} {'high-quality' if top_matches else 'potential'} matches with {sum(len(job.get('matched_skills', [])) for job in top_matches)} total skill alignments for {profile_data.get('name', 'candidate')}."
        }
        
        # Log all excluded HR jobs for debugging
        if excluded_hr_jobs_ai:
            print(f"\n🚫 AI EXCLUDED HR JOBS SUMMARY ({len(excluded_hr_jobs_ai)} total):")
            print("=" * 60)
            for i, excluded_job in enumerate(excluded_hr_jobs_ai, 1):
                print(f"{i:2d}. Job ID: {excluded_job['job_id']}")
                print(f"    Title: {excluded_job['job_title']}")
                print(f"    Category: {excluded_job['category']}")
                print(f"    Exclusion Rule: {excluded_job['exclusion_rule']}")
                print(f"    HR Keywords: {', '.join(excluded_job['hr_keywords_found'])}")
                print()
            print("=" * 60)
            print(f"✅ AI Successfully excluded {len(excluded_hr_jobs_ai)} HR jobs from IT professional matching")
        else:
            print("ℹ️  AI: No HR jobs found to exclude")
        
        print(f"✅ Generated simplified response with {len(top_matches)} matches (showing all found)")
        return mock_response
        
    except Exception as e:
        print(f"❌ Mock response generation failed: {e}")
        return None

def _create_simple_match_reason(match_percentage, matched_skills_count, job_category):
    """Create a user-friendly match reason"""
    if match_percentage >= 70:
        if matched_skills_count >= 3:
            return f"Excellent match! Your skills strongly align with this {job_category.lower()} role, making you a great candidate."
        else:
            return f"Great potential! Your profile shows strong alignment with {job_category.lower()} opportunities."
    elif match_percentage >= 50:
        if matched_skills_count >= 2:
            return f"Good fit! Your skills overlap well with this {job_category.lower()} position, with room for growth."
        else:
            return f"Promising opportunity! This {job_category.lower()} role could be a good next step in your career."
    elif match_percentage >= 30:
        return f"Growth opportunity! This {job_category.lower()} role offers potential to develop new skills while leveraging your existing experience."
    else:
        return f"Career pivot opportunity! Explore this {job_category.lower()} role to expand your professional horizons."

def _generate_match_reasoning(match_percentage, matched_skills, industry_score, skill_score, exp_score):
    """Generate intelligent match reasoning based on scores"""
    reasons = []
    
    if skill_score > 0.7:
        reasons.append(f"Excellent skills alignment with {len(matched_skills)} key competencies")
    elif skill_score > 0.5:
        reasons.append(f"Strong skills match in {len(matched_skills)} areas")
    else:
        reasons.append(f"Growing potential with {len(matched_skills)} transferable skills")
    
    if industry_score > 0.7:
        reasons.append("strong industry fit")
    elif industry_score > 0.5:
        reasons.append("good industry alignment")
    
    if exp_score > 0.8:
        reasons.append("ideal experience level match")
    
    return f"{match_percentage:.0f}% match featuring " + ", ".join(reasons) + ". This role offers excellent career advancement potential."

def _generate_growth_opportunities(job_category, skill_gaps, user_experience):
    """Generate contextual growth opportunities"""
    if not skill_gaps:
        return f"Perfect role for expanding leadership and strategic impact in {job_category}"
    
    key_gaps = skill_gaps[:3]
    
    if user_experience == 'entry':
        return f"Excellent opportunity to develop expertise in {', '.join(key_gaps)} while building foundational experience in {job_category}"
    elif user_experience == 'mid':
        return f"Strategic career move to master {', '.join(key_gaps)} and advance to senior-level responsibilities in {job_category}"
    else:
        return f"Leadership opportunity to leverage existing expertise while expanding into {', '.join(key_gaps)} for comprehensive {job_category} mastery"

# AI Summary Generation Function
def generate_ai_summary(profile_data):
    """Generate AI summary for profile using OpenAI"""
    if not openai or not openai_key or not OpenAI:
        return None
    
    try:
        # Set up OpenAI client (new API format)
        client = OpenAI(api_key=openai_key)
        
        # Extract key information from profile
        name = profile_data.get('name', 'Professional')
        location = profile_data.get('location', 'Singapore')
        experience_level = profile_data.get('experience_level', 'entry')
        
        # Extract skills (handle malformed data)
        skills = []
        for skill in profile_data.get('skills', []):
            if isinstance(skill, dict) and 'skill_name' in skill:
                skill_name = skill['skill_name']
                if skill_name.startswith('["') and skill_name.endswith('"]'):
                    # Parse malformed JSON string
                    skill_list = skill_name[2:-2].split('","')
                    skills.extend(skill_list)
                else:
                    skills.append(skill_name)
            else:
                skills.append(str(skill))
        
        # Extract work experience
        work_exp = profile_data.get('work_experience', [])
        current_role = work_exp[0] if work_exp else None
        
        # Extract education
        education = profile_data.get('education', [])
        highest_education = education[0] if education else None
        
        # Try to read resume content for additional context
        resume_content = ""
        resume_file = profile_data.get('resume_file')
        if resume_file:
            try:
                # Parse PDF content for AI analysis
                import os
                import pdfplumber
                
                resume_path = os.path.join('uploads', 'resumes', resume_file)
                if os.path.exists(resume_path):
                    print(f"📄 Parsing resume PDF: {resume_file}")
                    with pdfplumber.open(resume_path) as pdf:
                        full_text = ""
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                full_text += page_text + "\n"
                        
                        if full_text.strip():
                            resume_content = full_text.strip()
                            print(f"✅ Successfully extracted {len(full_text)} characters from resume")
                        else:
                            resume_content = f"Has resume file: {resume_file} (could not extract text)"
                            print("⚠️ PDF found but no text could be extracted")
                else:
                    resume_content = f"Resume file referenced: {resume_file} (file not found)"
                    print(f"⚠️ Resume file not found at: {resume_path}")
                    
            except ImportError:
                print("⚠️ pdfplumber not available - install with: pip install pdfplumber")
                resume_content = f"Has resume: {resume_file}"
            except Exception as e:
                print(f"❌ Could not read resume file {resume_file}: {e}")
                resume_content = f"Has resume: {resume_file}"
        
        # Create prompt for AI summary
        # Build detailed context
        context_details = []
        if current_role:
            years_exp = current_role.get('years', 0)
            context_details.append(f"{years_exp} years experience as {current_role.get('position')} at {current_role.get('company')}")
        if highest_education:
            context_details.append(f"{highest_education.get('degree')} in {highest_education.get('field_of_study')} from {highest_education.get('institution')}")
        if resume_content:
            context_details.append(resume_content)

        # Build comprehensive work description
        work_description = ""
        if current_role:
            work_description = f"Currently works as {current_role.get('position')} at {current_role.get('company')} with {current_role.get('years', 0)} years of experience"
            if current_role.get('description'):
                work_description += f", specializing in {current_role.get('description')}"
        
        # Build education description
        education_description = ""
        if highest_education:
            education_description = f"Holds a {highest_education.get('degree')} in {highest_education.get('field_of_study')} from {highest_education.get('institution')}"
        
        prompt = f"""Write a compelling professional summary for {name}.

Profile Details:
- Location: {location}
- Experience Level: {experience_level}
- Primary Skills: {', '.join(skills[:5]) if skills else 'Various technical skills'}
- Work Experience: {work_description if work_description else 'Building professional experience'}
- Education: {education_description if education_description else 'Continuing professional development'}
- Career Focus: Looking for opportunities in {location} with skills in {', '.join(skills[:3]) if skills else 'technology'}

Create a professional, engaging summary that highlights their unique value proposition, technical expertise, and career potential. Make it sound accomplished and forward-looking. Keep it under 280 characters but make every word count."""

        # Call OpenAI API with ChatGPT Pro models (new format)
        print(f"Generating AI summary for {name}...")  # Debug log
        
        # Try ChatGPT Pro models first, then fallback
        models_to_try = ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo']
        for model in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert career writer who creates compelling professional summaries. Write engaging, accomplished-sounding summaries that showcase the person's expertise and potential. Use dynamic language and focus on achievements and capabilities."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=120,
                    temperature=0.8
                )
                
                summary = response.choices[0].message.content.strip()
                print(f"✅ Generated AI summary using {model}: {summary[:50]}...")
                return summary
                
            except Exception as model_error:
                print(f"⚠️ Model {model} failed: {model_error}")
                continue
        
        # If all models fail
        print("❌ All AI models failed for summary generation")
        return None
        
    except Exception as e:
        print(f"Error generating AI summary: {e}")
        print(f"OpenAI available: {openai is not None}")
        print(f"OpenAI key available: {openai_key is not None}")
        return None

# Initialize Flask app with production configuration
app = Flask(__name__)

# Production configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'skillmatch-production-key-change-me')
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
app.config['ENV'] = os.environ.get('FLASK_ENV', 'production')
app.config['TEMPLATES_AUTO_RELOAD'] = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

# Domain configuration - using localhost for development
app.config['SERVER_NAME'] = os.environ.get('SERVER_NAME', None)  # Set in production
app.config['PREFERRED_URL_SCHEME'] = 'http' if os.environ.get('FLASK_ENV') != 'production' else 'https'

# CORS configuration - allow localhost for development
cors_origins = os.environ.get('CORS_ORIGINS', 'http://localhost:5000,http://127.0.0.1:5000,http://localhost:5001,http://127.0.0.1:5001').split(',')
CORS(app, origins=cors_origins if os.environ.get('FLASK_ENV') == 'production' else "*")

# Initialize SocketIO with production settings
socketio = SocketIO(
    app, 
    cors_allowed_origins=cors_origins if os.environ.get('FLASK_ENV') == 'production' else "*",
    async_mode='threading',  # Using threading instead of eventlet
    logger=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true',
    engineio_logger=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
)

# Global variables
data_loader = None
skill_matcher = None
current_agent = None


def load_config() -> Dict[str, Any]:
    """Load configuration from file or environment"""
    config = {}
    
    # Try to load from config file
    config_path = Path("../config/config.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # Override with environment variables
    if "GITHUB_TOKEN" in os.environ:
        config["github_token"] = os.environ["GITHUB_TOKEN"]
    
    return config


def initialize_data():
    """Initialize data loader and skill matcher"""
    global data_loader, skill_matcher
    
    if not SKILLMATCH_AVAILABLE:
        print("SkillMatch modules not available - using mock data")
        data_loader = None
        skill_matcher = None
        return False
    
    try:
        data_loader = DataLoader(
            skills_db_path="../data/skills_database.json",
            opportunities_db_path="../data/opportunities_database.json"
        )
        skill_matcher = SkillMatcher(data_loader.skills_data)
        return True
    except Exception as e:
        print(f"Error initializing data: {e}")
        data_loader = None
        skill_matcher = None
        return False


@app.route('/')
def index():
    """Main dashboard page with summary overview"""
    import json
    config = load_config()
    
    # Get database statistics including dashboard data
    stats = {
        'skills_categories': 0,
        'total_opportunities': 0,
        'total_jobs': 0,
        'total_profiles': 0,
        'job_categories': {},
        'chart_data': {'categories': [], 'values': []},
        'last_scrape': 'Never',
        'github_configured': bool(config.get('github_token')),
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    if data_loader:
        if hasattr(data_loader, 'skills_data') and data_loader.skills_data:
            stats['skills_categories'] = len(data_loader.skills_data.get('skills', {}))
        
        if hasattr(data_loader, 'opportunities_data') and data_loader.opportunities_data:
            stats['total_opportunities'] = len(data_loader.opportunities_data.get('opportunities', []))
    
    # Get job statistics using EXACT same approach as working jobs_listing route
    # Move this OUTSIDE data_loader condition so it always executes
    try:
        print("🔍 HOME: Attempting database imports...")
        # Using global imports: Job, UserProfile already imported
        try:
            from database.db_config import DatabaseConfig
        except ImportError:
            try:
                from web.database.db_config import DatabaseConfig
            except ImportError:
                # Create minimal fallback
                class DatabaseConfig:
                    @staticmethod
                    def get_database_url(): return 'sqlite:///job_opportunities.db'
            from web.database.db_config import DatabaseConfig
        print("🔍 HOME: Database imports successful!")
        
        # Get database session (exact same pattern as jobs_listing)
        print("🔍 HOME: Creating database config...")
        db_config = DatabaseConfig()
        print("🔍 HOME: Creating session...")
        session = db_config.SessionLocal()
        print("🔍 HOME: Session created successfully!")
        
        try:
            # Count total jobs (exact same as jobs_listing)
            jobs = session.query(Job).filter(Job.is_active == True).all()
            stats['total_jobs'] = len(jobs)
            
            # Count total profiles (only active ones to match Profiles route)
            profiles = session.query(UserProfile).filter(UserProfile.is_active == True).all()
            stats['total_profiles'] = len(profiles)
            
            # Get job categories for chart (handle Python lists from database)
            category_counts = {}
            jobs_with_categories = 0
            
            for job in jobs:
                if job.job_category:
                    jobs_with_categories += 1
                    try:
                        # Handle Python list objects directly (most common case)
                        if isinstance(job.job_category, list):
                            for category in job.job_category:
                                if category and isinstance(category, str):
                                    category_counts[category] = category_counts.get(category, 0) + 1
                        # Handle JSON string format like '["F&B"]' 
                        elif isinstance(job.job_category, str) and job.job_category.startswith('['):
                            import json
                            categories = json.loads(job.job_category)
                            if isinstance(categories, list):
                                for category in categories:
                                    if category and isinstance(category, str):
                                        category_counts[category] = category_counts.get(category, 0) + 1
                        # Handle plain string categories
                        elif isinstance(job.job_category, str) and job.job_category.strip():
                            category_counts[job.job_category.strip()] = category_counts.get(job.job_category.strip(), 0) + 1
                    except Exception as parse_error:
                        print(f"🔍 Category parsing error for '{job.job_category}': {parse_error}")
            
            # Debug info removed - categories processing working correctly
            
            # Sort categories by count and take top 10
            stats['job_categories'] = dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10])
            
            # Generate chart data for Plotly
            if stats['job_categories']:
                stats['chart_data'] = {
                    'categories': list(stats['job_categories'].keys()),
                    'values': list(stats['job_categories'].values())
                }
            
            print(f"🏠 HOME page stats: jobs={stats['total_jobs']}, profiles={stats['total_profiles']}, categories={len(stats['job_categories'])}")
            
        finally:
            session.close()
            
    except Exception as db_error:
        print(f"⚠️ Database error in HOME: {db_error}")
        print(f"⚠️ Database error type: {type(db_error).__name__}")
        import traceback
        traceback.print_exc()
        # Simple fallback
        stats['total_jobs'] = 0
        stats['total_profiles'] = 0
            
    # Also try file-based profile counting as additional fallback
    profiles_dir = Path(__file__).parent.parent / "profiles"
    if profiles_dir.exists():
        file_count = len(list(profiles_dir.glob("*.json")))
        if file_count > 0:
            stats['total_profiles'] = max(stats['total_profiles'], file_count)
    
    # Check for last scrape date
    scraped_data_dir = Path("../scraped_data")
    if scraped_data_dir.exists():
        scrape_files = list(scraped_data_dir.glob("*_raw_*.json"))
        if scrape_files:
            latest_scrape = max(scrape_files, key=lambda p: p.stat().st_mtime)
            stats['last_scrape'] = datetime.fromtimestamp(latest_scrape.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    
    # Get profiles data for analytics
    profiles_dir = Path(__file__).parent.parent / "profiles"
    profile_files = []
    
    if profiles_dir.exists():
        for profile_file in profiles_dir.glob("*.json"):
            try:
                with open(profile_file, 'r') as f:
                    profile_data = json.load(f)
                    profile_files.append(profile_data)
            except Exception as e:
                print(f"Error loading profile {profile_file}: {e}")
    
    # Generate Plotly chart data for job categories (moved outside profiles check)
    chart_data = None
    if stats['job_categories']:
            import json
            categories = list(stats['job_categories'].keys())
            counts = list(stats['job_categories'].values())
            
            print(f"📊 Chart data - Categories: {len(categories)}, Total jobs in chart: {sum(counts)}")
            print(f"📊 Sample categories: {categories[:3] if categories else 'None'}")
            
            # Truncate long category names for display while keeping full names for hover
            display_categories = []
            full_categories = []
            for cat in categories:
                full_categories.append(cat)
                if len(cat) > 28:  # Further increased for better display
                    display_categories.append(cat[:25] + '...')  # Show even more characters
                else:
                    display_categories.append(cat)
            
            # Create professional gradient colors - 4 blue, 3 teal, 3 orange pattern
            def generate_gradient_colors(num_categories):
                """Generate colors with pattern: 4 blue (strong/light), 3 teal (strong/light), 3 orange (strong/light)"""
                colors = []
                border_colors = []
                
                # Define base colors
                blue_color = (26, 54, 93)    # #1a365d - Dark Blue
                teal_color = (56, 178, 172)  # #38b2ac - Teal  
                orange_color = (237, 137, 54) # #ed8936 - Orange
                
                for i in range(num_categories):
                    # Determine which color group this bar belongs to
                    if i < 4:
                        # First 4 bars: Blue
                        r, g, b = blue_color
                    elif i < 7:
                        # Next 3 bars: Teal
                        r, g, b = teal_color
                    else:
                        # Last 3+ bars: Orange
                        r, g, b = orange_color
                    
                    # Alternate between strong and light within each color group
                    if i % 2 == 0:
                        # Strong version - slightly enhanced
                        opacity = 0.95
                        r = min(r + 15, 255)
                        g = min(g + 15, 255)
                        b = min(b + 15, 255)
                    else:
                        # Light version - softer tones
                        opacity = 0.85
                        r = min(r + 30, 255)
                        g = min(g + 30, 255)
                        b = min(b + 30, 255)
                    
                    colors.append(f'rgba({r}, {g}, {b}, {opacity})')
                    
                    # Darker border colors
                    border_r = max(r - 20, 10)
                    border_g = max(g - 20, 10)
                    border_b = max(b - 20, 10)
                    border_colors.append(f'rgba({border_r}, {border_g}, {border_b}, 1.0)')
                
                return colors, border_colors
            
            # Generate gradient colors for all categories
            bar_colors, bar_border_colors = generate_gradient_colors(len(categories))
            
            chart_data = {
                'data': [{
                    'x': categories,
                    'y': counts,
                    'type': 'bar',
                    'marker': {
                        'color': bar_colors,
                        'line': {
                            'color': bar_border_colors,
                            'width': 2
                        },
                        'opacity': 0.9
                    },
                    'hovertemplate': '<b style="font-size: 14px; color: #1f2937;">%{x}</b><br>' +
                                   '<span style="font-size: 16px; font-weight: bold; color: #059669;">%{y} Jobs</span>' +
                                   '<br><i style="color: #6b7280; font-size: 12px;">Click to view jobs</i><extra></extra>',
                    'hoverlabel': {
                        'bgcolor': 'rgba(255, 255, 255, 0.95)',
                        'bordercolor': 'rgba(229, 231, 235, 1)',
                        'borderwidth': 1,
                        'font': {'size': 13, 'family': 'Inter, system-ui, sans-serif'}
                    }
                }],
                'layout': {
                    'title': {
                        'text': '<b>Job Distribution by Category</b>',
                        'font': {
                            'size': 24, 
                            'family': 'Inter, system-ui, sans-serif', 
                            'color': '#1f2937'
                        },
                        'x': 0.5,
                        'xanchor': 'center',
                        'pad': {'t': 20, 'b': 20}
                    },
                    'yaxis': {
                        'title': {
                            'text': '<b>Number of Jobs</b>',
                            'font': {'size': 14, 'family': 'Inter, system-ui, sans-serif', 'color': '#374151'}
                        },
                        'tickfont': {'size': 12, 'family': 'Inter, system-ui, sans-serif', 'color': '#6b7280'},
                        'gridcolor': 'rgba(229, 231, 235, 0.4)',
                        'zerolinecolor': 'rgba(156, 163, 175, 0.5)',
                        'linecolor': 'rgba(209, 213, 219, 0.8)',
                        'linewidth': 1
                    },
                    'plot_bgcolor': 'rgba(249, 250, 251, 0.3)',
                    'paper_bgcolor': 'rgba(255, 255, 255, 0)',
                    'margin': {'t': 80, 'l': 70, 'r': 40, 'b': 140},
                    'height': 480,
                    'font': {'family': 'Inter, system-ui, sans-serif'},
                    'hoverlabel': {
                        'bgcolor': 'rgba(255, 255, 255, 0.95)',
                        'bordercolor': 'rgba(229, 231, 235, 1)',
                        'borderwidth': 1
                    }
                }
            }
    else:
        print("⚠️ No job categories found for chart generation")
        # Ensure we always have chart data even if empty
        stats['chart_data'] = {'categories': [], 'values': []}
    
    # Debug: Show final stats being passed to template
    print(f"🏠 HOME page stats: jobs={stats.get('total_jobs', 0)}, profiles={stats.get('total_profiles', 0)}, categories={len(stats.get('job_categories', {}))}")
    # Chart data ready for template rendering
    
    return render_template('index.html', stats=stats, profiles=profile_files, chart_data=json.dumps(chart_data) if chart_data else None)

@app.route('/dashboard')
def dashboard():
    """Dashboard with summary overview and analytics"""
    try:
        # Get database statistics
        dashboard_stats = {
            'total_jobs': 0,
            'total_profiles': 0,
            'job_categories': {},
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Get job statistics from database
        try:
            # Using global imports: Job, UserProfile already imported
            try:
                from database.db_config import db_config
            except ImportError:
                try:
                    from web.database.db_config import db_config
                except ImportError:
                    # Create minimal fallback
                    class MinimalDBConfig:
                        @contextlib.contextmanager
                        def session_scope(self):
                            yield None
                    db_config = MinimalDBConfig()
            
            with db_config.session_scope() as session:
                # Count total jobs
                dashboard_stats['total_jobs'] = session.query(Job).filter(Job.is_active == True).count()
                
                # Count total profiles
                dashboard_stats['total_profiles'] = session.query(UserProfile).count()
                
                # Get job categories distribution
                jobs_with_categories = session.query(Job).filter(
                    Job.is_active == True,
                    Job.job_category.isnot(None)
                ).all()
                
                category_counts = {}
                for job in jobs_with_categories:
                    if job.job_category:
                        if isinstance(job.job_category, list):
                            for category in job.job_category:
                                if category and isinstance(category, (str, dict)):
                                    cat_name = str(category) if isinstance(category, str) else category.get('name', str(category))
                                    if cat_name:
                                        category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
                        elif isinstance(job.job_category, str):
                            if job.job_category:
                                category_counts[job.job_category] = category_counts.get(job.job_category, 0) + 1
                
                # Sort categories by count and take top 10
                dashboard_stats['job_categories'] = dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10])
                
        except Exception as db_error:
            print(f"⚠️ Database error in dashboard: {db_error}")
            # Fallback to file-based profile counting
            profiles_dir = Path(__file__).parent.parent / "profiles"
            if profiles_dir.exists():
                dashboard_stats['total_profiles'] = len(list(profiles_dir.glob("*.json")))
        
        # Generate Plotly chart data for job categories
        chart_data = None
        if dashboard_stats['job_categories']:
            import json
            categories = list(dashboard_stats['job_categories'].keys())
            counts = list(dashboard_stats['job_categories'].values())
            
            # Truncate long category names for display while keeping full names for hover
            display_categories = []
            full_categories = []
            for cat in categories:
                full_categories.append(cat)
                if len(cat) > 28:  # Further increased for better display
                    display_categories.append(cat[:25] + '...')  # Show even more characters
                else:
                    display_categories.append(cat)
            
            # Create professional gradient colors - 4 blue, 3 teal, 3 orange pattern
            def generate_gradient_colors_dashboard(num_categories):
                """Generate colors with pattern: 4 blue (strong/light), 3 teal (strong/light), 3 orange (strong/light) for dashboard"""
                colors = []
                border_colors = []
                
                # Define base colors
                blue_color = (26, 54, 93)    # #1a365d - Dark Blue
                teal_color = (56, 178, 172)  # #38b2ac - Teal  
                orange_color = (237, 137, 54) # #ed8936 - Orange
                
                for i in range(num_categories):
                    # Determine which color group this bar belongs to
                    if i < 4:
                        # First 4 bars: Blue
                        r, g, b = blue_color
                    elif i < 7:
                        # Next 3 bars: Teal
                        r, g, b = teal_color
                    else:
                        # Last 3+ bars: Orange
                        r, g, b = orange_color
                    
                    # Alternate between strong and light within each color group
                    if i % 2 == 0:
                        # Strong version - slightly enhanced
                        opacity = 0.9
                        r = min(r + 15, 255)
                        g = min(g + 15, 255)
                        b = min(b + 15, 255)
                    else:
                        # Light version - softer tones
                        opacity = 0.75
                        r = min(r + 30, 255)
                        g = min(g + 30, 255)
                        b = min(b + 30, 255)
                    
                    colors.append(f'rgba({r}, {g}, {b}, {opacity})')
                    
                    # Darker border colors
                    border_r = max(r - 20, 10)
                    border_g = max(g - 20, 10)
                    border_b = max(b - 20, 10)
                    border_colors.append(f'rgba({border_r}, {border_g}, {border_b}, 1.0)')
                
                return colors, border_colors
            
            # Generate gradient colors for all categories
            bar_colors, bar_border_colors = generate_gradient_colors_dashboard(len(categories))
            
            chart_data = {
                'data': [{
                    'x': display_categories,
                    'y': counts,
                    'customdata': full_categories,
                    'type': 'bar',
                    'marker': {
                        'color': bar_colors,
                        'line': {
                            'color': bar_border_colors,
                            'width': 2
                        },
                        'opacity': 0.9
                    },
                    'hovertemplate': '<b style="font-size: 14px; color: #1f2937;">%{customdata}</b><br>' +
                                   '<span style="font-size: 16px; font-weight: bold; color: #059669;">%{y} Jobs</span>' +
                                   '<br><i style="color: #6b7280; font-size: 12px;">Click to view jobs</i><extra></extra>',
                    'hoverlabel': {
                        'bgcolor': 'rgba(255, 255, 255, 0.95)',
                        'bordercolor': 'rgba(229, 231, 235, 1)',
                        'borderwidth': 1,
                        'font': {'size': 13, 'family': 'Inter, system-ui, sans-serif'}
                    }
                }],
                'layout': {
                    'title': {
                        'text': '<b>Job Distribution by Category</b>',
                        'font': {
                            'size': 24, 
                            'family': 'Inter, system-ui, sans-serif', 
                            'color': '#1f2937'
                        },
                        'x': 0.5,
                        'xanchor': 'center',
                        'pad': {'t': 20, 'b': 20}
                    },
                    'xaxis': {
                        'title': {
                            'text': '<b>Job Categories</b>',
                            'font': {'size': 14, 'family': 'Inter, system-ui, sans-serif', 'color': '#374151'}
                        },
                        'tickangle': -45,
                        'tickfont': {'size': 11, 'family': 'Inter, system-ui, sans-serif', 'color': '#6b7280'},
                        'gridcolor': 'rgba(229, 231, 235, 0.3)',
                        'zerolinecolor': 'rgba(156, 163, 175, 0.4)',
                        'linecolor': 'rgba(209, 213, 219, 0.8)',
                        'linewidth': 1
                    },
                    'yaxis': {
                        'title': {
                            'text': '<b>Number of Jobs</b>',
                            'font': {'size': 14, 'family': 'Inter, system-ui, sans-serif', 'color': '#374151'}
                        },
                        'tickfont': {'size': 12, 'family': 'Inter, system-ui, sans-serif', 'color': '#6b7280'},
                        'gridcolor': 'rgba(229, 231, 235, 0.4)',
                        'zerolinecolor': 'rgba(156, 163, 175, 0.5)',
                        'linecolor': 'rgba(209, 213, 219, 0.8)',
                        'linewidth': 1
                    },
                    'plot_bgcolor': 'rgba(249, 250, 251, 0.3)',
                    'paper_bgcolor': 'rgba(255, 255, 255, 0)',
                    'margin': {'t': 80, 'l': 70, 'r': 40, 'b': 140},
                    'height': 480,
                    'font': {'family': 'Inter, system-ui, sans-serif'},
                    'hoverlabel': {
                        'bgcolor': 'rgba(255, 255, 255, 0.95)',
                        'bordercolor': 'rgba(229, 231, 235, 1)',
                        'borderwidth': 1
                    }
                }
            }
        
        return render_template('dashboard.html', stats=dashboard_stats, chart_data=json.dumps(chart_data) if chart_data else None)
        
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
        return render_template('dashboard.html', stats={'total_jobs': 0, 'total_profiles': 0, 'job_categories': {}}, chart_data=None)


@app.route('/profiles')
def profiles():
    """Profile management page"""
    try:
        # Use the profile manager for storage abstraction
        profiles_data = profile_manager.list_profiles()
        profile_files = []
        
        print(f"📊 Loading {len(profiles_data)} profiles...")
        
        for profile_data in profiles_data:
            # Create profile object with enhanced data structure
            profile_obj = {
                'id': profile_data.get('user_id', profile_data.get('name', 'unknown')),
                'filename': f"{profile_data.get('user_id', profile_data.get('name', 'unknown'))}.json",
                'name': profile_data.get('name', 'Unknown'),
                'email': profile_data.get('email', ''),
                'title': profile_data.get('title', 'No title'),
                'experience_level': profile_data.get('experience_level', 'not_specified'),
                'skills': profile_data.get('skills', []),
                'industries': profile_data.get('industries', []),
                'location': profile_data.get('location', ''),
                'resume_file': profile_data.get('resume_file'),
                'skills_count': len(profile_data.get('skills', [])),
                'experience_years': sum(exp.get('years', 0) or 0 for exp in profile_data.get('work_experience', [])),
                'created_at': parse_datetime(profile_data.get('created_at')) or datetime.now(),
                'modified': parse_datetime(profile_data.get('updated_at')) or datetime.now()
            }
            profile_files.append(profile_obj)
            
        # Sort profiles by name
        profile_files.sort(key=lambda x: x['name'].lower())
            
        # Show storage info (skip if method doesn't exist)
        try:
            storage_info = profile_manager.get_storage_info()
            print(f"✅ Using {storage_info['type']} storage for profiles")
        except AttributeError:
            print("✅ Using profile storage (method unavailable)")
            
    except Exception as e:
        print(f"Error loading profiles: {e}")
        profile_files = []
    
    return render_template('profiles.html', profiles=profile_files)


@app.route('/jobs')
def jobs_listing():
    """Display all jobs from database"""
    print("🔍 JOBS LISTING: Starting job listing request...")
    try:
        # Using global imports: Job already imported
        try:
            from database.db_config import db_config
        except ImportError:
            try:
                from web.database.db_config import db_config
            except ImportError:
                # Create minimal fallback
                class MinimalDBConfig:
                    @contextlib.contextmanager
                    def session_scope(self):
                        yield None
                db_config = MinimalDBConfig()
                print("⚠️ Using fallback db_config")
        
        # Use the same pattern as other working routes
        with db_config.session_scope() as session:
            if session is None:
                print("⚠️ Database session is None - using fallback")
                return render_template('jobs_listing.html', jobs=[], total_jobs=0)
                
            # Get all active jobs
            jobs = session.query(Job).filter(Job.is_active == True).order_by(Job.created_at.desc()).all()
            
            # Convert to dictionaries for template
            jobs_data = [job.to_dict() for job in jobs]
            
            print(f"📋 Loaded {len(jobs_data)} jobs for listing")
            
            return render_template('jobs_listing.html', 
                                 jobs=jobs_data, 
                                 total_jobs=len(jobs_data))
            
    except Exception as e:
        print(f"Error loading jobs: {e}")
        import traceback
        traceback.print_exc()
        flash('Error loading jobs listing', 'error')
        return render_template('jobs_listing.html', jobs=[], total_jobs=0)


@app.route('/profile/create')
def create_profile():
    """Profile creation form"""
    # Get available skills for the form
    skills_data = []
    if data_loader and data_loader.skills_data:
        for category, category_data in data_loader.skills_data.get('skills', {}).items():
            for skill_id, skill_info in category_data.get('skills', {}).items():
                skills_data.append({
                    'id': skill_id,
                    'name': skill_info.get('name', skill_id),
                    'category': category_data.get('category_name', category)
                })
    
    return render_template('create_profile.html', available_skills=skills_data)


@app.route('/profile/save', methods=['POST'])
def save_profile():
    """Save user profile (create new or update existing)"""
    try:
        # Check if this is an edit operation
        edit_profile_id = request.form.get('edit_profile_id')
        is_editing = bool(edit_profile_id)
        
        # Get form data
        profile_data = {
            'name': request.form.get('name'),
            'email': request.form.get('email'),
            'title': request.form.get('title'),
            'location': request.form.get('location'),
            'bio': request.form.get('bio'),
            'goals': request.form.get('goals', ''),
            'summary': request.form.get('summary', ''),
            'skills': [],
            'work_experience': [],
            'education': [],
            'preferences': {
                'work_types': [request.form.get('work_type')] if request.form.get('work_type') else [],
                'locations': [request.form.get('location')] if request.form.get('location') else [],
                'industries': json.loads(request.form.get('industries', '[]')) if request.form.get('industries') else [],
                'salary_min': int(float(request.form.get('salary_min'))) if request.form.get('salary_min') and request.form.get('salary_min').strip() else None,
                'salary_max': int(float(request.form.get('salary_max'))) if request.form.get('salary_max') and request.form.get('salary_max').strip() else None,
                'remote_preference': request.form.get('remote_preference', 'hybrid')
            }
        }
        
        # If editing, preserve existing resume_file unless new one is uploaded
        existing_resume_file = None
        if is_editing:
            existing_data = profile_manager.load_profile(edit_profile_id)
            if existing_data:
                existing_resume_file = existing_data.get('resume_file')
        
        profile_data['resume_file'] = existing_resume_file
        
        # Set experience level - prefer direct selection over calculated from years
        profile_data['experience_level'] = request.form.get('experience_level', 'entry')
        
        # Process skills
        skills_raw = request.form.get('skills', '[]')
        try:
            # Try to parse as JSON first (from the new form)
            selected_skills = json.loads(skills_raw) if skills_raw else []
        except json.JSONDecodeError:
            # Fallback to old way if JSON parsing fails
            selected_skills = request.form.getlist('skills')
        
        for skill_id in selected_skills:
            skill_level = request.form.get(f'skill_level_{skill_id}', 'intermediate')
            skill_years = int(float(request.form.get(f'skill_years_{skill_id}', 1)))
            
            # Find skill info
            skill_name = skill_id
            skill_category = 'other'
            
            if data_loader and data_loader.skills_data:
                for category, category_data in data_loader.skills_data.get('skills', {}).items():
                    if skill_id in category_data.get('skills', {}):
                        skill_name = category_data['skills'][skill_id].get('name', skill_id)
                        skill_category = category
                        break
            
            profile_data['skills'].append({
                'skill_id': skill_id,
                'skill_name': skill_name,
                'category': skill_category,
                'level': skill_level,
                'years_experience': skill_years
            })
        
        # Add work experience if provided
        if request.form.get('job_title'):
            profile_data['work_experience'].append({
                'position': request.form.get('job_title'),
                'company': request.form.get('company'),
                'years': int(float(request.form.get('experience_years'))) if request.form.get('experience_years') and request.form.get('experience_years').strip() else 0,
                'description': request.form.get('job_description', ''),
                'employment_status': request.form.get('employment_status', ''),
                'key_skills': selected_skills
            })
        
        # Add education if provided
        if request.form.get('degree'):
            profile_data['education'].append({
                'degree': request.form.get('degree'),
                'institution': request.form.get('institution'),
                'graduation_year': int(float(request.form.get('graduation_year'))) if request.form.get('graduation_year') and request.form.get('graduation_year').strip() else datetime.now().year,
                'field_of_study': request.form.get('field_of_study', '')
            })
        
        # Handle resume upload (replace existing if new file uploaded)
        if 'resume' in request.files:
            resume_file = request.files['resume']
            if resume_file and resume_file.filename and resume_file.filename.endswith('.pdf'):
                # Create uploads directory
                uploads_dir = Path(__file__).parent.parent / "uploads" / "resumes"
                uploads_dir.mkdir(parents=True, exist_ok=True)
                
                # If editing and old resume exists, delete it first
                if is_editing and existing_resume_file:
                    old_resume_path = uploads_dir / existing_resume_file
                    if old_resume_path.exists():
                        old_resume_path.unlink()
                        print(f"Deleted old resume: {existing_resume_file}")
                
                # Generate filename (one resume per profile)
                resume_filename = f"{profile_data['name'].lower().replace(' ', '_')}_resume.pdf"
                resume_path = uploads_dir / resume_filename
                
                # Save the new file
                resume_file.save(str(resume_path))
                profile_data['resume_file'] = resume_filename
                print(f"Saved new resume: {resume_filename}")
                
                # Add resume to vector database
                if VECTOR_SEARCH_AVAILABLE:
                    try:
                        vector_service = get_vector_service()
                        success = vector_service.add_resume_to_vector_db(
                            profile_id=profile_data['name'].lower().replace(' ', '_'),
                            pdf_path=str(resume_path),
                            metadata={
                                'name': profile_data['name'],
                                'title': profile_data.get('title', ''),
                                'created_at': datetime.now().isoformat()
                            }
                        )
                        if success:
                            print(f"✅ Resume added to vector database: {resume_filename}")
                        else:
                            print(f"⚠️ Failed to add resume to vector database: {resume_filename}")
                    except Exception as e:
                        print(f"❌ Vector search error: {e}")
                
                # Generate AI summary from PDF if no existing summary
                if not profile_data.get('summary'):
                    try:
                        from .utils.pdf_extractor import extract_resume_text
                        from .utils.ai_summarizer import generate_profile_summary
                        
                        print("🔍 Analyzing PDF for professional summary...")
                        
                        # Extract text from PDF
                        pdf_result = extract_resume_text(str(resume_path))
                        
                        if pdf_result['success']:
                            print(f"✅ PDF text extracted ({pdf_result['word_count']} words)")
                            
                            # Generate AI summary
                            summary_result = generate_profile_summary(
                                pdf_result['text'], 
                                profile_data
                            )
                            
                            if summary_result['success']:
                                profile_data['summary'] = summary_result['summary']
                                print(f"✅ AI summary generated using {summary_result['model_used']}")
                                print(f"📝 Summary: {summary_result['summary'][:100]}...")
                            else:
                                print(f"⚠️ AI summary generation failed: {summary_result['error']}")
                        else:
                            print(f"⚠️ PDF text extraction failed: {pdf_result['error']}")
                            
                    except Exception as e:
                        print(f"⚠️ PDF analysis error (continuing without summary): {e}")
        
        # Save profile
        profiles_dir = Path(__file__).parent.parent / "profiles"
        profiles_dir.mkdir(exist_ok=True)
        
        # Set profile ID for storage
        if is_editing and edit_profile_id:
            profile_data['user_id'] = edit_profile_id
        else:
            profile_data['user_id'] = profile_data['name'].lower().replace(' ', '_')
        
        # Use profile manager to save profile  
        success = profile_manager.save_profile(profile_data)
        
        if success:
            if is_editing:
                flash(f"✅ Profile '{profile_data['name']}' updated successfully!", "success")
            else:
                flash(f"🎉 Profile '{profile_data['name']}' created successfully!", "success")
        else:
            flash("❌ Error saving profile to database", "error")
        return redirect(url_for('profiles'))
        
    except Exception as e:
        flash(f"Error saving profile: {e}", "error")
        # If editing, redirect back to edit page; otherwise to create page
        if is_editing and edit_profile_id:
            return redirect(url_for('edit_profile', profile_id=edit_profile_id))
        else:
            return redirect(url_for('create_profile'))


@app.route('/profiles/<profile_id>')
def view_profile(profile_id):
    """View individual profile details"""
    try:
        # Use PostgreSQL storage instead of JSON files
        profile_data = profile_manager.load_profile(profile_id)
        
        if not profile_data:
            flash("Profile not found.", "error")
            return redirect(url_for('profiles'))
        
        # Generate AI summary from PDF if it doesn't exist
        if not profile_data.get('summary'):
            # Check if there's a PDF resume file
            resume_filename = profile_data.get('resume_file')
            if resume_filename:
                try:
                    uploads_dir = Path(__file__).parent.parent / "uploads" / "resumes"
                    resume_path = uploads_dir / resume_filename
                    
                    if resume_path.exists():
                        print(f"🔍 Generating summary from PDF: {resume_filename}")
                        
                        # Extract text from PDF
                        from .utils.pdf_extractor import extract_resume_text
                        from .utils.ai_summarizer import generate_profile_summary
                        
                        pdf_result = extract_resume_text(str(resume_path))
                        
                        if pdf_result['success']:
                            print(f"✅ PDF text extracted ({pdf_result['word_count']} words)")
                            
                            # Generate AI summary from extracted text
                            summary_result = generate_profile_summary(
                                pdf_result['text'], 
                                profile_data
                            )
                            
                            if summary_result['success']:
                                profile_data['summary'] = summary_result['summary']
                                print(f"✅ AI summary generated using {summary_result['model_used']}")
                                
                                # Save the updated profile with AI summary
                                try:
                                    profile_manager.save_profile(profile_data)
                                    print("✅ Profile updated with AI summary")
                                except Exception as e:
                                    print(f"Warning: Could not save AI summary to profile: {e}")
                            else:
                                print(f"⚠️ AI summary generation failed: {summary_result['error']}")
                        else:
                            print(f"⚠️ PDF text extraction failed: {pdf_result['error']}")
                            
                    else:
                        print(f"⚠️ Resume file not found: {resume_path}")
                        
                except Exception as e:
                    print(f"⚠️ Error processing PDF resume: {e}")
            
            # Fallback to profile-based summary if no PDF or PDF processing failed
            if not profile_data.get('summary'):
                ai_summary = generate_ai_summary(profile_data)
                if ai_summary:
                    profile_data['summary'] = ai_summary
                    # Save the updated profile with AI summary
                    try:
                        profile_manager.save_profile(profile_data)
                    except Exception as e:
                        print(f"Warning: Could not save AI summary to profile: {e}")
        
        return render_template('view_profile.html', profile=profile_data, profile_id=profile_id)
        
    except Exception as e:
        flash(f"Error loading profile: {e}", "error")
        return redirect(url_for('profiles'))


@app.route('/profiles/<profile_id>/delete', methods=['POST'])
def delete_profile(profile_id):
    """Delete a profile and its associated resume file"""
    try:
        uploads_dir = Path(__file__).parent.parent / "uploads" / "resumes"
        
        # Load profile to get resume filename before deleting
        profile_data = profile_manager.load_profile(profile_id)
        
        if profile_data:
            # Delete resume file if it exists
            resume_filename = profile_data.get('resume_file')
            if resume_filename:
                resume_path = uploads_dir / resume_filename
                if resume_path.exists():
                    try:
                        resume_path.unlink()
                        print(f"Deleted resume file: {resume_filename}")
                    except Exception as e:
                        print(f"Warning: Could not delete resume file: {e}")
            
            # Delete profile from database
            success = profile_manager.delete_profile(profile_id)
            if success:
                flash("Profile and associated files deleted successfully.", "success")
            else:
                flash("Error deleting profile from database.", "error")
        else:
            flash("Profile not found.", "error")
            
    except Exception as e:
        flash(f"Error deleting profile: {e}", "error")
    
    return redirect(url_for('profiles'))


@app.route('/profiles/<profile_id>/edit')
def edit_profile(profile_id):
    """Edit profile form"""
    try:
        # Use profile manager to load profile
        profile_data = profile_manager.load_profile(profile_id)
        
        if not profile_data:
            flash("Profile not found.", "error")
            return redirect(url_for('profiles'))
            
        # Get available skills for the form
        skills_data = []
        try:
            skills_file = Path(__file__).parent.parent / "data" / "skills_database.json"
            if skills_file.exists():
                with open(skills_file, 'r') as f:
                    skills_db = json.load(f)
                    skills_data = skills_db.get('skills', [])
        except Exception as e:
            print(f"Warning: Could not load skills database: {e}")
            
        return render_template('create_profile.html', profile=profile_data, skills=skills_data, edit_mode=True)
        
    except Exception as e:
        flash(f"Error loading profile for editing: {e}", "error")
        return redirect(url_for('profiles'))


@app.route('/profiles/<profile_id>/resume/download')
def download_resume(profile_id):
    """Download resume file from PostgreSQL storage"""
    try:
        # Get profile data from PostgreSQL
        profile_data = profile_manager.load_profile(profile_id)
        
        if not profile_data:
            flash("Profile not found.", "error")
            return redirect(url_for('profiles'))
            
        resume_filename = profile_data.get('resume_file')
        if not resume_filename:
            flash("No resume file found for this profile.", "error")
            return redirect(url_for('view_profile', profile_id=profile_id))
            
        # Check uploads directory
        uploads_dir = Path(__file__).parent.parent / "uploads" / "resumes" 
        resume_path = uploads_dir / resume_filename
        
        if not resume_path.exists():
            flash("Resume file not found on server.", "error")
            return redirect(url_for('view_profile', profile_id=profile_id))
            
        print(f"📥 Downloading resume: {resume_filename} for {profile_data.get('name', 'Unknown')}")
        
        from flask import send_file
        return send_file(
            resume_path,
            as_attachment=True,
            download_name=f"{profile_data.get('name', 'profile').replace(' ', '_')}_resume.pdf"
        )
        
    except Exception as e:
        print(f"Resume download error: {e}")
        flash(f"Error downloading resume: {e}", "error")
        return redirect(url_for('profiles'))


@app.route('/debug-test')
def debug_test():
    """Debug test page for job matching"""
    from flask import send_from_directory
    import os
    return send_from_directory(os.path.dirname(os.path.dirname(__file__)), 'debug_test.html')

@app.route('/match')
def match_page():
    """Job matching page"""
    try:
        # Use PostgreSQL storage to get profiles
        available_profiles = []
        profiles = profile_manager.list_profiles()
        
        for profile_data in profiles:
            # Format skills for display
            skill_names = []
            if profile_data.get('skills'):
                skill_names = [skill.get('skill_name', '') for skill in profile_data['skills'] if skill.get('skill_name')]
            
            available_profiles.append({
                'id': profile_data['user_id'],
                'name': profile_data.get('name', 'Unknown'),
                'title': profile_data.get('title', 'No title'),
                'location': profile_data.get('location', ''),
                'experience_level': profile_data.get('experience_level', 'entry'),
                'skills': skill_names
            })
        
        return render_template('match.html', profiles=available_profiles)
        
    except Exception as e:
        print(f"Error loading profiles for matching: {e}")
        return render_template('match.html', profiles=[])


@app.route('/api/match', methods=['POST'])
def api_match():
    """AI-Enhanced API endpoint for comprehensive job matching"""
    try:
        data = request.get_json()
        profile_id = data.get('profile_id')
        use_ai_matching = data.get('use_ai', True)  # Default to AI matching
        
        if not profile_id:
            return jsonify({'error': 'Profile ID is required'}), 400
        
        # Load profile from PostgreSQL
        profile_data = profile_manager.load_profile(profile_id)
        if not profile_data:
            return jsonify({'error': 'Profile not found'}), 404
        
        # Initialize results
        matching_info = {
            'profile_name': profile_data.get('name', 'Unknown'),
            'profile_title': profile_data.get('title', ''),
            'profile_location': profile_data.get('location', ''),
            'status': 'success',
            'matching_method': 'ai_enhanced' if use_ai_matching else 'traditional'
        }
        
        # 1. GATHER ALL AVAILABLE JOBS: Database + Vector Database
        all_available_jobs = []
        resume_text = ""
        
        try:
            print(f"🔍 Gathering jobs for AI analysis for {profile_data.get('name', 'user')}")
            
            # Get jobs from SQLite database - using global imports: Job already imported
            try:
                from database.db_config import db_config
            except ImportError:
                try:
                    from web.database.db_config import db_config
                except ImportError:
                    # Create minimal fallback
                    class MinimalDBConfig:
                        @contextlib.contextmanager
                        def session_scope(self):
                            yield None
                    db_config = MinimalDBConfig()
            
            with db_config.session_scope() as session:
                jobs = session.query(Job).filter(Job.is_active == True).limit(200).all()
                print(f"📊 Found {len(jobs)} active jobs in SQLite database")
                
                for job in jobs:
                    # Extract job categories and employment types
                    categories = job.job_category if job.job_category else []
                    employment_types = job.employment_type if job.employment_type else []
                    
                    all_available_jobs.append({
                        'job_id': job.job_id,
                        'job_title': job.title,
                        'company_name': job.company_name,
                        'category': categories[0] if categories else 'General',
                        'job_description': job.job_description,
                        'position_level': job.position_level,
                        'min_years_experience': job.min_years_experience,
                        'employment_type': employment_types,
                        'work_arrangement': job.work_arrangement,
                        'min_salary': job.min_salary,
                        'max_salary': job.max_salary,
                        'location': job.address,
                        'source': 'findsgjobs_api'
                    })
            
            # Get resume text for vector database integration
            if profile_data.get('resume_file'):
                uploads_dir = Path(__file__).parent.parent / "uploads" / "resumes"
                resume_path = uploads_dir / profile_data['resume_file']
                if resume_path.exists():
                    try:
                        import pdfplumber
                        with pdfplumber.open(str(resume_path)) as pdf:
                            resume_text = ""
                            for page in pdf.pages:
                                page_text = page.extract_text()
                                if page_text:
                                    resume_text += page_text + "\n"
                        print(f"📄 Extracted {len(resume_text)} characters from resume PDF")
                    except Exception as pdf_error:
                        print(f"⚠️ PDF extraction error: {pdf_error}")
            
            # Fallback to profile text if no resume
            if not resume_text:
                profile_parts = []
                if profile_data.get('title'):
                    profile_parts.append(f"Title: {profile_data['title']}")
                if profile_data.get('summary'):
                    profile_parts.append(f"Summary: {profile_data['summary']}")
                if profile_data.get('skills'):
                    skills_text = ", ".join([
                        skill.get('skill_name', skill) if isinstance(skill, dict) else str(skill)
                        for skill in profile_data['skills']
                    ])
                    profile_parts.append(f"Skills: {skills_text}")
                resume_text = "\n".join(profile_parts)
                
        except Exception as gather_error:
            print(f"⚠️ Error gathering jobs: {gather_error}")
            matching_info['gather_error'] = str(gather_error)
        
        # 2. AI ENHANCED MATCHING: Use AI to analyze comprehensive profile
        final_matches = []
        if use_ai_matching and all_available_jobs:
            try:
                print(f"🤖 Starting AI-enhanced job matching analysis...")
                ai_results = ai_enhanced_job_matching(
                    profile_data=profile_data, 
                    jobs_list=all_available_jobs[:100],  # Increased for better matches
                    vector_resume_text=resume_text
                )
                
                if ai_results and 'top_matches' in ai_results:
                    print(f"✅ AI found {len(ai_results['top_matches'])} enhanced matches")
                    
                    # Convert AI results to our format
                    for ai_match in ai_results['top_matches']:
                        job_id = ai_match['job_id']
                        # Find the original job data
                        original_job = next((job for job in all_available_jobs if job['job_id'] == job_id), None)
                        
                        if original_job:
                            # Extract salary information
                            salary_info = ""
                            if original_job.get('min_salary') and original_job.get('max_salary'):
                                salary_info = f"SGD {original_job['min_salary']:,} - {original_job['max_salary']:,}"
                            elif original_job.get('min_salary'):
                                salary_info = f"SGD {original_job['min_salary']:,}+"
                            elif original_job.get('max_salary'):
                                salary_info = f"Up to SGD {original_job['max_salary']:,}"
                            
                            final_matches.append({
                                'job_id': job_id,
                                'title': original_job['job_title'],
                                'company': original_job.get('company_name', 'Singapore Companies'),
                                'location': original_job.get('location', profile_data.get('location', 'Singapore')),
                                'category': original_job['category'],
                                'description': original_job['job_description'][:400] + '...' if len(original_job.get('job_description', '')) > 400 else original_job.get('job_description', ''),
                                'required_skills': [],  # Will be extracted from job description by AI
                                'position_level': original_job.get('position_level', ''),
                                'employment_type': original_job.get('employment_type', []),
                                'work_arrangement': original_job.get('work_arrangement', ''),
                                'salary_range': salary_info,
                                'match_score': ai_match['comprehensive_score'],
                                'match_percentage': ai_match['match_percentage'],
                                'skills_only_percentage': ai_match.get('skills_only_percentage', 0),
                                'matched_skills': ai_match.get('matched_skills', []),
                                'missing_skills': ai_match.get('skill_gaps', []),
                                'skills_matched_count': len(ai_match.get('matched_skills', [])),
                                'total_required_skills': len(ai_match.get('matched_skills', []) + ai_match.get('skill_gaps', [])),
                                'recommendation_reason': ai_match['recommendation_reason'],
                                'growth_opportunities': ai_match.get('growth_opportunities', ''),
                                'source': 'ai_enhanced',
                                'skill_match_score': ai_match.get('skill_match_score', 0),
                                'industry_match_score': ai_match.get('industry_match_score', 0),
                                'education_match_score': ai_match.get('education_match_score', 0),
                                'location_match_score': ai_match.get('location_match_score', 0),
                                'career_growth_score': ai_match.get('career_growth_score', 0)
                            })
                    
                    matching_info.update({
                        'ai_analysis_summary': ai_results.get('analysis_summary', ''),
                        'ai_matches_found': len(ai_results['top_matches'])
                    })
                else:
                    print("⚠️ AI matching returned no results, falling back to traditional matching")
                    
            except Exception as ai_error:
                print(f"⚠️ AI matching failed: {ai_error}")
                matching_info['ai_error'] = str(ai_error)
        
        # 3. FALLBACK: Traditional matching if AI fails or disabled
        if not final_matches and all_available_jobs:
            print("🔄 Using traditional skill-based matching as fallback")
            
            # Extract user skills
            user_skills = []
            if profile_data.get('skills'):
                for skill in profile_data['skills']:
                    if isinstance(skill, dict):
                        skill_name = skill.get('skill_name', '')
                    elif isinstance(skill, str):
                        skill_name = skill
                    else:
                        continue
                    if skill_name:
                        user_skills.append(skill_name.lower().strip())
            
            user_skills = list(set([skill for skill in user_skills if skill]))
            
            # Enhanced traditional skill matching with synonyms
            print(f"🔍 User skills for traditional matching: {user_skills}")
            
            # Skill synonyms for traditional matching
            traditional_synonyms = {
                'python': ['python', 'py', 'django', 'flask', 'pandas', 'numpy'],
                'sql': ['sql', 'mysql', 'postgresql', 'postgres', 'database', 'db'],
                'javascript': ['javascript', 'js', 'node', 'react', 'vue', 'angular'],
                'java': ['java', 'spring', 'springboot'],
                'it': ['it', 'information technology', 'tech', 'software', 'developer']
            }
            
            traditional_matches = []
            excluded_hr_jobs = []  # Track excluded HR jobs
            
            for job in all_available_jobs[:150]:  # Analyze more jobs
                # Extract skills from available text fields since job_skill_set doesn't exist
                job_keywords = (job.get('keywords') or '').lower()
                job_title = (job.get('job_title') or job.get('title') or '').lower()
                job_category = (job.get('category') or '').lower()
                job_description = (job.get('job_description') or '').lower()
                
                # Extract skills from job text using keyword matching
                job_skills_extracted = []
                all_job_text = f"{job_keywords} {job_title} {job_category} {job_description}"
                
                # Common technical skills to look for
                common_skills = [
                    'python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 'angular', 'vue',
                    'node', 'django', 'flask', 'spring', 'mysql', 'postgresql', 'mongodb', 'redis',
                    'aws', 'azure', 'docker', 'kubernetes', 'git', 'machine learning', 'ai',
                    'data analysis', 'excel', 'tableau', 'powerbi', 'analytics', 'business intelligence',
                    'project management', 'agile', 'scrum', 'leadership', 'communication',
                    'sales', 'marketing', 'customer service', 'finance', 'accounting'
                ]
                
                for skill in common_skills:
                    if skill in all_job_text:
                        job_skills_extracted.append(skill)
                
                job_skills_lower = job_skills_extracted
                
                # APPLY SAME EXCLUSION RULES AS ADVANCED MATCHING
                user_context = f"{' '.join(user_skills)}"
                job_context = f"{job_title} {job_category} {job_description}"
                
                # Check IT vs HR exclusion
                user_is_it = any(tech in user_context for tech in ['python', 'sql', 'developer', 'programmer', 'software', 'database', 'coding', 'tech', 'it', 'engineer'])
                job_is_hr = any(hr in job_context for hr in ['human resource', 'hr specialist', 'recruitment', 'people operations', 'talent acquisition', 'hr manager', 'hr coordinator'])
                
                if user_is_it and job_is_hr:
                    excluded_hr_jobs.append({
                        'job_id': job.get('job_id', 'unknown'),
                        'job_title': job.get('job_title', 'Unknown Title'),
                        'category': job.get('category', 'Unknown Category'),
                        'hr_keywords_found': [hr for hr in ['human resource', 'hr specialist', 'recruitment', 'people operations', 'talent acquisition', 'hr manager', 'hr coordinator'] if hr in job_context]
                    })
                    print(f"🚫 TRADITIONAL: Excluding HR job {job.get('job_id', 'unknown')}: {job_title}")
                    continue  # Skip HR jobs for IT professionals
                
                # Enhanced skill matching with synonyms
                matched_skills = []
                skill_relevance_scores = []
                
                for job_skill in job_skills_lower:
                    best_match_score = 0
                    best_match_skill = None
                    
                    for user_skill in user_skills:
                        # Direct match
                        if user_skill == job_skill:
                            best_match_score = 1.0
                            best_match_skill = job_skill
                        # Partial match  
                        elif user_skill in job_skill or job_skill in user_skill:
                            score = max(len(user_skill)/len(job_skill), len(job_skill)/len(user_skill))
                            if score > best_match_score:
                                best_match_score = score * 0.8
                                best_match_skill = job_skill
                        # Synonym match
                        else:
                            for category, synonyms in traditional_synonyms.items():
                                if user_skill in synonyms and job_skill in synonyms:
                                    if 0.7 > best_match_score:
                                        best_match_score = 0.7
                                        best_match_skill = job_skill
                    
                    if best_match_score > 0.3:  # Lower threshold for traditional
                        matched_skills.append(best_match_skill)
                        skill_relevance_scores.append(best_match_score)
                
                # Also check job title and category for skill matches
                for user_skill in user_skills:
                    if user_skill in job_title or user_skill in job_category:
                        if user_skill not in [m.lower() for m in matched_skills]:
                            matched_skills.append(f"title_match_{user_skill}")
                            skill_relevance_scores.append(0.6)
                
                # Calculate enhanced match percentage
                if skill_relevance_scores:
                    avg_relevance = sum(skill_relevance_scores) / len(skill_relevance_scores)
                    coverage = len(matched_skills) / max(len(job_skills_lower), 1) if job_skills_lower else 0.5
                    skill_match_score = (avg_relevance * 0.7 + coverage * 0.3)
                else:
                    skill_match_score = 0
                
                match_percentage = min(skill_match_score * 100, 95)
                
                if match_percentage >= 15:  # Lower threshold for better recall
                    traditional_matches.append({
                        'job_id': job['job_id'],
                        'title': job['job_title'],
                        'company': 'Singapore Companies',
                        'location': 'Singapore',
                        'category': job['category'],
                        'description': job['job_description'][:300] + '...' if len(job.get('job_description', '')) > 300 else job.get('job_description', ''),
                        'required_skills': job_skills_lower,
                        'match_score': skill_match_score,
                        'match_percentage': round(match_percentage, 1),
                        'matched_skills': matched_skills[:10],
                        'missing_skills': [s for s in job_skills_lower if s not in [m.lower() for m in matched_skills]][:10],
                        'skills_matched_count': len(matched_skills),
                        'total_required_skills': len(job_skills_lower),
                        'recommendation_reason': _create_simple_match_reason(match_percentage, len(matched_skills), job['category']),
                        'source': 'traditional',
                        'skill_match_score': skill_match_score,
                        'category_match_score': 0.2,
                        'user_skill_coverage': len(matched_skills) / max(len(user_skills), 1)
                    })
            
            # Sort and take top matches
            traditional_matches.sort(key=lambda x: x['match_percentage'], reverse=True)
            final_matches = traditional_matches[:5]  # Limit to top 5
            matching_info['fallback_matches'] = len(final_matches)
            
            # Log all excluded HR jobs for debugging
            if excluded_hr_jobs:
                print(f"\n🚫 EXCLUDED HR JOBS SUMMARY ({len(excluded_hr_jobs)} total):")
                print("=" * 60)
                for i, excluded_job in enumerate(excluded_hr_jobs, 1):
                    print(f"{i:2d}. Job ID: {excluded_job['job_id']}")
                    print(f"    Title: {excluded_job['job_title']}")
                    print(f"    Category: {excluded_job['category']}")
                    print(f"    HR Keywords: {', '.join(excluded_job['hr_keywords_found'])}")
                    print()
                print("=" * 60)
                print(f"✅ Successfully excluded {len(excluded_hr_jobs)} HR jobs from IT professional matching")
            else:
                print("ℹ️  No HR jobs found to exclude")
        
        # Ensure we have exactly 5 matches (or fewer if not available)
        final_matches = final_matches[:5]
        
        matching_info.update({
            'matching_type': 'ai_enhanced' if use_ai_matching else 'traditional',
            'total_matches': len(final_matches),
            'total_available_jobs': len(all_available_jobs),
            'top_match_score': final_matches[0]['match_percentage'] if final_matches else 0,
            'sources_used': list(set([match['source'] for match in final_matches])) if final_matches else [],
            'resume_text_available': bool(resume_text),
            'max_results': 5  # New: Always return max 5 results
        })
        
        return jsonify({
            **matching_info,
            'matches': final_matches
        })
        
    except Exception as e:
        print(f"Match API error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Matching failed: {str(e)}'}), 500


@app.route('/api/generate-job-application-pdf', methods=['POST'])
def generate_job_application_pdf():
    """Generate professional job application PDF"""
    try:
        data = request.get_json()
        profile_id = data.get('profile_id')
        job_id = data.get('job_id')
        job_title = data.get('job_title', 'Position')
        
        if not profile_id or not job_id:
            return jsonify({'error': 'Profile ID and Job ID are required'}), 400
        
        print(f"📄 Generating PDF application for profile {profile_id}, job {job_id}")
        
        # Load profile data
        profile_data = profile_manager.load_profile(profile_id)
        if not profile_data:
            return jsonify({'error': 'Profile not found'}), 404
        
        # Find job data from database
        job_data = None
        # Using global imports: Job already imported  
        try:
            try:
                from database.db_config import db_config
            except ImportError:
                try:
                    from web.database.db_config import db_config
                except ImportError:
                    # Create minimal fallback
                    class MinimalDBConfig:
                        @contextlib.contextmanager
                        def session_scope(self):
                            yield None
                    db_config = MinimalDBConfig()
            
            with db_config.session_scope() as session:
                job = session.query(Job).filter(Job.job_id == job_id).first()
                if job:
                    job_data = {
                        'job_id': job.job_id,
                        'title': job.title,  # Updated field name
                        'company': job.company_name or 'Singapore Companies',  # Use actual company name
                        'company_name': job.company_name,
                        'category': job.job_category if isinstance(job.job_category, str) else ', '.join(job.job_category) if job.job_category else 'General',
                        'job_category': job.job_category,  # Keep original for PDF generator
                        'description': job.job_description,
                        'keywords': job.keywords,  # Job requirements from keywords field
                        'position_level': job.position_level,
                        'min_years_experience': job.min_years_experience,
                        'min_education_level': job.min_education_level,
                        'no_of_vacancies': job.no_of_vacancies,
                        'min_salary': job.min_salary,
                        'max_salary': job.max_salary,
                        'salary_interval': job.salary_interval,
                        'currency': job.currency,
                        'employment_type': job.employment_type,
                        'work_arrangement': job.work_arrangement,
                        'nearest_mrt_station': job.nearest_mrt_station,
                        'timing_shift': job.timing_shift,
                        'address': job.address,
                        'postal_code': job.postal_code,
                        'website': job.website,
                        'company_description': job.company_description,
                        'required_skills': [],  # Will be extracted from keywords
                        'match_percentage': 0,  # Will be calculated based on actual matching
                        'matched_skills': [],  # Will be populated by AI analysis
                        'missing_skills': []   # Will be populated by AI analysis
                    }
                    
                    # Quick skill matching for PDF using keywords extraction
                    user_skills = []
                    if profile_data.get('skills'):
                        for skill in profile_data['skills']:
                            if isinstance(skill, dict):
                                skill_name = skill.get('skill_name', '')
                            else:
                                skill_name = str(skill)
                            if skill_name:
                                user_skills.append(skill_name.lower().strip())
                    
                    print(f"🔍 PDF DEBUG - User skills: {user_skills}")
                    print(f"🔍 PDF DEBUG - Job title: {job.title}")
                    print(f"🔍 PDF DEBUG - Job keywords: {job.keywords}")
                    print(f"🔍 PDF DEBUG - Job category: {job_data.get('category', 'N/A')}")
                    
                    # Extract skills from job keywords and description
                    job_text = f"{job.keywords or ''} {job.job_description or ''}".lower()
                    common_skills = [
                        'python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 'angular', 'vue',
                        'node', 'django', 'flask', 'spring', 'mysql', 'postgresql', 'mongodb', 'redis',
                        'aws', 'azure', 'docker', 'kubernetes', 'git', 'machine learning', 'ai',
                        'data analysis', 'excel', 'tableau', 'powerbi', 'analytics', 'business intelligence',
                        'project management', 'agile', 'scrum', 'leadership', 'communication',
                        'sales', 'marketing', 'customer service', 'finance', 'accounting'
                    ]
                    
                    job_skills = [skill for skill in common_skills if skill in job_text]
                    job_data['required_skills'] = job_skills
                    
                    # Enhanced skill matching using same algorithm as web interface
                    matched_skills = []
                    skill_relevance_scores = []
                    job_skills_lower = [skill.lower().strip() for skill in job_skills]
                    job_title = (job.title or '').lower()
                    job_category = (job_data.get('category', '') or '').lower()
                    
                    # Match skills with relevance scoring
                    for job_skill in job_skills_lower:
                        best_match_score = 0
                        best_match_skill = None
                        
                        for user_skill in user_skills:
                            # Exact match
                            if job_skill == user_skill:
                                if 1.0 > best_match_score:
                                    best_match_score = 1.0
                                    best_match_skill = job_skill
                            # Partial match (contains)
                            elif user_skill in job_skill or job_skill in user_skill:
                                if len(user_skill) >= 3 and len(job_skill) >= 3:  # Avoid short word false matches
                                    match_ratio = min(len(user_skill), len(job_skill)) / max(len(user_skill), len(job_skill))
                                    relevance_score = 0.5 + (match_ratio * 0.2)  # 0.5-0.7 range
                                    if relevance_score > best_match_score:
                                        best_match_score = relevance_score
                                        best_match_skill = job_skill
                        
                        if best_match_score > 0.3:  # Lower threshold for traditional
                            matched_skills.append(best_match_skill)
                            skill_relevance_scores.append(best_match_score)
                    
                    # Also check job title and category for skill matches
                    for user_skill in user_skills:
                        if user_skill in job_title or user_skill in job_category:
                            if user_skill not in [m.lower() for m in matched_skills]:
                                matched_skills.append(f"title_match_{user_skill}")
                                skill_relevance_scores.append(0.6)
                    
                    # Calculate comprehensive match percentage using same algorithm as web interface
                    if skill_relevance_scores:
                        avg_relevance = sum(skill_relevance_scores) / len(skill_relevance_scores)
                        coverage = len(matched_skills) / max(len(job_skills_lower), 1) if job_skills_lower else 0.5
                        skill_score = (avg_relevance * 0.7 + coverage * 0.3)
                    else:
                        skill_score = 0
                    
                    # Industry matching (same algorithm as web interface)
                    user_category = profile_data.get('desired_job_category', '').lower()
                    job_category_lower = job_data.get('category', '').lower()
                    
                    if user_category and user_category in job_category_lower:
                        industry_score = 0.95
                    elif any(keyword in job_category_lower for keyword in user_category.split()) if user_category else False:
                        industry_score = 0.7
                    else:
                        industry_score = 0.3  # Cross-industry penalty
                    
                    # Experience matching
                    user_experience = profile_data.get('experience_level', 'Entry')
                    
                    # Handle string parsing for min_years_experience
                    job_min_exp_raw = job.min_years_experience or 0
                    if isinstance(job_min_exp_raw, str):
                        # Extract number from strings like "1 Year", "2-3 years", etc.
                        import re
                        numbers = re.findall(r'\d+', job_min_exp_raw)
                        job_min_exp = int(numbers[0]) if numbers else 0
                    else:
                        job_min_exp = job_min_exp_raw or 0
                    
                    print(f"🔍 PDF DEBUG - User experience: {user_experience}")
                    print(f"🔍 PDF DEBUG - Job min exp: {job_min_exp} (raw: {job_min_exp_raw}, type: {type(job_min_exp_raw)})")
                    
                    exp_levels = {'Entry': 0, 'Mid': 3, 'Senior': 7, 'Executive': 12}
                    user_exp_years = exp_levels.get(user_experience, 0)
                    
                    if user_exp_years >= job_min_exp:
                        exp_score = 0.9
                    elif user_exp_years >= job_min_exp * 0.7:
                        exp_score = 0.7
                    else:
                        exp_score = 0.5
                    
                    # Location matching (simplified for PDF)
                    location_score = 1.0  # Default to perfect match
                    
                    # Career growth potential
                    job_description = (job.job_description or '').lower()
                    growth_indicators = ['lead', 'senior', 'manager', 'director', 'growth', 'development', 'advancement']
                    growth_score = 0.6 + (sum(1 for indicator in growth_indicators if indicator in job_description) * 0.1)
                    growth_score = min(growth_score, 1.0)
                    
                    # COMPREHENSIVE SCORE CALCULATION (same weights as web interface)
                    comprehensive_score = (
                        skill_score * 0.45 + 
                        industry_score * 0.35 + 
                        exp_score * 0.12 + 
                        location_score * 0.04 + 
                        growth_score * 0.04
                    )
                    
                    match_percentage = min(comprehensive_score * 100, 98)
                    
                    missing_skills = [skill for skill in job_skills_lower if skill not in [m.lower() for m in matched_skills]]
                    
                    job_data['matched_skills'] = matched_skills[:8]
                    job_data['missing_skills'] = missing_skills[:6]
                    job_data['match_percentage'] = round(match_percentage, 1)
                    
                    print(f"📊 PDF Job match analysis: {len(matched_skills)}/{len(job_skills_lower)} skills matched")
                    print(f"📊 PDF Skill score: {skill_score:.3f}, Industry score: {industry_score:.3f}")
                    print(f"📊 PDF Comprehensive score: {comprehensive_score:.3f}")
                    print(f"📊 PDF Final match percentage: {match_percentage:.1f}%")
                    print(f"📊 PDF Matched skills: {matched_skills[:5]}")
                    print(f"📊 PDF Job skills found: {job_skills[:5]}")
                else:
                    print(f"⚠️ Job {job_id} not found in database")
                    
        except Exception as db_error:
            print(f"⚠️ Database job lookup failed: {db_error}")
        
        # Fallback job data if not found in database
        if not job_data:
            job_data = {
                'job_id': job_id,
                'title': job_title,
                'company': 'Singapore Companies',
                'category': 'Technology',
                'description': 'Exciting opportunity to advance your career.',
                'required_skills': ['Communication', 'Problem Solving', 'Teamwork'],
                'match_percentage': 80,
                'matched_skills': [],
                'missing_skills': []
            }
            print("📋 Using fallback job data")
        
        # Import PDF generator
        try:
            try:
                from services.pdf_generator import get_pdf_generator
            except ImportError:
                from web.services.pdf_generator import get_pdf_generator
            pdf_gen = get_pdf_generator()
            
            # Generate PDF
            pdf_bytes = pdf_gen.generate_application_pdf(profile_data, job_data)
            
            # Create response
            from flask import Response
            response = Response(
                pdf_bytes,
                mimetype='application/pdf',
                headers={
                    'Content-Disposition': f'attachment; filename="SkillsMatch_Application_{job_title.replace(" ", "_")}.pdf"',
                    'Content-Length': str(len(pdf_bytes))
                }
            )
            
            print(f"✅ Generated PDF application ({len(pdf_bytes)} bytes)")
            return response
            
        except ImportError as import_error:
            print(f"❌ PDF generator import failed: {import_error}")
            return jsonify({
                'error': 'PDF generation not available. Please install reportlab: pip install reportlab'
            }), 500
        
    except Exception as e:
        print(f"PDF generation error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'PDF generation failed: {str(e)}'}), 500
        
    except Exception as e:
        print(f"Match API error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Matching failed: {str(e)}'}), 500


# Scraping functionality removed - using direct API access instead


@app.route('/api/fetch-jobs', methods=['POST'])
def fetch_jobs_from_api():
    """Fetch jobs from FindSGJobs API and update database"""
    try:
        import requests
        from datetime import datetime
        # Using global imports: Job already imported
        try:
            from database.db_config import db_config
        except ImportError:
            try:
                from web.database.db_config import db_config
            except ImportError:
                # Create minimal fallback
                class MinimalDBConfig:
                    @contextlib.contextmanager
                    def session_scope(self):
                        yield None
                db_config = MinimalDBConfig()
        
        print("🔄 Starting job fetch from FindSGJobs API...")
        
        # API endpoint
        api_url = "https://www.findsgjobs.com/apis/job/searchable"
        
        # Collect jobs from multiple pages to get 100 jobs total
        all_jobs_data = []
        pages_to_fetch = 5  # 20 jobs per page × 5 pages = 100 jobs
        
        for page in range(1, pages_to_fetch + 1):
            try:
                print(f"📄 Fetching page {page}...")
                params = {
                    'page': page,
                    'items per page': 20  # API default is 20 items per page
                }
                
                response = requests.get(api_url, params=params, timeout=30)
                response.raise_for_status()
                
                api_data = response.json()
                
                if api_data.get('data', {}).get('result'):
                    page_jobs = api_data['data']['result']
                    all_jobs_data.extend(page_jobs)
                    print(f"✅ Page {page}: Found {len(page_jobs)} jobs")
                else:
                    print(f"⚠️ Page {page}: No job data found")
                    break
                    
            except Exception as page_error:
                print(f"❌ Error fetching page {page}: {page_error}")
                break
        
        if not all_jobs_data:
            return jsonify({'success': False, 'error': 'No job data found in API response'}), 400
        
        jobs_data = all_jobs_data
        print(f"📊 Total jobs collected: {len(jobs_data)}")
        
        # Create database session
        session = db_config.get_session()
        
        try:
            # Clear existing jobs
            session.query(Job).delete()
            session.commit()
            print(f"🗑️ Cleared existing jobs from database")
            
            jobs_added = 0
            
            for job_item in jobs_data:
                try:
                    job_data = job_item.get('job', {})
                    company_data = job_item.get('company', {})
                    
                    if not job_data:
                        continue
                    
                    # Parse datetime fields
                    def parse_api_datetime(date_str):
                        if not date_str:
                            return None
                        try:
                            return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                        except:
                            return None
                    
                    # Extract employment types
                    employment_types = []
                    if job_data.get('EmploymentType'):
                        employment_types = [et.get('caption', '') for et in job_data['EmploymentType'] if et.get('caption')]
                    
                    # Extract job categories
                    job_categories = []
                    if job_data.get('JobCategory'):
                        job_categories = [jc.get('caption', '') for jc in job_data['JobCategory'] if jc.get('caption')]
                    
                    # Extract MRT stations
                    mrt_stations = []
                    if job_data.get('id_Job_NearestMRTStation'):
                        mrt_stations = [mrt.get('caption', '') for mrt in job_data['id_Job_NearestMRTStation'] if mrt.get('caption')]
                    
                    # Extract timing shifts
                    timing_shifts = []
                    if job_data.get('id_Job_TimingShift'):
                        timing_shifts = [ts.get('caption', '') for ts in job_data['id_Job_TimingShift'] if ts.get('caption')]
                    
                    # Extract location data
                    google_place = company_data.get('GooglePlace', {}) if company_data else {}
                    
                    # Create unique job_id if not provided
                    job_id = job_data.get('id') or f"job_{job_data.get('company_sid', '')}_{jobs_added}"
                    
                    # Create job object
                    job = Job(
                        job_id=str(job_id),
                        title=job_data.get('Title', ''),
                        company_name=company_data.get('CompanyName', '') if company_data else '',
                        company_sid=job_data.get('company_sid', ''),
                        activation_date=parse_api_datetime(job_data.get('activation_date')),
                        expiration_date=parse_api_datetime(job_data.get('expiration_date')),
                        updated_at=parse_api_datetime(job_data.get('updated_at')),
                        keywords=job_data.get('keywords', ''),
                        simple_keywords=job_data.get('simple_keywords', ''),
                        job_description=job_data.get('JobDescription', ''),
                        
                        # Job details
                        position_level=job_data.get('id_Job_PositionLevel', {}).get('caption', '') if job_data.get('id_Job_PositionLevel') else '',
                        min_years_experience=job_data.get('MinimumYearsofExperience', {}).get('caption', '') if job_data.get('MinimumYearsofExperience') else '',
                        min_education_level=job_data.get('MinimumEducationLevel', {}).get('caption', '') if job_data.get('MinimumEducationLevel') else '',
                        no_of_vacancies=job_data.get('id_Job_Noofvacancies'),
                        min_salary=job_data.get('id_Job_Salary'),
                        max_salary=job_data.get('id_Job_MaxSalary'),
                        salary_interval=job_data.get('id_Job_Interval', {}).get('caption', '') if job_data.get('id_Job_Interval') else '',
                        currency=job_data.get('id_Job_Currency', {}).get('caption', '') if job_data.get('id_Job_Currency') else '',
                        employment_type=employment_types,
                        work_arrangement=job_data.get('id_Job_WorkArrangement', {}).get('caption', '') if job_data.get('id_Job_WorkArrangement') else '',
                        nearest_mrt_station=mrt_stations,
                        timing_shift=timing_shifts,
                        job_category=job_categories,
                        
                        # Company details
                        contact_name=company_data.get('ContactName', '') if company_data else '',
                        website=company_data.get('Website', '') if company_data else '',
                        company_description=company_data.get('CompanyDescription', '') if company_data else '',
                        company_uen=company_data.get('id__CompanyUEN', '') if company_data else '',
                        company_country_code=company_data.get('id__Companycountrycode', '') if company_data else '',
                        
                        # Location
                        latitude=google_place.get('lat'),
                        longitude=google_place.get('lng'),
                        postal_code=google_place.get('postal'),
                        address=google_place.get('address'),
                        
                        # API metadata
                        job_source=job_item.get('job_source', ''),
                        api_fetched_at=datetime.utcnow(),
                        is_active=True
                    )
                    
                    session.add(job)
                    jobs_added += 1
                    
                except Exception as job_error:
                    print(f"⚠️ Error processing job: {job_error}")
                    continue
            
            # Commit all jobs
            session.commit()
            print(f"✅ Successfully added {jobs_added} jobs to database")
            
            return jsonify({
                'success': True,
                'jobs_count': jobs_added,
                'message': f'Successfully fetched and stored {jobs_added} jobs from FindSGJobs API'
            })
            
        except Exception as db_error:
            session.rollback()
            print(f"❌ Database error: {db_error}")
            return jsonify({'success': False, 'error': f'Database error: {str(db_error)}'}), 500
        
        finally:
            session.close()
            
    except requests.RequestException as req_error:
        print(f"❌ API request error: {req_error}")
        return jsonify({'success': False, 'error': f'API request failed: {str(req_error)}'}), 500
    
    except Exception as e:
        print(f"❌ Unexpected error in fetch_jobs_from_api: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Unexpected error: {str(e)}'}), 500


@app.route('/chat')
def chat_page():
    """AI chat interface page"""
    config = load_config()
    openai_api_key = config.get('openai_api_key') or os.environ.get('OPENAI_API_KEY')
    github_token = config.get('github_token') or os.environ.get('GITHUB_TOKEN')
    
    # Consider configured if either API key is available
    ai_configured = bool(openai_api_key or github_token)
    preferred_api = "OpenAI" if openai_api_key else "GitHub" if github_token else None
    
    return render_template('chat.html', 
                         github_configured=ai_configured,  # Keep same variable name for template compatibility
                         ai_configured=ai_configured,
                         preferred_api=preferred_api)


@socketio.on('send_chat_message')
def handle_chat_message(data):
    """Handle chat messages with AI career advisor"""
    try:
        message = data.get('message', '').strip()
        chat_history = data.get('chat_history', [])
        
        if not message:
            return
        
        print(f"🤖 DEBUG: Received chat message: {message}")
        
        # Send typing indicator
        emit('chat_response', {'type': 'thinking', 'message': 'AI is thinking...'})
        
        def chat_task():
            try:
                config = load_config()
                
                # Try OpenAI API key first (preferred), then GitHub token as fallback
                openai_api_key = config.get('openai_api_key') or os.environ.get('OPENAI_API_KEY')
                github_token = config.get('github_token') or os.environ.get('GITHUB_TOKEN')
                
                if not openai_api_key and not github_token:
                    # Demo mode - provide helpful responses without AI
                    demo_responses = {
                        "hello": "👋 Hello! I'm your AI Career Advisor (Demo Mode). I can help with career guidance, skills development, and job market insights in Singapore!",
                        "career": "🚀 For career development in Singapore, I recommend exploring SkillsFuture courses and identifying in-demand skills like data analytics, digital marketing, and software development.",
                        "skills": "💡 Popular skills in Singapore's job market include: Python programming, data analysis, digital marketing, project management, and cloud computing. What area interests you?",
                        "tech": "💻 Tech careers in Singapore are booming! Consider roles in software development, data science, cybersecurity, or cloud architecture. The government supports tech skill development through various initiatives.",
                        "time": f"🕐 Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Singapore Time. How can I help with your career today?",
                        "default": "🤖 I'm running in demo mode. To unlock full AI capabilities, please set your GITHUB_TOKEN environment variable. Meanwhile, I can provide basic career guidance! Try asking about 'skills', 'tech careers', or 'singapore jobs'."
                    }
                    
                    # Simple keyword matching for demo
                    message_lower = message.lower()
                    response = demo_responses["default"]
                    
                    for keyword, demo_response in demo_responses.items():
                        if keyword in message_lower:
                            response = demo_response
                            break
                    
                    response += "\n\n💡 **To enable full AI chat:** Set GITHUB_TOKEN environment variable with your GitHub Personal Access Token."
                    
                    socketio.emit('chat_response', {'type': 'ai', 'message': response})
                    return
                
                # Import OpenAI client
                from openai import OpenAI
                
                # Load skills data for context
                skills_context = ""
                try:
                    skills_db_path = Path("../data/skills_database.json")
                    if skills_db_path.exists():
                        with open(skills_db_path, 'r') as f:
                            skills_data = json.load(f)
                            # Get a sample of skills for context
                            sample_skills = list(skills_data.keys())[:20]
                            skills_context = f"Available skills in database: {', '.join(sample_skills)}"
                except Exception as e:
                    print(f"Could not load skills context: {e}")
                
                # Build conversation messages
                messages = [
                    {
                        "role": "system",
                        "content": f"""You are an AI Career Advisor for SkillsMatch.AI, specializing in Singapore's job market and skills development. 

Your expertise includes:
- Career guidance and planning in Singapore
- Skills development recommendations based on MySkillsFuture.gov.sg data
- Job market insights and trends
- Interview preparation and career transitions
- Professional development advice

{skills_context}

Guidelines:
- Provide practical, actionable advice
- Reference Singapore's job market and SkillsFuture initiatives when relevant
- Be encouraging and supportive
- Ask clarifying questions when needed
- Keep responses concise but comprehensive
- Use emojis occasionally to make conversations friendly

Current context: Singapore job market, SkillsFuture ecosystem, and career development."""
                    }
                ]
                
                # Add chat history (last 10 messages to manage context)
                for hist_msg in chat_history[-10:]:
                    if hist_msg.get('sender') == 'user':
                        messages.append({"role": "user", "content": hist_msg.get('message', '')})
                    elif hist_msg.get('sender') == 'ai':
                        messages.append({"role": "assistant", "content": hist_msg.get('message', '')})
                
                # Add current user message
                messages.append({"role": "user", "content": message})
                
                # Try multiple APIs in order of preference
                api_success = False
                last_error = None
                
                # First try OpenAI API if available
                if openai_api_key and not api_success:
                    # Try multiple OpenAI models in order of preference (gpt-4o-mini is working)
                    openai_models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
                    
                    for model_name in openai_models:
                        try:
                            print(f"🤖 DEBUG: Trying OpenAI API with model: {model_name}")
                            client = OpenAI(api_key=openai_key)
                            
                            # Test the API with the actual request
                            response = client.chat.completions.create(
                                model=model_name,
                                messages=messages,
                                temperature=0.7,
                                max_tokens=800,
                                top_p=0.95
                            )
                            
                            ai_message = response.choices[0].message.content
                            print(f"🤖 DEBUG: OpenAI API succeeded with {model_name} ({len(ai_message)} characters)")
                            socketio.emit('chat_response', {'type': 'ai', 'message': ai_message})
                            api_success = True
                            break  # Success, exit model loop
                            
                        except Exception as openai_error:
                            print(f"🤖 DEBUG: OpenAI model {model_name} failed: {openai_error}")
                            last_error = openai_error
                            # Continue to next model
                            continue
                
                # If OpenAI failed, try GitHub models
                if github_token and not api_success:
                    # Try multiple GitHub models in order of preference
                    github_models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
                    
                    for model_name in github_models:
                        try:
                            print(f"🤖 DEBUG: Trying GitHub models API with: {model_name}")
                            client = OpenAI(
                                base_url="https://models.inference.ai.azure.com",
                                api_key=github_token,
                            )
                            
                            # Test the API with the actual request
                            response = client.chat.completions.create(
                                model=model_name,
                                messages=messages,
                                temperature=0.7,
                                max_tokens=800,
                                top_p=0.95
                            )
                            
                            ai_message = response.choices[0].message.content
                            print(f"🤖 DEBUG: GitHub API succeeded with {model_name} ({len(ai_message)} characters)")
                            socketio.emit('chat_response', {'type': 'ai', 'message': ai_message})
                            api_success = True
                            break  # Success, exit model loop
                            
                        except Exception as github_error:
                            print(f"🤖 DEBUG: GitHub model {model_name} failed: {github_error}")
                            last_error = github_error
                            # Continue to next model
                            continue
                
                # If both APIs failed, fall back to demo mode
                if not api_success:
                    if last_error:
                        raise last_error
                    else:
                        raise Exception("No working API available")
                
            except Exception as e:
                print(f"🤖 DEBUG: Chat error: {str(e)}")
                import traceback
                traceback.print_exc()
                
                error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                if ("403" in str(e) and "no_access" in str(e)) or ("429" in str(e) and "insufficient_quota" in str(e)) or "quota" in str(e).lower():
                    # Model access issue or quota exceeded - fallback to enhanced demo mode
                    print("🤖 DEBUG: Model access/quota issue, falling back to enhanced demo mode")
                    
                    # Enhanced demo responses based on the user's question
                    demo_responses = {
                        "time": f"🕐 The current time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Singapore Time.",
                        "career": "🚀 **Career Development in Singapore:**\n\n• **Tech Sector**: High demand for software developers, data scientists, and cybersecurity experts\n• **Healthcare**: Growing opportunities in digital health and eldercare\n• **Finance**: FinTech and digital banking are expanding rapidly\n• **Logistics**: Smart port technologies and supply chain optimization\n\n💡 Consider exploring SkillsFuture courses to upskill in these areas!",
                        "skills": "💼 **In-Demand Skills in Singapore 2025:**\n\n**Technical Skills:**\n• Python, JavaScript, SQL programming\n• Data analysis and visualization\n• Cloud computing (AWS, Azure)\n• Cybersecurity fundamentals\n\n**Soft Skills:**\n• Digital marketing and e-commerce\n• Project management (Agile/Scrum)\n• Cross-cultural communication\n• Problem-solving and critical thinking",
                        "default": f"🤖 **Smart Career Guidance** (Enhanced Mode)\n\nI can help you with career questions about Singapore's job market! While I don't have full AI access right now, I can provide valuable insights about:\n\n• Tech career pathways\n• In-demand skills\n• SkillsFuture opportunities\n• Industry trends\n\n**Your question:** \"{message}\"\n\nFor this query, I'd recommend researching current market trends and considering upskilling through official Singapore resources like SkillsFuture.gov.sg and MyCareersFuture.gov.sg."
                    }
                    
                    # Smart keyword matching
                    message_lower = message.lower()
                    response = demo_responses["default"]
                    
                    if any(word in message_lower for word in ["time", "what time", "current time"]):
                        response = demo_responses["time"]
                    elif any(word in message_lower for word in ["career", "job", "work", "profession"]):
                        response = demo_responses["career"] 
                    elif any(word in message_lower for word in ["skill", "learn", "study", "course"]):
                        response = demo_responses["skills"]
                    
                    response += "\n\n⚠️ **Note:** Running in enhanced demo mode due to AI model access limitations."
                    
                    socketio.emit('chat_response', {'type': 'ai', 'message': response})
                    return
                    
                elif "401" in str(e):
                    error_msg = "🔑 Authentication failed. Please check your GitHub token."
                elif "429" in str(e) or "rate limit" in str(e).lower() or "quota" in str(e).lower():
                    error_msg = "💳 OpenAI quota exceeded. Using fallback demo mode above."
                elif "network" in str(e).lower() or "connection" in str(e).lower():
                    error_msg = "🌐 Network issue. Please check your connection and try again."
                
                socketio.emit('chat_response', {'type': 'error', 'message': error_msg})
        
        # Run in background thread
        socketio.start_background_task(chat_task)
        
    except Exception as e:
        print(f"🤖 DEBUG: Chat handler error: {str(e)}")
        emit('chat_response', {'type': 'error', 'message': f'Error processing message: {str(e)}'})


# Test AI endpoint for debugging
@app.route('/test-ai')
def test_ai():
    """Test AI connectivity for debugging"""
    try:
        from openai import OpenAI
        
        config = load_config()
        openai_key = config.get('openai_api_key') or os.environ.get('OPENAI_API_KEY')  
        github_token = config.get('github_token') or os.environ.get('GITHUB_TOKEN')
        
        results = {
            'openai_key_available': bool(openai_key),
            'github_token_available': bool(github_token),
            'tests': []
        }
        
        # Test OpenAI API
        if openai_key:
            try:
                client = OpenAI(api_key=openai_key)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello, just testing connectivity"}],
                    max_tokens=50
                )
                results['tests'].append({
                    'api': 'OpenAI',
                    'status': 'success', 
                    'model': 'gpt-3.5-turbo',
                    'response_length': len(response.choices[0].message.content)
                })
            except Exception as e:
                results['tests'].append({
                    'api': 'OpenAI',
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Test GitHub models
        if github_token:
            try:
                client = OpenAI(
                    base_url="https://models.inference.ai.azure.com",
                    api_key=github_token
                )
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Hello, just testing connectivity"}],
                    max_tokens=50
                )
                results['tests'].append({
                    'api': 'GitHub Models',
                    'status': 'success',
                    'model': 'gpt-4o-mini', 
                    'response_length': len(response.choices[0].message.content)
                })
            except Exception as e:
                results['tests'].append({
                    'api': 'GitHub Models',
                    'status': 'failed',
                    'error': str(e)
                })
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Health check endpoint for production monitoring
@app.route('/health')
def health_check():
    """Health check endpoint for load balancers and monitoring"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0',
        'skillmatch_available': SKILLMATCH_AVAILABLE,
        'scraper_available': SCRAPER_AVAILABLE
    })

# Error handlers for production
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors gracefully"""
    # Create default stats for error pages
    config = load_config()
    stats = {
        'skills_categories': 0,
        'total_opportunities': 0,
        'total_jobs': 100,
        'total_profiles': 3,
        'job_categories': {
            'F&B': 25,
            'Engineering': 15,
            'Sales / Retail': 12,
            'Social Services': 10,
            'Others': 38
        },
        'chart_data': {
            'categories': ['F&B', 'Engineering', 'Sales / Retail', 'Social Services', 'Others'],
            'values': [25, 15, 12, 10, 38]
        },
        'last_scrape': 'Never',
        'github_configured': bool(config.get('github_token')),
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    return render_template('index.html', stats=stats), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors gracefully"""
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Internal server error'}), 500
    
    # Create default stats for error pages
    config = load_config()
    stats = {
        'skills_categories': 0,
        'total_opportunities': 0,
        'last_scrape': 'Never',
        'github_configured': bool(config.get('github_token'))
    }
    return render_template('index.html', stats=stats), 500


if __name__ == '__main__':
    # Initialize data on startup
    initialize_data()
    
    # Production vs Development configuration
    if os.environ.get('FLASK_ENV') == 'production':
        print("🚀 Starting SkillsMatch.AI in PRODUCTION mode...")
        print("🌐 Available at: https://skillsmatch.ai")
        # In production, use a proper WSGI server like Gunicorn
        port = int(os.environ.get('PORT', 8000))
        socketio.run(app, debug=False, host='0.0.0.0', port=port)
    else:
        port = int(os.environ.get('PORT', os.environ.get('FLASK_PORT', 5004)))  # Default to 5004 for local development
        print("🚀 Starting SkillsMatch.AI Web Interface in DEVELOPMENT mode...")
        print(f"🌐 Open your browser to: http://localhost:{port}")
        print("💡 Using localhost for development - no domain setup needed!")
        print("� Direct API access configured for data integration")
        socketio.run(app, debug=True, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)