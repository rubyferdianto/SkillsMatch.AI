"""
SkillsMatch.AI Flask Web Application

A modern web interface for the SkillsMatch.AI career matching system
with real-time features, beautiful UI, and comprehensive functionality.
"""

import os
import sys
import json
import asyncio
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
    from services.simple_vector_service import get_vector_service
    VECTOR_SEARCH_AVAILABLE = True
    print("✅ Vector search service available")
except ImportError as e:
    print(f"⚠️ Vector search service not available: {e}")
    VECTOR_SEARCH_AVAILABLE = False

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import storage layer
try:
    from web.storage import profile_manager
except ImportError:
    # Fallback for when running from web directory
    from storage import profile_manager

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
    """Generate significantly improved mock AI response with advanced matching logic"""
    try:
        print("🎯 Generating ADVANCED mock AI matching response...")
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
            job_skills = job.get('job_skill_set', []) or []
            job_skills_lower = [skill.lower().strip() for skill in job_skills if skill and isinstance(skill, str)]
            job_category = (job.get('category') or '').lower()
            job_title = (job.get('job_title') or '').lower()
            job_description = (job.get('job_description') or '').lower()
            
            # 1. ADVANCED SKILL MATCHING (50% weight)
            matched_skills = []
            skill_relevance_scores = []
            
            for job_skill in job_skills_lower:
                best_match_score = 0
                best_match_skill = None
                
                for user_skill in user_skills:
                    # Exact match
                    if user_skill == job_skill:
                        best_match_score = 1.0
                        best_match_skill = job_skill
                    # Partial match
                    elif user_skill in job_skill or job_skill in user_skill:
                        score = max(len(user_skill) / len(job_skill), len(job_skill) / len(user_skill))
                        if score > best_match_score:
                            best_match_score = score * 0.8
                            best_match_skill = job_skill
                    # Synonym match
                    else:
                        for category, synonyms in skill_synonyms.items():
                            if user_skill in synonyms and job_skill in synonyms:
                                if 0.7 > best_match_score:
                                    best_match_score = 0.7
                                    best_match_skill = job_skill
                
                if best_match_score > 0.4:  # Only count meaningful matches
                    matched_skills.append(best_match_skill)
                    skill_relevance_scores.append(best_match_score)
            
            avg_skill_relevance = sum(skill_relevance_scores) / len(skill_relevance_scores) if skill_relevance_scores else 0
            skill_coverage = len(matched_skills) / max(len(job_skills_lower), 1) if job_skills_lower else 0
            skill_score = (avg_skill_relevance * 0.6 + skill_coverage * 0.4)
            
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
            
            # BALANCED QUALITY FILTER: Allow matches with either good skills or industry fit
            has_meaningful_skills = len(matched_skills) >= 1 and skill_score > 0.15
            has_industry_alignment = industry_score > 0.2  # Lowered from 0.3 to 0.2
            meets_threshold = match_percentage >= 20  # Lowered from 30 to 20
            
            # Allow match if has either good skills OR industry alignment (not both required)
            if has_meaningful_skills and (has_industry_alignment or skill_score > 0.4) and meets_threshold:
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
                
                scored_jobs.append({
                    'job_id': job['job_id'],
                    'match_percentage': round(match_percentage, 1),
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
        
        print(f"✅ Generated ADVANCED mock AI response with {len(top_matches)} diverse, high-quality matches")
        return mock_response
        
    except Exception as e:
        print(f"❌ Advanced mock AI response generation failed: {e}")
        return None

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
    """Main dashboard page"""
    config = load_config()
    
    # Get database statistics
    stats = {
        'skills_categories': 0,
        'total_opportunities': 0,
        'last_scrape': 'Never',
        'github_configured': bool(config.get('github_token'))
    }
    
    if data_loader:
        if hasattr(data_loader, 'skills_data') and data_loader.skills_data:
            stats['skills_categories'] = len(data_loader.skills_data.get('skills', {}))
        
        if hasattr(data_loader, 'opportunities_data') and data_loader.opportunities_data:
            stats['total_opportunities'] = len(data_loader.opportunities_data.get('opportunities', []))
    
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
    
    return render_template('index.html', stats=stats, profiles=profile_files)


@app.route('/profiles')
def profiles():
    """Profile management page"""
    try:
        # Use the profile manager for storage abstraction
        profiles_data = profile_manager.list_profiles()
        profile_files = []
        
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
            
        # Show storage info
        storage_info = profile_manager.get_storage_info()
        print(f"✅ Using {storage_info['type']} storage for profiles")
            
    except Exception as e:
        print(f"Error loading profiles: {e}")
        profile_files = []
    
    return render_template('profiles.html', profiles=profile_files)


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
            
            # Get jobs from PostgreSQL database
            from database.db_config import db_config
            from database.models import Job
            
            with db_config.session_scope() as session:
                jobs = session.query(Job).filter(Job.is_active == True).limit(200).all()
                print(f"📊 Found {len(jobs)} active jobs in PostgreSQL database")
                
                for job in jobs:
                    all_available_jobs.append({
                        'job_id': job.job_id,
                        'job_title': job.job_title,
                        'category': job.category,
                        'job_description': job.job_description,
                        'job_skill_set': job.job_skill_set or [],
                        'source': 'database'
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
                            final_matches.append({
                                'job_id': job_id,
                                'title': original_job['job_title'],
                                'company': 'Singapore Companies',
                                'location': profile_data.get('location', 'Singapore'),
                                'category': original_job['category'],
                                'description': original_job['job_description'][:400] + '...' if len(original_job.get('job_description', '')) > 400 else original_job.get('job_description', ''),
                                'required_skills': original_job['job_skill_set'],
                                'match_score': ai_match['comprehensive_score'],
                                'match_percentage': ai_match['match_percentage'],
                                'matched_skills': ai_match.get('matched_skills', []),
                                'missing_skills': ai_match.get('skill_gaps', []),
                                'skills_matched_count': len(ai_match.get('matched_skills', [])),
                                'total_required_skills': len(original_job['job_skill_set']),
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
                job_skills = job.get('job_skill_set', []) or []
                job_skills_lower = [skill.lower().strip() for skill in job_skills if skill and isinstance(skill, str)]
                job_title = (job.get('job_title') or '').lower()
                job_category = (job.get('category') or '').lower()
                job_description = (job.get('job_description') or '').lower()
                
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
                        'required_skills': job_skills,
                        'match_score': skill_match_score,
                        'match_percentage': round(match_percentage, 1),
                        'matched_skills': matched_skills[:10],
                        'missing_skills': [s for s in job_skills_lower if s not in [m.lower() for m in matched_skills]][:10],
                        'skills_matched_count': len(matched_skills),
                        'total_required_skills': len(job_skills_lower),
                        'recommendation_reason': f"Traditional skill matching: {len(matched_skills)}/{len(job_skills_lower)} skills matched",
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
        try:
            from database.db_config import db_config
            from database.models import Job
            
            with db_config.session_scope() as session:
                job = session.query(Job).filter(Job.job_id == job_id).first()
                if job:
                    job_data = {
                        'job_id': job.job_id,
                        'title': job.job_title,
                        'company': 'Singapore Companies',  # Default since we don't have specific companies
                        'category': job.category,
                        'description': job.job_description,
                        'required_skills': job.job_skill_set or [],
                        'match_percentage': 85,  # Default high match since user selected this job
                        'matched_skills': [],  # Will be populated by AI analysis
                        'missing_skills': []   # Will be populated by AI analysis
                    }
                    
                    # Quick skill matching for PDF
                    user_skills = []
                    if profile_data.get('skills'):
                        for skill in profile_data['skills']:
                            if isinstance(skill, dict):
                                skill_name = skill.get('skill_name', '')
                            else:
                                skill_name = str(skill)
                            if skill_name:
                                user_skills.append(skill_name.lower().strip())
                    
                    job_skills = [skill.lower().strip() for skill in (job.job_skill_set or []) if skill]
                    
                    # Find matches
                    matched_skills = []
                    for job_skill in job_skills:
                        for user_skill in user_skills:
                            if user_skill in job_skill or job_skill in user_skill:
                                matched_skills.append(job_skill)
                                break
                    
                    missing_skills = [skill for skill in job_skills if skill not in [m.lower() for m in matched_skills]]
                    
                    job_data['matched_skills'] = matched_skills[:8]
                    job_data['missing_skills'] = missing_skills[:6]
                    job_data['match_percentage'] = min(100, (len(matched_skills) / max(len(job_skills), 1)) * 100) if job_skills else 80
                    
                    print(f"📊 Job match analysis: {len(matched_skills)}/{len(job_skills)} skills matched")
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
            from services.pdf_generator import get_pdf_generator
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
        'last_scrape': 'Never',
        'github_configured': bool(config.get('github_token'))
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
        port = int(os.environ.get('PORT', 5003))  # Default to 5003 for local development
        print("🚀 Starting SkillsMatch.AI Web Interface in DEVELOPMENT mode...")
        print(f"🌐 Open your browser to: http://localhost:{port}")
        print("💡 Using localhost for development - no domain setup needed!")
        print("� Direct API access configured for data integration")
        socketio.run(app, debug=True, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)