"""
Professional PDF Application Generator for SkillsMatch.AI
Creates AI-powered job application PDFs with SkillsMatch.AI branding
"""

import io
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.colors import HexColor, black, white, gray
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.platypus.flowables import HRFlowable
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class SkillsMatchPDFGenerator:
    """Professional PDF generator for job applications"""
    
    def __init__(self):
        self.colors = {
            'primary': HexColor('#2563eb'),      # Blue
            'secondary': HexColor('#059669'),    # Green  
            'accent': HexColor('#7c3aed'),       # Purple
            'text': HexColor('#1f2937'),         # Dark gray
            'light_gray': HexColor('#f9fafb'),   # Light gray
            'border': HexColor('#e5e7eb')        # Border gray
        }
        
        # Initialize OpenAI if available
        self.openai_client = None
        openai_key = os.environ.get('OPENAI_API_KEY')
        if OPENAI_AVAILABLE and openai_key:
            self.openai_client = OpenAI(api_key=openai_key)
    
    def create_styles(self):
        """Create custom styles for the PDF"""
        styles = getSampleStyleSheet()
        
        # Title style
        styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            textColor=self.colors['primary'],
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Header style
        styles.add(ParagraphStyle(
            name='CustomHeader',
            parent=styles['Heading1'],
            fontSize=16,
            textColor=self.colors['primary'],
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold'
        ))
        
        # Subheader style
        styles.add(ParagraphStyle(
            name='CustomSubHeader',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=self.colors['secondary'],
            spaceAfter=8,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))
        
        # Body text style
        styles.add(ParagraphStyle(
            name='CustomBody',
            parent=styles['Normal'],
            fontSize=11,
            textColor=self.colors['text'],
            spaceAfter=8,
            alignment=TA_JUSTIFY,
            fontName='Helvetica'
        ))
        
        # Small text style
        styles.add(ParagraphStyle(
            name='CustomSmall',
            parent=styles['Normal'],
            fontSize=9,
            textColor=gray,
            spaceAfter=4,
            fontName='Helvetica'
        ))
        
        return styles
    
    def generate_ai_cover_letter(self, profile_data: Dict[str, Any], job_data: Dict[str, Any]) -> str:
        """Generate AI-powered cover letter"""
        if not self.openai_client:
            return self._generate_template_cover_letter(profile_data, job_data)
        
        try:
            # Build context for AI
            user_name = profile_data.get('name', 'Professional')
            job_title = job_data.get('title', 'Position')
            company = job_data.get('company', 'Your Company')
            
            # Extract skills
            skills = []
            if profile_data.get('skills'):
                for skill in profile_data['skills']:
                    if isinstance(skill, dict):
                        skill_name = skill.get('skill_name', '')
                    else:
                        skill_name = str(skill)
                    if skill_name:
                        skills.append(skill_name)
            
            # Extract experience
            experience = profile_data.get('work_experience', [])
            education = profile_data.get('education', [])
            
            prompt = f"""Write a professional, compelling cover letter for this job application:

APPLICANT: {user_name}
JOB TITLE: {job_title}
COMPANY: {company}
LOCATION: {profile_data.get('location', 'Singapore')}

APPLICANT PROFILE:
- Current Title: {profile_data.get('title', 'Professional')}
- Experience Level: {profile_data.get('experience_level', 'Experienced')}
- Key Skills: {', '.join(skills[:8])}
- Work Experience: {len(experience)} positions
- Education: {len(education)} qualifications
- Professional Summary: {profile_data.get('summary', 'Dedicated professional')}

JOB REQUIREMENTS:
- Required Skills: {', '.join(job_data.get('required_skills', [])[:6])}
- Job Description: {job_data.get('description', '')[:200]}...

INSTRUCTIONS:
1. Write a professional, engaging cover letter (300-400 words)
2. Highlight relevant skills and experience matches
3. Show enthusiasm for the specific role and company
4. Include specific achievements or capabilities
5. End with a strong call to action
6. Use professional but personable tone
7. Avoid generic phrases - make it specific to this application

Format as plain text paragraphs."""

            response = self.openai_client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {"role": "system", "content": "You are an expert career writer who creates compelling, personalized cover letters that get results. Write engaging, specific cover letters that showcase the applicant's unique value."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.7
            )
            
            cover_letter = response.choices[0].message.content.strip()
            print(f"✅ Generated AI cover letter ({len(cover_letter)} characters)")
            return cover_letter
            
        except Exception as e:
            print(f"⚠️ AI cover letter generation failed: {e}")
            return self._generate_template_cover_letter(profile_data, job_data)
    
    def _generate_template_cover_letter(self, profile_data: Dict[str, Any], job_data: Dict[str, Any]) -> str:
        """Generate template cover letter when AI is not available"""
        user_name = profile_data.get('name', 'Professional')
        job_title = job_data.get('title', 'Position')
        company = job_data.get('company', 'Your Company')
        
        return f"""Dear Hiring Manager at {company},

I am writing to express my strong interest in the {job_title} position. With my background in {profile_data.get('title', 'technology')} and proven experience in {profile_data.get('experience_level', 'professional development')}, I am confident I would be a valuable addition to your team.

Throughout my career, I have developed expertise in key areas that align with your requirements. My technical skills include {', '.join([skill.get('skill_name', skill) if isinstance(skill, dict) else str(skill) for skill in (profile_data.get('skills', [])[:4])])}, which directly match the requirements for this role.

I am particularly drawn to this opportunity because it combines my passion for {profile_data.get('title', 'technology')} with my desire to contribute to a forward-thinking organization like {company}. My experience has taught me the importance of {profile_data.get('summary', 'continuous learning and professional excellence')}.

I would welcome the opportunity to discuss how my skills and enthusiasm can contribute to your team's success. Thank you for considering my application.

Sincerely,
{user_name}"""
    
    def generate_application_pdf(self, profile_data: Dict[str, Any], job_data: Dict[str, Any]) -> bytes:
        """Generate complete job application PDF"""
        if not REPORTLAB_AVAILABLE:
            raise ImportError("ReportLab is required for PDF generation. Install with: pip install reportlab")
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        
        # Create styles
        styles = self.create_styles()
        story = []
        
        # Header with SkillsMatch.AI branding
        story.append(Paragraph("SkillsMatch.AI", styles['CustomTitle']))
        story.append(Paragraph("Professional Job Application", styles['CustomSmall']))
        story.append(Spacer(1, 12))
        
        # Horizontal line
        story.append(HRFlowable(width="100%", thickness=2, color=self.colors['primary']))
        story.append(Spacer(1, 20))
        
        # Applicant Information
        story.append(Paragraph("APPLICANT INFORMATION", styles['CustomHeader']))
        
        applicant_info = [
            ['Name:', profile_data.get('name', 'N/A')],
            ['Title:', profile_data.get('title', 'N/A')],
            ['Location:', profile_data.get('location', 'N/A')],
            ['Experience Level:', profile_data.get('experience_level', 'N/A').title()],
            ['Application Date:', datetime.now().strftime('%B %d, %Y')]
        ]
        
        applicant_table = Table(applicant_info, colWidths=[2*inch, 4*inch])
        applicant_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), self.colors['secondary']),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(applicant_table)
        story.append(Spacer(1, 20))
        
        # Job Information
        story.append(Paragraph("TARGET POSITION", styles['CustomHeader']))
        
        job_info = [
            ['Position:', job_data.get('title', 'N/A')],
            ['Company:', job_data.get('company', 'Various Companies')],
            ['Category:', job_data.get('category', 'General')],
            ['Match Score:', f"{job_data.get('match_percentage', 0)}%"]
        ]
        
        job_table = Table(job_info, colWidths=[2*inch, 4*inch])
        job_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), self.colors['secondary']),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(job_table)
        story.append(Spacer(1, 20))
        
        # Cover Letter
        story.append(Paragraph("COVER LETTER", styles['CustomHeader']))
        cover_letter = self.generate_ai_cover_letter(profile_data, job_data)
        
        # Split cover letter into paragraphs
        paragraphs = cover_letter.strip().split('\n\n')
        for para in paragraphs:
            if para.strip():
                story.append(Paragraph(para.strip(), styles['CustomBody']))
                story.append(Spacer(1, 8))
        
        story.append(Spacer(1, 20))
        
        # Skills Summary
        story.append(Paragraph("SKILLS SUMMARY", styles['CustomHeader']))
        
        skills = []
        if profile_data.get('skills'):
            for skill in profile_data['skills']:
                if isinstance(skill, dict):
                    skill_name = skill.get('skill_name', '')
                else:
                    skill_name = str(skill)
                if skill_name:
                    skills.append(skill_name)
        
        if skills:
            # Create skills table (2 columns)
            skills_data = []
            for i in range(0, len(skills), 2):
                row = [skills[i] if i < len(skills) else '', 
                       skills[i+1] if i+1 < len(skills) else '']
                skills_data.append(row)
            
            skills_table = Table(skills_data, colWidths=[3*inch, 3*inch])
            skills_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('TEXTCOLOR', (0, 0), (-1, -1), self.colors['text']),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('BACKGROUND', (0, 0), (-1, -1), self.colors['light_gray']),
                ('GRID', (0, 0), (-1, -1), 1, self.colors['border']),
            ]))
            story.append(skills_table)
        else:
            story.append(Paragraph("Skills information not available", styles['CustomBody']))
        
        story.append(Spacer(1, 20))
        
        # Match Analysis (if available)
        if job_data.get('matched_skills') or job_data.get('missing_skills'):
            story.append(Paragraph("MATCH ANALYSIS", styles['CustomHeader']))
            
            if job_data.get('matched_skills'):
                story.append(Paragraph("Matching Skills:", styles['CustomSubHeader']))
                matched_text = ', '.join(job_data['matched_skills'][:10])
                story.append(Paragraph(matched_text, styles['CustomBody']))
                story.append(Spacer(1, 8))
            
            if job_data.get('missing_skills'):
                story.append(Paragraph("Skills to Develop:", styles['CustomSubHeader']))
                missing_text = ', '.join(job_data['missing_skills'][:8])
                story.append(Paragraph(missing_text, styles['CustomBody']))
                story.append(Spacer(1, 8))
        
        # Footer
        story.append(Spacer(1, 30))
        story.append(HRFlowable(width="100%", thickness=1, color=self.colors['border']))
        story.append(Spacer(1, 8))
        story.append(Paragraph("Generated by SkillsMatch.AI - Your AI-Powered Career Partner", styles['CustomSmall']))
        story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['CustomSmall']))
        
        # Build PDF
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes

# Global instance
pdf_generator = SkillsMatchPDFGenerator()

def get_pdf_generator() -> SkillsMatchPDFGenerator:
    """Get the global PDF generator instance"""
    return pdf_generator