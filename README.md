# SkillMatch.AI 🎯

**Intelligent Career and Skill Matching System**

SkillMatch.AI is a comprehensive AI-powered platform that matches users with jobs, projects, and learning opportunities based on their skills, experience, and preferences. Built with Microsoft Agent Framework and powered by GitHub models.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![AI Framework](https://img.shields.io/badge/AI-Microsoft%20Agent%20Framework-purple.svg)

## ✨ Features

### Core Capabilities
- 🤖 **AI-Powered Matching**: Uses advanced language models for intelligent skill and opportunity matching
- 🎯 **Multi-Type Opportunities**: Supports jobs, projects, internships, and learning opportunities
- 📊 **Comprehensive Scoring**: Detailed match scores with skill gaps and strengths analysis
- 🧠 **Smart Skill Analysis**: Identifies related skills and provides learning recommendations
- 💬 **Interactive Chat**: Natural language interface for career guidance
- 📈 **Career Planning**: Skill gap analysis and personalized learning paths
- 🔧 **Extensible**: Modular design for easy customization and extension

### Web Interface
- 🌐 **Modern Web UI**: Beautiful, responsive interface built with Bootstrap 5
- ⚡ **Real-time Updates**: Live progress tracking and instant feedback
- 📱 **Mobile-Friendly**: Optimized for all devices and screen sizes
- 🎨 **Professional Design**: Clean, intuitive interface with smooth animations
- 🔄 **Interactive Experience**: Dynamic forms, live validation, and instant feedback

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- GitHub Personal Access Token (for model access)
- Conda environment manager (recommended)
- **Important**: Always use the `smai` conda environment for this project

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/rubyferdianto/SkillMatch.AI.git
   cd SkillMatch.AI
   ```

2. **Create and activate the smai conda environment**
   ```bash
   conda create -n smai python=3.11
   conda activate smai
   ```
   
   ⚠️ **Important**: Always activate the `smai` environment before running any commands:
   ```bash
   conda activate smai
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   > **Note**: The `--pre` flag is required for the Microsoft Agent Framework as it's in preview.

4. **Set up configuration**
   ```bash
   python skillmatch.py setup
   ```
   
   You'll need to provide:
   - Your GitHub Personal Access Token ([Get one here](https://github.com/settings/tokens))
   - Preferred AI model (default: `openai/gpt-4.1-mini`)

### First Run

Try the sample profile:
```bash
python skillmatch.py match --profile profiles/john_developer.json
```

## 🌐 Web Interface

SkillMatch.AI now includes a modern, responsive web interface for better user experience!

### Quick Start (Web Interface)
To run the web interface:

```bash
# Make sure you're in the smai environment
conda activate smai

# Install web dependencies
pip install flask flask-cors flask-socketio eventlet

# Run the web application
python web/app.py
```

Visit `http://localhost:5003` to access the web interface.

⚠️ **Important**: Always ensure the `smai` conda environment is activated before running the web application.

### Web Features
- **📊 Interactive Dashboard** - Overview of your career data and system status
- **👤 Comprehensive Profile Management** - Create, edit, view, and delete career profiles with full CRUD operations
- **💼 Professional Profile Builder** - Complete forms with work experience, education, skills, and salary preferences
- **📄 Resume Management** - Upload, download, and replace PDF resumes with secure file handling
- **📈 Profile Analytics** - View statistics on total profiles, skills distribution, and experience levels
- **🎯 Smart Matching** - Find opportunities with real-time progress tracking
- **🤖 AI Career Chat** - Get personalized career advice with conversational interface
- **📱 Responsive Design** - Works perfectly on desktop, tablet, and mobile devices

### Web Interface Highlights
- **Modern UI**: Bootstrap 5 design with custom SkillMatch.AI styling and smooth animations
- **Real-time Updates**: Live progress tracking for all operations
- **Smart Forms**: Interactive forms with validation, suggestion chips, and dynamic content
- **Professional UX**: Intuitive navigation, flash messages, modals, and responsive layouts
- **Complete Functionality**: All CLI features available through beautiful web interface

### Files Structure
```
web/
├── app.py              # Main web application
├── templates/          # HTML templates
│   ├── base.html       # Base template with navigation
│   ├── index.html      # Dashboard page
│   ├── profiles.html   # Profile listing with analytics and management
│   ├── create_profile.html  # Profile creation/editing form with all fields
│   ├── view_profile.html    # Detailed profile view with all sections
│   ├── match.html      # Matching interface
│   └── chat.html       # AI chat interface
├── static/             # Static assets (CSS, JS)
│   ├── css/
│   │   └── style.css   # Professional styling with enterprise colors
│   └── js/
├── uploads/            # File storage
│   └── resumes/        # PDF resume files
└── requirements.txt    # Web dependencies

profiles/               # Profile data storage
├── *.json             # Individual profile files
└── resumes/           # Resume file storage
```

### Local Development Focus
This project is currently optimized for local development. All features work on localhost without requiring production deployment.

## � Career Profiles System

SkillMatch.AI features a comprehensive career profiles management system that allows users to create, manage, and optimize their professional profiles for better job matching.

### 🌟 Profile Management Features

#### **Complete Profile Creation**
- **📝 Basic Information**: Name, title, location, and professional bio
- **🧠 Skills Management**: Add technical and soft skills with proficiency levels
- **💼 Work Experience**: Current and previous positions with detailed descriptions
- **🎓 Education**: Academic qualifications, institutions, and graduation years
- **💰 Salary Preferences**: Expected salary range and compensation preferences
- **🏢 Work Preferences**: Remote work options, employment types, and industry preferences
- **📄 Resume Upload**: PDF resume storage with download and replacement capabilities

#### **Professional Profile Analytics**
- **📊 Statistics Dashboard**: Track total profiles, skills, industries, and experience levels
- **📈 Automatic Experience Level Classification**: 
  - **Entry Level**: 0-2 years of experience (Fresh graduates, junior roles)
  - **Mid Level**: 3-5 years of experience (Experienced professionals)
  - **Senior Level**: 6+ years of experience (Senior roles, team leads)
- **🎯 Smart Categorization**: Automatic skill categorization (programming, database, soft skills, etc.)
- **📈 Real-time Analytics**: Live statistics updating as profiles are created/modified
- **🔍 Data Insights**: Skills distribution, industry coverage, and experience diversity

#### **Advanced Profile Operations**
- **✏️ Full Edit Functionality**: Modify all profile aspects with pre-populated forms
- **👁️ Comprehensive View**: Detailed profile view showing all information sections
- **🗑️ Secure Deletion**: Complete profile and file removal with confirmation dialogs
- **📱 Responsive Design**: Optimized for all devices and screen sizes

### 🔧 Profile Management Interface

#### **Profile Cards Display**
Each profile is displayed in a professional card format showing:
- **Profile avatar** with gradient background
- **Name and creation date**
- **Experience level badge** (automatically calculated)
- **Skills overview** with skill count and top skills display
- **Industry preferences** and location information
- **Resume status** with download option when available
- **Action menu** with view, edit, and delete options

#### **Enhanced Forms**
- **Smart Form Validation**: Real-time validation with helpful error messages
- **Dynamic Field Management**: Add/remove skills and industries dynamically
- **File Upload Handling**: Secure PDF resume upload with file validation
- **Edit Mode Support**: Pre-populate all fields when editing existing profiles
- **Professional Styling**: Clean, modern interface with enterprise color scheme

#### **Profile Data Structure**
```json
{
  "name": "Professional Name",
  "email": "email@example.com",
  "title": "Job Title",
  "location": "City, Country",
  "experience_level": "mid",
  "bio": "Professional summary",
  "skills": [
    {
      "skill_id": "python",
      "skill_name": "Python",
      "category": "programming",
      "level": "advanced",
      "years_experience": 3
    }
  ],
  "work_experience": [
    {
      "position": "Software Developer",
      "company": "Tech Company",
      "years": 3,
      "description": "Job responsibilities",
      "employment_status": "employed"
    }
  ],
  "education": [
    {
      "degree": "Bachelor",
      "institution": "University Name",
      "field_of_study": "Computer Science",
      "graduation_year": 2020
    }
  ],
  "preferences": {
    "work_types": ["full_time"],
    "salary_min": 50000,
    "salary_max": 80000,
    "remote_preference": "hybrid",
    "locations": ["Singapore"]
  },
  "career_goals": [
    "Become a senior developer",
    "Learn new technologies"
  ],
  "resume_file": "resume.pdf"
}
```

### 🚀 Getting Started with Profiles

1. **Access Profile Management**
   ```
   http://localhost:5003/profiles
   ```

2. **Create Your First Profile**
   - Click "Create New Profile"
   - Fill in basic information (name, location, etc.)
   - Add your skills with proficiency levels
   - Include work experience and education
   - Set salary and work preferences
   - Upload your resume (PDF format)
   - Save and view your profile

3. **Manage Existing Profiles**
   - **View**: See complete profile details
   - **Edit**: Modify any profile information
   - **Download Resume**: Get PDF resume file
   - **Delete**: Remove profile and associated files

### 💡 Profile Best Practices

- **Complete All Sections**: Fill in work experience, education, and preferences for better matching
- **Keep Skills Current**: Regularly update skills and proficiency levels
- **Upload Resume**: PDF resume improves matching accuracy
- **Set Realistic Expectations**: Use appropriate salary ranges and experience levels
- **Update Regularly**: Keep profile information current for best results

## �📖 Usage

### Command Line Interface

SkillMatch.AI provides several CLI commands:

#### 🔍 Find Matching Opportunities
```bash
# Use existing profile
python skillmatch.py match --profile profiles/john_developer.json

# Interactive profile creation
python skillmatch.py match --interactive
```

#### 📚 Analyze Skill Gaps
```bash
# General skill gap analysis
python skillmatch.py gaps --profile profiles/john_developer.json

# Target-specific analysis
python skillmatch.py gaps --profile profiles/john_developer.json --target-role "Senior Data Scientist"
```

#### 🎓 Find Learning Opportunities
```bash
# Search for specific skills
python skillmatch.py learn --skills "machine learning, deep learning" --difficulty intermediate --max-cost 500

# Free courses only
python skillmatch.py learn --skills "python, sql" --max-cost 0
```

#### 💬 Interactive Chat
```bash
python skillmatch.py chat
```




### Python API

You can also use SkillMatch.AI programmatically:

```python
import asyncio
from skillmatch import SkillMatchAgent, UserProfile

async def main():
    # Initialize the agent
    agent = SkillMatchAgent(
        github_token="your_github_token",
        model_id="openai/gpt-4.1-mini"
    )
    await agent.initialize()
    
    # Load user profile
    with open("profiles/john_developer.json", "r") as f:
        profile_data = json.load(f)
    user_profile = UserProfile(**profile_data)
    
    # Find matches
    matches = await agent.find_matching_opportunities(
        user_profile_json=json.dumps(profile_data, default=str)
    )
    
    print(matches)
    await agent.close()

# Run the async function
asyncio.run(main())
```

## 🚀 Quick Command Reference

### Core SkillMatch.AI Commands
```bash
# Setup (first time only)
python skillmatch.py setup

# Find job matches
python skillmatch.py match --profile profiles/john_developer.json

# Analyze skill gaps  
python skillmatch.py gaps --profile profiles/john_developer.json

# Find learning opportunities
python skillmatch.py learn --skills "python, machine learning"

# Chat with AI career advisor
python skillmatch.py chat
```

### File Structure After Setup
```
SkillsMatch.AI/
├── data/
│   ├── skills_database.json
│   └── opportunities_database.json
├── profiles/                   # Your career profiles
└── config/                     # AI model configuration
```

## 🏗️ Architecture

### Core Components

```
SkillMatch.AI/
├── src/skillmatch/
│   ├── models/              # Data models (Pydantic)
│   │   ├── user_profile.py  # User profiles and skills
│   │   └── opportunities.py # Jobs, projects, learning
│   ├── agents/              # AI agents
│   │   └── skill_match_agent.py  # Main AI agent
│   ├── utils/               # Utilities
│   │   ├── data_loader.py   # Data loading utilities
│   │   └── skill_matcher.py # Matching algorithms
│   └── cli.py              # Command line interface
├── data/                   # Data files
│   ├── skills_database.json
│   └── opportunities_database.json
├── config/                 # Configuration
├── profiles/              # Sample user profiles
└── tests/                 # Test files
```

### AI Agent Architecture

The system uses Microsoft Agent Framework with the following components:

- **SkillMatchAgent**: Main AI agent with specialized tools
- **Tools**: Function calling for specific operations
  - `find_matching_opportunities`
  - `calculate_match_score`
  - `identify_skill_gaps`
  - `get_skill_recommendations`
  - `analyze_user_strengths`
  - `search_learning_opportunities`

### Matching Algorithm

The matching algorithm considers multiple factors:

1. **Skill Compatibility** (50% weight)
   - Exact skill matches
   - Skill level comparison
   - Related skills consideration
   - Experience years bonus

2. **Experience Level** (30% weight)
   - Total years of experience
   - Role-specific experience
   - Career progression

3. **Preference Alignment** (20% weight)
   - Work type (remote/hybrid/onsite)
   - Location preferences
   - Salary expectations
   - Industry preferences

## 🎯 Skill Categories

The system organizes skills into categories:

- **Programming Languages**: Python, JavaScript, Java, C++, etc.
- **Web Development**: React, Vue.js, Node.js, Django, etc.
- **Data Science & ML**: Machine Learning, Deep Learning, Statistics, etc.
- **Database Technologies**: SQL, PostgreSQL, MongoDB, Redis, etc.
- **Cloud & DevOps**: AWS, Azure, Docker, Kubernetes, etc.
- **Soft Skills**: Leadership, Communication, Problem Solving, etc.

## 📊 Example Output

### Match Results
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Title                       ┃ Type        ┃ Company              ┃ Location        ┃ Match Score ┃ Skills Match  ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Senior Python Developer    │ Job         │ TechCorp Inc         │ San Francisco   │    87.3%    │     92.1%     │
│ Data Scientist             │ Job         │ Analytics Pro        │ New York, NY    │    78.9%    │     85.4%     │
│ Machine Learning Fundamentals│ Learning   │ DataLearn Academy    │ Remote          │    65.2%    │     78.8%     │
└─────────────────────────────┴─────────────┴──────────────────────┴─────────────────┴─────────────┴───────────────┘

Top Match Details:
📋 Senior Python Developer
🏢 TechCorp Inc
📍 San Francisco, CA
📊 Overall Match: 87.3%
💡 This is an excellent match for your profile. Your skills align very well with the requirements. Your experience level meets requirements.

✓ Your Strengths:
  • Python (experienced)
  • SQL (experienced)
  • Strong in Programming Languages

📚 Skills to Develop:
  • Docker (None → beginner)
  • AWS (beginner → intermediate)
```

## 🔧 Configuration

### Environment Variables

```bash
# Required
export GITHUB_TOKEN="your_github_personal_access_token"

# Optional
export SKILLMATCH_MODEL="openai/gpt-4.1-mini"
export SKILLMATCH_CONFIG_PATH="config/config.json"
```

### Configuration File

Create `config/config.json`:

```json
{
  "github_token": "your_github_token",
  "model_id": "openai/gpt-4.1-mini",
  "skills_db_path": "data/skills_database.json",
  "opportunities_db_path": "data/opportunities_database.json"
}
```

## 🧪 Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/skillmatch
```

## 📈 Extending SkillMatch.AI

### Adding New Skills

Edit `data/skills_database.json`:

```json
{
  "skill_categories": {
    "your_category": {
      "category_name": "Your Category",
      "description": "Description of your category",
      "skills": {
        "your_skill": {
          "name": "Your Skill",
          "description": "Skill description",
          "related_skills": ["related1", "related2"],
          "levels": ["beginner", "intermediate", "advanced", "expert"]
        }
      }
    }
  }
}
```

### Adding New Opportunities

Add to `data/opportunities_database.json`:

```json
{
  "opportunities": [
    {
      "opportunity_id": "unique_id",
      "title": "Job Title",
      "description": "Job description",
      "opportunity_type": "job",
      "required_skills": [
        {
          "skill_id": "python",
          "skill_name": "Python",
          "category": "programming",
          "required_level": "intermediate",
          "importance": 0.8,
          "is_mandatory": true
        }
      ]
    }
  ]
}
```

### Custom Matching Logic

Extend the `SkillMatcher` class:

```python
from skillmatch.utils import SkillMatcher

class CustomSkillMatcher(SkillMatcher):
    def _calculate_custom_score(self, user_profile, opportunity):
        # Your custom matching logic
        return score
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Microsoft Agent Framework** for the powerful AI agent capabilities
- **GitHub Models** for providing access to state-of-the-art language models
- **Pydantic** for robust data validation and modeling
- **Rich** for beautiful terminal output
- **Click** for the CLI framework

## 📚 Additional Documentation

- **[VECTOR_DATABASE_SUCCESS.md](VECTOR_DATABASE_SUCCESS.md)** - Vector database integration details
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Full project overview and results

## �📞 Support

- 📧 Email: [team@skillmatch.ai](mailto:team@skillmatch.ai)
- 🐛 Issues: [GitHub Issues](https://github.com/rubyferdianto/SkillMatch.AI/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/rubyferdianto/SkillMatch.AI/discussions)

## 🗺️ Roadmap

### ✅ Completed Features
- [x] **Modern Web Interface** - ✅ **COMPLETED!**
- [x] **Comprehensive Career Profiles System** - ✅ **COMPLETED!**
- [x] **Professional Profile Management** - ✅ **COMPLETED!**
- [x] **Resume Upload & Management** - ✅ **COMPLETED!**
- [x] **Profile Analytics & Statistics** - ✅ **COMPLETED!**
- [x] **Full CRUD Operations** - ✅ **COMPLETED!**
- [x] **Vector Database Integration** - ✅ **COMPLETED!**
- [x] **PDF Resume Processing** - ✅ **COMPLETED!**

### 🚧 In Development
- [ ] Enhanced AI matching algorithms
- [ ] Advanced profile recommendations
- [ ] Bulk profile operations

### 📋 Future Plans
- [ ] Integration with additional job boards (LinkedIn, Indeed)
- [ ] Advanced learning path recommendations
- [ ] Team skill analysis and planning
- [ ] API for third-party integrations
- [ ] Mobile application
- [ ] Enterprise features and deployment
- [ ] React frontend migration
- [ ] Real-time collaboration features

---

**Built with ❤️ by the SkillMatch.AI team**