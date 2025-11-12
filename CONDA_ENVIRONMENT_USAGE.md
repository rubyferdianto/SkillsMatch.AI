# SkillsMatch.AI Conda Environment Usage Summary

This document summarizes how SkillsMatch.AI consistently uses the dedicated `smai` conda environment.

## Dedicated Environment: `smai`

All SkillsMatch.AI scripts and applications use the dedicated conda environment named `smai`.

### Environment Activation Scripts

1. **activate_smai.sh** - Standalone activation helper
   ```bash
   source ./activate_smai.sh
   ```
   - Initializes conda from multiple possible locations
   - Uses `conda activate smai`
   - Verifies activation success

2. **start_skillmatch.sh** - Main application launcher
   ```bash
   ./start_skillmatch.sh
   ```
   - Sources conda initialization
   - Uses `conda activate smai`
   - Falls back to `conda run -n smai` if needed
   - Starts Flask app in web directory

3. **debug_run.sh** - Debug mode launcher
   ```bash
   ./debug_run.sh
   ```
   - Sources activate_smai.sh
   - Runs app in debug mode

4. **web/start_web.sh** - Web-specific launcher
   ```bash
   cd web && ./start_web.sh
   ```
   - Uses `conda activate smai`
   - Falls back to `conda run -n smai` if needed

### Application Environment Checks

**web/app.py** includes startup environment verification:
```python
def check_conda_environment():
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env != 'smai':
        print("⚠️  WARNING: Not running in 'smai' conda environment!")
    else:
        print(f"✅ Running in correct conda environment: {conda_env}")
```

### Environment Verification

**check_smai_env.sh** - Complete environment verification:
```bash
./check_smai_env.sh
```
- Activates smai environment
- Verifies Python version and path
- Tests key imports (Flask, SQLAlchemy, etc.)
- Confirms database and storage modules

## Environment Setup

To create the smai environment:
```bash
conda create -n smai python=3.11
conda activate smai
pip install -r requirements.txt
```

## Consistent Usage Pattern

All scripts follow this pattern:
1. Initialize conda from multiple possible locations
2. Activate the `smai` environment using `conda activate smai`
3. Verify activation with `$CONDA_DEFAULT_ENV`
4. Fall back to `conda run -n smai` if direct activation fails
5. Run the application with proper environment validation

## Benefits

- **Isolation**: Dedicated environment prevents dependency conflicts
- **Consistency**: All components use the same Python/package versions
- **Reliability**: Multiple fallback methods ensure activation works
- **Verification**: Built-in checks confirm correct environment usage
- **Development**: Easy switching between environments for testing

## Current Status

✅ All startup scripts use `conda activate smai`
✅ Application includes environment verification
✅ Multiple activation methods for reliability
✅ Comprehensive testing and verification tools