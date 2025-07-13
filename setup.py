#!/usr/bin/env python3
"""
setup.py - AGIcommander setup and installation script

Sets up the AGIcommander environment and dependencies.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(command, description, exit_on_fail=True):
    """Run a shell command with error handling"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Command: {command}")
        print(f"   Error: {e.stderr}")
        if exit_on_fail:
            sys.exit(1)
        return None


def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        sys.exit(1)
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")


def create_directory_structure():
    """Create the required directory structure"""
    print("📁 Creating directory structure...")
    
    directories = [
        "core",
        "llm_adapters",
        "servers/code",
        "servers/self_reflection", 
        "servers/memory",
        "servers/learning",
        "servers/external",
        "agents",
        "config",
        "memory/vector",
        "memory/relational", 
        "memory/cache",
        "memory/logs",
        "scripts",
        "tests/unit",
        "tests/integration",
        "docs",
        "examples"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        # Create __init__.py files for Python packages
        if not directory.startswith(("config", "memory", "scripts", "tests", "docs", "examples")):
            init_file = Path(directory) / "__init__.py"
            if not init_file.exists():
                init_file.touch()
    
    print("✅ Directory structure created")


def setup_virtual_environment():
    """Set up Python virtual environment"""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return
    
    print("🌐 Creating virtual environment...")
    run_command(f"{sys.executable} -m venv venv", "Virtual environment creation")
    
    # Determine activation script path
    if os.name == 'nt':  # Windows
        activate_script = "venv\\Scripts\\activate"
        pip_path = "venv\\Scripts\\pip"
    else:  # Unix/Linux/MacOS
        activate_script = "venv/bin/activate"
        pip_path = "venv/bin/pip"
    
    print(f"✅ Virtual environment created")
    print(f"   Activate with: source {activate_script}")
    
    return pip_path


def install_dependencies(pip_path=None):
    """Install Python dependencies"""
    pip_cmd = pip_path or "pip"
    
    print("📦 Installing dependencies...")
    
    # Core dependencies (minimal for basic functionality)
    core_deps = [
        "langchain-google-genai>=2.0.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
        "chromadb>=0.4.0",
        "requests>=2.31.0",
        "aiohttp>=3.8.0"
    ]
    
    for dep in core_deps:
        run_command(f"{pip_cmd} install {dep}", f"Installing {dep.split('>=')[0]}")
    
    # Install from requirements.txt if it exists
    if Path("requirements.txt").exists():
        run_command(f"{pip_cmd} install -r requirements.txt", "Installing from requirements.txt", exit_on_fail=False)


def create_env_file():
    """Create .env file template"""
    env_file = Path(".env")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return
    
    print("🔑 Creating .env template...")
    
    env_template = """# AGIcommander Environment Variables
# Copy this file and add your actual API keys

# Required: Google Gemini API Key
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional: Other LLM Provider API Keys
# OPENAI_API_KEY=your_openai_api_key_here
# ANTHROPIC_API_KEY=your_anthropic_api_key_here
# XAI_API_KEY=your_xai_api_key_here

# Optional: External Services
# TAVILY_API_KEY=your_tavily_api_key_here
# GITHUB_TOKEN=your_github_token_here

# Optional: Configuration Overrides
# LOG_LEVEL=INFO
# SAFETY_MODE=true
"""
    
    with open(env_file, "w") as f:
        f.write(env_template)
    
    print("✅ .env template created")
    print("   ⚠️  Please edit .env and add your actual API keys!")


def verify_installation():
    """Verify the installation is working"""
    print("🧪 Verifying installation...")
    
    # Check if core modules can be imported
    try:
        sys.path.insert(0, str(Path.cwd()))
        
        from core.config import AGIConfig
        from llm_adapters.base import BaseLLMAdapter
        
        print("✅ Core modules import successfully")
        
        # Test configuration loading
        config = AGIConfig("config/default.yaml")
        print("✅ Configuration loading works")
        
    except ImportError as e:
        print(f"❌ Module import failed: {e}")
        return False
    except Exception as e:
        print(f"⚠️ Verification warning: {e}")
    
    return True


def main():
    """Main setup function"""
    print("🚀 AGIcommander Setup")
    print("=" * 50)
    
    # Pre-flight checks
    check_python_version()
    
    # Setup steps
    create_directory_structure()
    pip_path = setup_virtual_environment()
    install_dependencies(pip_path)
    create_env_file()
    
    # Verification
    if verify_installation():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Edit .env file and add your Google API key")
        print("2. Activate virtual environment:")
        if os.name == 'nt':
            print("   .\\venv\\Scripts\\activate")
        else:
            print("   source venv/bin/activate")
        print("3. Run the test script:")
        print("   python test_basic.py")
    else:
        print("\n⚠️ Setup completed with warnings")
        print("Please check the error messages above")


if __name__ == "__main__":
    main()
