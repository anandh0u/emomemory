"""
EmoMemory Setup Script

Quick setup utility for the EmoMemory project.
"""

import subprocess
import sys
import os
from pathlib import Path


def print_banner():
    """Print setup banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║     🧠 EmoMemory Setup                               ║
    ║     Memory-Enabled Emotion AI powered by Cognee      ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """
    print(banner)


def check_python_version():
    """Check if Python version is 3.9+."""
    print("\n[1/5] Checking Python version...")
    
    if sys.version_info < (3, 9):
        print("❌ Python 3.9 or higher is required!")
        print(f"   Current version: {sys.version}")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")


def check_virtual_env():
    """Check if running in virtual environment."""
    print("\n[2/5] Checking virtual environment...")
    
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if not in_venv:
        print("⚠️  Not running in a virtual environment")
        print("   Recommendation: Create a venv first")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("\n📖 To create a virtual environment:")
            print("   python -m venv .venv")
            print("   .venv\\Scripts\\activate  (Windows)")
            print("   source .venv/bin/activate  (Linux/Mac)")
            sys.exit(0)
    else:
        print("✅ Running in virtual environment")


def install_dependencies():
    """Install required packages."""
    print("\n[3/5] Installing dependencies...")
    print("   This may take a few minutes...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        sys.exit(1)


def check_cognee_setup():
    """Check if Cognee is properly installed."""
    print("\n[4/5] Verifying Cognee installation...")
    
    try:
        import cognee
        print("✅ Cognee is installed")
        
        # Check for API key
        api_key = os.getenv("COGNEE_API_KEY")
        if api_key:
            print("✅ Cognee API key found in environment")
        else:
            print("⚠️  No Cognee API key found")
            print("   For Cognee Cloud, set COGNEE_API_KEY environment variable")
            print("   For local usage, no key is needed")
            
    except ImportError:
        print("❌ Cognee not found!")
        print("   Installing Cognee...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "cognee"
            ])
            print("✅ Cognee installed")
        except subprocess.CalledProcessError:
            print("❌ Failed to install Cognee")
            sys.exit(1)


def create_directories():
    """Create necessary directories."""
    print("\n[5/5] Setting up directories...")
    
    directories = [
        "artifacts",
        "data",
        "agents",
        ".agents"
    ]
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   Created: {dir_name}/")
    
    print("✅ Directory structure ready")


def print_next_steps():
    """Print next steps for user."""
    print("\n" + "="*60)
    print("  🎉 Setup Complete!")
    print("="*60)
    print("\n📖 Next Steps:\n")
    
    print("1. Run the CLI demo:")
    print("   python chat_interface.py\n")
    
    print("2. Run the web demo:")
    print("   python web_demo.py")
    print("   Then open: http://localhost:7860\n")
    
    print("3. (Optional) Set Cognee Cloud API key:")
    print("   set COGNEE_API_KEY=your_key  (Windows)")
    print("   export COGNEE_API_KEY=your_key  (Linux/Mac)\n")
    
    print("4. Read the documentation:")
    print("   - README.md - Project overview")
    print("   - HACKATHON_SUBMISSION.md - Submission details")
    print("   - DEMO_VIDEO_SCRIPT.md - Video guide\n")
    
    print("="*60)
    print("  🧠 EmoMemory is ready to use!")
    print("="*60 + "\n")


def main():
    """Main setup function."""
    try:
        print_banner()
        check_python_version()
        check_virtual_env()
        install_dependencies()
        check_cognee_setup()
        create_directories()
        print_next_steps()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
