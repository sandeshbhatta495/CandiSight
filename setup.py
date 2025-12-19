#!/usr/bin/env python
"""
Setup script to create virtual environment and install dependencies.
Run: python setup.py
"""

import os
import subprocess
import sys
import platform

def create_venv():
    """Create virtual environment"""
    print("Creating virtual environment...")
    venv_path = "venv"
    
    if os.path.exists(venv_path):
        print(f"Virtual environment '{venv_path}' already exists.")
    else:
        subprocess.check_call([sys.executable, "-m", "venv", venv_path])
        print(f"Virtual environment created at '{venv_path}'")

def get_pip_executable():
    """Get the path to pip executable"""
    if platform.system() == "Windows":
        return os.path.join("venv", "Scripts", "pip.exe")
    else:
        return os.path.join("venv", "bin", "pip")

def install_requirements():
    """Install required packages"""
    print("\nInstalling required packages...")
    pip_exe = get_pip_executable()
    
    if not os.path.exists(pip_exe):
        print("ERROR: pip executable not found. Virtual environment may not be created properly.")
        return False
    
    # Upgrade pip first
    print("Upgrading pip...")
    subprocess.check_call([pip_exe, "install", "--upgrade", "pip"])
    
    # Install from requirements.txt
    if os.path.exists("requirements.txt"):
        print("Installing packages from requirements.txt...")
        subprocess.check_call([pip_exe, "install", "-r", "requirements.txt"])
    else:
        print("ERROR: requirements.txt not found!")
        return False
    
    print("All packages installed successfully!")
    return True

def main():
    """Main setup function"""
    print("=" * 60)
    print("CandiSight ML Project - Setup Script")
    print("=" * 60)
    
    try:
        create_venv()
        success = install_requirements()
        
        if success:
            print("\n" + "=" * 60)
            print("✓ Setup Complete!")
            print("=" * 60)
            print("\nTo activate the virtual environment, run:")
            if platform.system() == "Windows":
                print("  venv\\Scripts\\activate.bat")
            else:
                print("  source venv/bin/activate")
            print("\nTo deactivate, run:")
            print("  deactivate")
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Setup failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
