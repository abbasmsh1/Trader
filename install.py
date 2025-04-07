#!/usr/bin/env python
"""
Crypto Trader - Installation Script

This script helps with the initial setup of the Crypto Trader system by:
1. Checking and installing required dependencies
2. Creating necessary directories
3. Generating default configuration files
4. Setting up a basic environment
"""

import os
import sys
import json
import logging
import argparse
import platform
import subprocess
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("install")


def check_python_version():
    """Check if Python version is compatible."""
    logger.info("Checking Python version...")
    
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        logger.error("Crypto Trader requires Python 3.8 or higher.")
        logger.error(f"Current Python version: {platform.python_version()}")
        return False
    
    logger.info(f"Python version {platform.python_version()} is compatible.")
    return True


def check_pip():
    """Check if pip is installed and up to date."""
    logger.info("Checking pip installation...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "--version"], 
                              stdout=subprocess.DEVNULL, 
                              stderr=subprocess.DEVNULL)
        
        # Update pip to latest version
        logger.info("Updating pip to latest version...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                             stdout=subprocess.DEVNULL)
        
        return True
    except subprocess.SubprocessError:
        logger.error("pip is not installed or not working correctly.")
        return False


def create_directories():
    """Create required directories."""
    logger.info("Creating directories...")
    
    directories = [
        "config",
        "data",
        "data/demo",
        "data/historical",
        "data/cache",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    return True


def install_dependencies(skip_dependencies=False):
    """Install required dependencies."""
    
    if skip_dependencies:
        logger.info("Skipping dependency installation as requested.")
        return True
    
    logger.info("Installing dependencies...")
    
    # First, try to install TA-Lib which can be tricky
    install_talib()
    
    # Then install other requirements
    try:
        # Use the requirements.txt file
        req_file = Path("requirements.txt")
        if req_file.exists():
            logger.info("Installing packages from requirements.txt...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        else:
            logger.warning("requirements.txt not found. Creating minimal requirements...")
            # Create minimal requirements
            with open("requirements.txt", "w") as f:
                f.write("""
flask==2.3.3
flask-socketio==5.3.5
numpy==1.24.3
pandas==2.0.3
python-dotenv==1.0.0
ccxt==4.0.0
scikit-learn==1.3.0
matplotlib==3.7.2
joblib==1.3.2
requests==2.31.0
                """.strip())
            
            # Install from the newly created file
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        logger.info("Dependencies installed successfully.")
        return True
    except subprocess.SubprocessError as e:
        logger.error(f"Failed to install dependencies: {str(e)}")
        return False


def install_talib():
    """Attempt to install TA-Lib which can be challenging on some systems."""
    logger.info("Checking TA-Lib installation...")
    
    # First check if TA-Lib is already installed
    try:
        import talib
        logger.info("TA-Lib is already installed.")
        return True
    except ImportError:
        pass
    
    # Installation methods differ by platform
    system = platform.system()
    
    if system == "Windows":
        logger.info("Windows system detected, installing TA-Lib from wheel...")
        try:
            # For Windows, use pre-built wheel based on Python version
            if sys.version_info.major == 3:
                if sys.version_info.minor == 8:
                    wheel = "TA_Lib-0.4.24-cp38-cp38-win_amd64.whl"
                elif sys.version_info.minor == 9:
                    wheel = "TA_Lib-0.4.24-cp39-cp39-win_amd64.whl"
                elif sys.version_info.minor == 10:
                    wheel = "TA_Lib-0.4.24-cp310-cp310-win_amd64.whl"
                elif sys.version_info.minor == 11:
                    wheel = "TA_Lib-0.4.24-cp311-cp311-win_amd64.whl"
                else:
                    logger.warning("No pre-built TA-Lib wheel for your Python version.")
                    logger.warning("TA-Lib will be installed from source (if possible).")
                    wheel = None
                
                if wheel:
                    url = f"https://download.lfd.uci.edu/pythonlibs/archived/TA_Lib/{wheel}"
                    logger.info(f"Downloading {wheel}...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", url])
                    return True
                    
        except subprocess.SubprocessError as e:
            logger.error(f"Failed to install TA-Lib from wheel: {str(e)}")
    
    elif system == "Linux":
        logger.info("Linux system detected, installing TA-Lib dependencies...")
        try:
            # Install system dependencies first
            subprocess.check_call(["sudo", "apt-get", "update"])
            subprocess.check_call(["sudo", "apt-get", "install", "-y", "build-essential"])
            subprocess.check_call(["sudo", "apt-get", "install", "-y", "ta-lib"])
        except subprocess.SubprocessError as e:
            logger.warning(f"Could not install system dependencies: {str(e)}")
            logger.warning("You might need to install TA-Lib manually.")
    
    elif system == "Darwin":  # macOS
        logger.info("macOS system detected, installing TA-Lib via homebrew...")
        try:
            # On macOS, use homebrew
            subprocess.check_call(["brew", "install", "ta-lib"])
        except subprocess.SubprocessError as e:
            logger.warning(f"Could not install TA-Lib via homebrew: {str(e)}")
            logger.warning("You might need to install TA-Lib manually.")
    
    # Finally, try to pip install regardless of platform
    logger.info("Attempting to install TA-Lib Python wrapper...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "TA-Lib"])
        logger.info("TA-Lib installation succeeded!")
        return True
    except subprocess.SubprocessError as e:
        logger.error(f"TA-Lib installation failed: {str(e)}")
        logger.error("You will need to install TA-Lib manually. See README.md for instructions.")
        return False


def create_config_files():
    """Create default configuration files."""
    logger.info("Creating configuration files...")
    
    # Import config module to create default config
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from config.config import create_default_config
        
        # Create default configuration
        config_created = create_default_config()
        
        if config_created:
            logger.info("Created default configuration file.")
        else:
            logger.info("Default configuration file already exists.")
        
        return True
    except ImportError:
        logger.error("Could not import config module. Make sure it exists.")
        
        # Create a minimal config file if not exists
        config_path = os.path.join("config", "config.json")
        if not os.path.exists(config_path):
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            # Minimal default config
            with open(config_path, "w") as f:
                json.dump({
                    "mode": "run",
                    "use_testnet": True,
                    "base_currency": "USDT",
                    "initial_balance": 10000.0,
                    "symbols": ["BTC/USDT", "ETH/USDT"],
                    "exchanges": {
                        "binance": {
                            "api_key": "",
                            "api_secret": "",
                            "testnet": True
                        }
                    }
                }, f, indent=4)
            
            logger.info(f"Created minimal configuration file at {config_path}")
        
        return False


def setup_demo():
    """Set up the demo environment."""
    logger.info("Setting up demo environment...")
    
    try:
        # Run the setup_demo script
        if os.path.exists("setup_demo.py"):
            subprocess.check_call([sys.executable, "setup_demo.py"])
            logger.info("Demo environment setup complete.")
            return True
        else:
            logger.error("setup_demo.py script not found.")
            return False
    except subprocess.SubprocessError as e:
        logger.error(f"Demo setup failed: {str(e)}")
        return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Crypto Trader Installation Script')
    
    parser.add_argument('--skip-dependencies', action='store_true',
                        help='Skip installing dependencies')
    parser.add_argument('--setup-demo', action='store_true',
                        help='Set up the demo environment')
    
    return parser.parse_args()


def main():
    """Main installation function."""
    logger.info("=== Crypto Trader Installation ===")
    logger.info(f"Time: {datetime.now()}")
    logger.info("-" * 50)
    
    args = parse_arguments()
    
    # Check Python version first
    if not check_python_version():
        return 1
    
    # Check pip installation
    if not check_pip():
        logger.error("Please install pip before continuing.")
        return 1
    
    # Create necessary directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies(args.skip_dependencies):
        logger.warning("Some dependencies may not have been installed correctly.")
        logger.warning("Check the output above for details.")
    
    # Create configuration files
    create_config_files()
    
    # Set up demo environment if requested
    if args.setup_demo:
        setup_demo()
    
    logger.info("\n=== Installation Complete ===")
    logger.info("To start the trading system: python main.py")
    logger.info("To start the web dashboard: cd frontend && python app.py")
    logger.info("For demo mode: python main.py --config config/demo_config.json")
    logger.info("To check system status: python check_status.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 