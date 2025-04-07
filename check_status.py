#!/usr/bin/env python
"""
Crypto Trader - System Status Check

This script performs a comprehensive check of the system environment,
dependencies, and configuration to help diagnose issues.
"""

import os
import sys
import json
import logging
import platform
import importlib
import subprocess
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("status_check")


def check_system_info():
    """Check system information."""
    logger.info("=== System Information ===")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"System: {platform.system()} {platform.release()}")
    if platform.system() == "Linux":
        try:
            distro = subprocess.check_output(["lsb_release", "-d"]).decode().strip()
            logger.info(f"Distribution: {distro}")
        except:
            pass
    logger.info(f"Processor: {platform.processor()}")
    logger.info(f"Current directory: {os.getcwd()}")


def check_dependencies():
    """Check installed packages and dependencies."""
    logger.info("\n=== Dependencies ===")
    requirements = [
        "flask", "flask_socketio", "numpy", "pandas", "ta-lib", 
        "scikit-learn", "tensorflow", "pytorch", "ccxt", 
        "matplotlib", "python-dotenv", "requests"
    ]
    
    for package in requirements:
        try:
            # Try to import the package
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "unknown")
            logger.info(f"{package}: Installed (version: {version})")
        except ImportError:
            # If package name doesn't match module name
            if package == "ta-lib":
                try:
                    module = importlib.import_module("talib")
                    version = getattr(module, "__version__", "unknown")
                    logger.info(f"{package}: Installed (version: {version})")
                except ImportError:
                    logger.warning(f"{package}: Not installed")
            elif package == "pytorch":
                try:
                    module = importlib.import_module("torch")
                    version = getattr(module, "__version__", "unknown")
                    logger.info(f"{package}: Installed (version: {version})")
                except ImportError:
                    logger.warning(f"{package}: Not installed")
            else:
                logger.warning(f"{package}: Not installed")
        except Exception as e:
            logger.warning(f"{package}: Error checking ({str(e)})")


def check_configs():
    """Check configuration files."""
    logger.info("\n=== Configuration Files ===")
    
    config_files = [
        "config/config.json",
        "config/demo_config.json"
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Check a few key config parameters
                mode = config.get("mode", "unknown")
                is_demo = config.get("is_demo", False)
                symbols = config.get("symbols", [])
                
                logger.info(f"{config_file}: Found (mode: {mode}, demo: {is_demo}, symbols: {len(symbols)})")
            except Exception as e:
                logger.warning(f"{config_file}: Error reading ({str(e)})")
        else:
            logger.warning(f"{config_file}: Not found")


def check_directories():
    """Check important directories."""
    logger.info("\n=== Directory Structure ===")
    
    directories = [
        "agents",
        "config",
        "data",
        "data/demo",
        "data/historical",
        "models",
        "strategies",
        "utils",
        "frontend"
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            files = os.listdir(directory)
            logger.info(f"{directory}: Found ({len(files)} files)")
        else:
            logger.warning(f"{directory}: Not found")


def check_database():
    """Check database status."""
    logger.info("\n=== Database Status ===")
    
    db_files = [
        "data/trader.db",
        "data/demo/trader.db"
    ]
    
    for db_file in db_files:
        if os.path.exists(db_file):
            size_mb = os.path.getsize(db_file) / (1024 * 1024)
            modified = datetime.fromtimestamp(os.path.getmtime(db_file))
            logger.info(f"{db_file}: Found (size: {size_mb:.2f} MB, last modified: {modified})")
        else:
            logger.info(f"{db_file}: Not found")


def check_environment_variables():
    """Check important environment variables."""
    logger.info("\n=== Environment Variables ===")
    
    env_vars = [
        "CRYPTO_TRADER_MODE",
        "CRYPTO_TRADER_BINANCE_API_KEY",
        "CRYPTO_TRADER_USE_TESTNET",
        "CRYPTO_TRADER_LOG_LEVEL"
    ]
    
    for var in env_vars:
        if var in os.environ:
            # Mask API keys
            if "API_KEY" in var or "SECRET" in var:
                value = "****" + os.environ[var][-4:] if len(os.environ[var]) > 4 else "****"
            else:
                value = os.environ[var]
            logger.info(f"{var}: Set ({value})")
        else:
            logger.info(f"{var}: Not set")


def check_frontend():
    """Check the frontend application."""
    logger.info("\n=== Frontend Application ===")
    
    if os.path.exists("frontend/app.py"):
        logger.info("Frontend application: Found")
        
        # Check templates
        if os.path.exists("frontend/templates"):
            templates = os.listdir("frontend/templates")
            logger.info(f"Templates: Found ({len(templates)} files)")
        else:
            logger.warning("Templates: Not found")
            
        # Check static files
        if os.path.exists("frontend/static"):
            static_files = os.listdir("frontend/static")
            logger.info(f"Static files: Found ({len(static_files)} files)")
        else:
            logger.warning("Static files: Not found")
    else:
        logger.warning("Frontend application: Not found")


def main():
    """Run all status checks."""
    logger.info("=== Crypto Trader System Status Check ===")
    logger.info(f"Time: {datetime.now()}")
    logger.info("-" * 50)
    
    try:
        # Run all checks
        check_system_info()
        check_dependencies()
        check_configs()
        check_directories()
        check_database()
        check_environment_variables()
        check_frontend()
        
        logger.info("\n=== Status Check Complete ===")
        logger.info("For more detailed diagnostics, run the system with --log-level=DEBUG")
        
    except Exception as e:
        logger.error(f"Error during status check: {str(e)}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 