#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
project launch file
"""

import sys
import logging
from pathlib import Path

# adding root directory to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """print launch banner"""
    banner = """
    ============================
    [                          ]
    [ Planner System Activated ]
    [                          ]
    ============================
    """
    print(banner)


def check_dependencies():
    """check critical args"""
    try:
        import langgraph
        logger.info("Dependency check passed")
        return True
    except Exception as e:
        logger.error(f"Error, lack dependencies: {e}")
        return False


def main():
    """main function for the planner system"""
    print_banner()

    if not check_dependencies():
        logger.error("Please try installing necessary dependencies.")
        sys.exit(1)
    
    from config.config import get_config_summary
    config = get_config_summary()
    logger.info("System config:")
    for key, value in config.items():
        logger.info(f"{key}: {value}")

    try:
        logger.info("System initializing...")
        from core.run_time import run_app
        run_app()
    except KeyboardInterrupt:
        logger.info("System exits, goodbye!")
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()