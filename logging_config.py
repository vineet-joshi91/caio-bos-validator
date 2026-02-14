# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 12:32:54 2026

@author: Vineet
"""

# logging_config.py
import logging
import sys
from logging.handlers import RotatingFileHandler

def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console output
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    
    # File output (rotates at 10MB)
    file_handler = RotatingFileHandler(
        'caio_api.log',
        maxBytes=10*1024*1024,
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    
    # Setup root logger
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)
    
    # Quiet down noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    
    return root