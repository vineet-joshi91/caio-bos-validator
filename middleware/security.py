# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 12:49:25 2026

@author: Vineet
"""

# middleware/security.py
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import time
import logging

logger = logging.getLogger(__name__)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        logger.info(f"➡️  {request.method} {request.url.path}")
        response = await call_next(request)
        duration = time.time() - start
        logger.info(f"⬅️  {response.status_code} ({duration:.3f}s)")
        return response