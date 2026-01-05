#!/usr/bin/env python
"""
Entrypoint script for Google App Engine Standard Python 3.
This script properly handles the PORT environment variable.
"""

import os
import uvicorn
from predict import app

if __name__ == "__main__":
    # Get PORT from environment variable, default to 8080
    port = int(os.environ.get("PORT", 8080))
    
    # Start uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=port)

