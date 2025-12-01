"""Run API server with error logging."""
import sys
import traceback

# Write to file for debugging
with open("server_log.txt", "w") as f:
    f.write(f"Python: {sys.version}\n")
    f.write("Starting server...\n")
    f.flush()
    
    try:
        f.write("Importing uvicorn...\n")
        f.flush()
        import uvicorn
        
        f.write("Importing api.main...\n")
        f.flush()
        from api.main import app
        
        f.write("Starting uvicorn...\n")
        f.flush()
        
    except Exception as e:
        f.write(f"ERROR: {type(e).__name__}: {e}\n")
        f.write(traceback.format_exc())
        f.flush()
        sys.exit(1)

# Run outside the with block
uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

