"""
Step 1: Start cloudflared, redirect ALL output to cf_live.log file, then exit.
The cloudflared process keeps writing to cf_live.log independently.
"""
import subprocess, sys, os, time

CF = r"tools\cloudflared.exe"
LOG = "cf_live.log"

# Kill existing
os.system("taskkill /F /IM cloudflared.exe /T 2>nul")
time.sleep(1)

# Open log file for cloudflared output
log_file = open(LOG, "w", buffering=1)

# Start cloudflared with stdout+stderr going to file
# Use CREATE_NEW_PROCESS_GROUP so it survives our exit
proc = subprocess.Popen(
    [CF, "tunnel", "--url", "http://127.0.0.1:7860"],
    stdout=log_file,
    stderr=log_file,
    stdin=subprocess.DEVNULL,
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
)

print(f"cloudflared started: PID={proc.pid}", flush=True)
print(f"Output going to: {LOG}", flush=True)
print(f"Waiting 15s then scanning for URL...", flush=True)

# Wait for cloudflared to write its URL (usually within 10s)
time.sleep(15)
log_file.flush()

# Now scan the log file for the URL
import re
log_file.close()

url = None
try:
    with open(LOG, "r", errors="replace") as f:
        for line in f:
            m = re.search(r'https://[a-z0-9\-]+\.trycloudflare\.com', line)
            if m:
                url = m.group(0)
                break
except Exception as e:
    print(f"Error reading log: {e}", flush=True)

if url:
    print(f"URL_FOUND={url}", flush=True)
    with open("tunnel_url.txt", "w") as f:
        f.write(url)
else:
    print("URL_NOT_FOUND - check cf_live.log manually", flush=True)

print("Script done. cloudflared still running.", flush=True)
