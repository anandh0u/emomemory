import subprocess
import sys
import time
import re
import os

CF = r"tools\cloudflared.exe"
LOG = "cf_live.log"
OUT_FILE = "tunnel_url.txt"

# Kill any existing cloudflared process
os.system("taskkill /F /IM cloudflared.exe /T 2>nul")
time.sleep(1)

# Clean up old files
if os.path.exists(OUT_FILE):
    try:
        os.remove(OUT_FILE)
    except:
        pass

# Open log file with line buffering
log_file = open(LOG, "w", encoding="utf-8", buffering=1)

# Start cloudflared
proc = subprocess.Popen(
    [CF, "tunnel", "--protocol", "http2", "--url", "http://127.0.0.1:7860"],
    stdout=log_file,
    stderr=log_file,
    stdin=subprocess.DEVNULL,
)

print(f"cloudflared spawned with PID={proc.pid}", flush=True)

# Poll the log file for the URL
url = None
start_time = time.time()
while time.time() - start_time < 30:
    time.sleep(1)
    if os.path.exists(LOG):
        try:
            with open(LOG, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
                m = re.search(r'https://[a-z0-9\-]+\.trycloudflare\.com', content)
                if m:
                    url = m.group(0)
                    break
        except Exception as e:
            # File might be locked momentarily
            continue

if url:
    print(f"URL_FOUND={url}", flush=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(url + "\n")
    
    # Enter infinite loop to keep the tunnel process alive
    print("Tunnel is live! Keeping process active...", flush=True)
    try:
        while True:
            # Check if cloudflared is still running
            if proc.poll() is not None:
                print("cloudflared process terminated unexpectedly!", flush=True)
                sys.exit(1)
            time.sleep(5)
    except KeyboardInterrupt:
        print("Stopping tunnel...", flush=True)
        proc.terminate()
else:
    print("URL NOT FOUND within 30 seconds. Exiting.", flush=True)
    proc.terminate()
    sys.exit(1)
