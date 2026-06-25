"""
Complete launcher: starts web demo + cloudflared, gets URL, exits.
Key: uses subprocess.DETACHED_PROCESS + CLOSE_FDS so children survive parent exit.
The parent exits after getting URL, children keep running.
"""
import subprocess, sys, time, re, urllib.request, socket

PYTHON = sys.executable
CF = r"e:\modelspeech\tools\cloudflared.exe"
WEB_SCRIPT = r"e:\modelspeech\web_demo.py"
URL_FILE = r"e:\modelspeech\tunnel_url.txt"
PORT = 7860

# Windows flags for truly detached processes
DETACHED = 0x00000008          # DETACHED_PROCESS
NEW_GROUP = 0x00000200         # CREATE_NEW_PROCESS_GROUP  
NO_WINDOW = 0x08000000         # CREATE_NO_WINDOW

def port_open(host, port, timeout=1):
    try:
        s = socket.create_connection((host, port), timeout=timeout)
        s.close()
        return True
    except:
        return False

# Step 1: Start web demo detached
print(f"[1] Starting web_demo.py...", flush=True)
web = subprocess.Popen(
    [PYTHON, WEB_SCRIPT, "--host", "127.0.0.1", "--port", str(PORT)],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    creationflags=DETACHED | NEW_GROUP | NO_WINDOW,
    close_fds=True,
)
print(f"[1] web_demo PID={web.pid}", flush=True)

# Step 2: Wait for web demo to be ready
print(f"[2] Waiting for web demo on port {PORT}...", flush=True)
for i in range(30):
    if port_open("127.0.0.1", PORT):
        print(f"[2] Web demo UP after {i+1}s", flush=True)
        break
    time.sleep(1)
else:
    print(f"[2] Web demo not responding after 30s - continuing anyway", flush=True)

# Step 3: Start cloudflared detached
print(f"[3] Starting cloudflared...", flush=True)
cf_log = open(r"e:\modelspeech\cf_live.log", "w", buffering=1)
cf = subprocess.Popen(
    [CF, "tunnel", "--url", f"http://127.0.0.1:{PORT}"],
    stdin=subprocess.DEVNULL,
    stdout=cf_log,
    stderr=cf_log,
    creationflags=NEW_GROUP | NO_WINDOW,
    close_fds=True,
)
print(f"[3] cloudflared PID={cf.pid}", flush=True)

# Step 4: Wait for cloudflared metrics API
print(f"[4] Waiting for cloudflared metrics API...", flush=True)
for i in range(30):
    if port_open("127.0.0.1", 20241):
        print(f"[4] Metrics API UP after {i+1}s", flush=True)
        break
    time.sleep(1)
else:
    print(f"[4] Metrics API not up after 30s", flush=True)

# Step 5: Read log file for URL
print(f"[5] Scanning cf_live.log for URL...", flush=True)
cf_log.flush()
url = None
try:
    with open(r"e:\modelspeech\cf_live.log", "r", errors="replace") as f:
        content = f.read()
    print(f"[5] Log content ({len(content)} bytes):", flush=True)
    print(content[:3000], flush=True)
    m = re.search(r'https://[a-z0-9\-]+\.trycloudflare\.com', content)
    if m:
        url = m.group(0)
except Exception as e:
    print(f"[5] Error reading log: {e}", flush=True)

if url:
    print(f"\n{'='*60}", flush=True)
    print(f"LIVE_URL={url}", flush=True)
    print(f"{'='*60}\n", flush=True)
    with open(URL_FILE, "w") as f:
        f.write(url)
else:
    print(f"[5] URL not found yet in log", flush=True)

print("Script complete. web_demo and cloudflared running in background.", flush=True)
