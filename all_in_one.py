"""
All-in-one launcher: starts web_demo.py then cloudflared tunnel.
Writes the public URL to tunnel_url.txt and keeps both processes alive.
Run with: python -u all_in_one.py
"""
import subprocess, re, sys, os, time, threading

WEB_CMD  = [sys.executable, "web_demo.py", "--host", "127.0.0.1", "--port", "7860"]
CF_CMD   = [r"tools\cloudflared.exe", "tunnel", "--url", "http://127.0.0.1:7860"]
URL_FILE = "tunnel_url.txt"

def pipe_stream(stream, prefix, found_event, url_holder):
    for line in stream:
        line = line.rstrip()
        print(f"[{prefix}] {line}", flush=True)
        if not found_event.is_set():
            m = re.search(r'https://[a-z0-9\-]+\.trycloudflare\.com', line)
            if m:
                url = m.group(0)
                url_holder.append(url)
                print(f"\n{'='*60}", flush=True)
                print(f">>> LIVE URL: {url}", flush=True)
                print(f"{'='*60}\n", flush=True)
                with open(URL_FILE, "w") as f:
                    f.write(url + "\n")
                found_event.set()

# Kill any stale processes
os.system("taskkill /F /IM cloudflared.exe /T 2>nul")
time.sleep(1)

# Start web demo
print("[*] Starting web_demo.py ...", flush=True)
web = subprocess.Popen(
    WEB_CMD,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True, bufsize=1
)
print(f"[*] web_demo PID={web.pid}", flush=True)

# Drain web demo output in background thread
threading.Thread(
    target=pipe_stream,
    args=(web.stdout, "WEB", threading.Event(), []),
    daemon=True
).start()

print("[*] Waiting 20s for web demo to initialize...", flush=True)
time.sleep(20)

# Start cloudflared
print("[*] Starting cloudflared tunnel...", flush=True)
cf = subprocess.Popen(
    CF_CMD,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True, bufsize=1
)
print(f"[*] cloudflared PID={cf.pid}", flush=True)

url_holder = []
found = threading.Event()
t = threading.Thread(
    target=pipe_stream,
    args=(cf.stdout, "CF", found, url_holder),
    daemon=True
)
t.start()

# Wait up to 90s for URL
found.wait(timeout=90)
if url_holder:
    print(f"[SUCCESS] App live at: {url_holder[0]}", flush=True)
else:
    print("[ERROR] Tunnel URL not found within 90s", flush=True)

# Keep alive
try:
    web.wait()
except KeyboardInterrupt:
    print("[*] Shutting down...", flush=True)
    web.terminate()
    cf.terminate()
