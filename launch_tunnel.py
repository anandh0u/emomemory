import subprocess
import sys
import time
import re
import os
import threading

URL_FILE = "tunnel_url.txt"

def start_web_demo():
    print("[INFO] Starting web_demo.py on port 7860...", flush=True)
    proc = subprocess.Popen(
        [sys.executable, "web_demo.py", "--host", "127.0.0.1", "--port", "7860"],
        stdout=open("web_demo.log", "w"),
        stderr=open("web_demo.err", "w"),
    )
    print(f"[INFO] Web demo PID: {proc.pid}", flush=True)
    return proc

def start_cloudflared():
    print("[INFO] Starting cloudflared...", flush=True)
    proc = subprocess.Popen(
        [r".\tools\cloudflared.exe", "tunnel", "--url", "http://127.0.0.1:7860"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    print(f"[INFO] Cloudflared PID: {proc.pid}", flush=True)
    return proc

def read_url(proc, timeout=90):
    """Read stderr from cloudflared to find the tunnel URL."""
    deadline = time.time() + timeout
    url = None
    
    def read_stderr():
        nonlocal url
        for line in proc.stderr:
            line = line.strip()
            if line:
                print(f"[CF] {line}", flush=True)
                m = re.search(r'https://[a-z0-9\-]+\.trycloudflare\.com', line)
                if m:
                    url = m.group(0)
                    print(f"\n=== TUNNEL URL FOUND: {url} ===\n", flush=True)
                    with open(URL_FILE, "w") as f:
                        f.write(url + "\n")
                    return
    
    t = threading.Thread(target=read_stderr, daemon=True)
    t.start()
    t.join(timeout=timeout)
    return url

if __name__ == "__main__":
    # Kill any existing processes
    os.system("taskkill /F /IM cloudflared.exe /T 2>nul")
    time.sleep(2)
    
    web_proc = start_web_demo()
    print("[INFO] Waiting 18s for web demo to initialize...", flush=True)
    time.sleep(18)
    
    cf_proc = start_cloudflared()
    url = read_url(cf_proc, timeout=90)
    
    if url:
        print(f"[SUCCESS] App is live at: {url}", flush=True)
    else:
        print("[ERROR] Could not find tunnel URL.", flush=True)
    
    # Keep processes alive
    print("[INFO] Keeping tunnel alive. Ctrl+C to stop.", flush=True)
    try:
        web_proc.wait()
    except KeyboardInterrupt:
        print("[INFO] Stopping...", flush=True)
        web_proc.terminate()
        cf_proc.terminate()
