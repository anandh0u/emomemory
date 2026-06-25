"""
Starts cloudflared as a DETACHED process, reads stderr until URL found,
prints the URL and writes to tunnel_url.txt, then EXITS.
The cloudflared process keeps running independently.
"""
import subprocess, re, sys, os

CF = r"tools\cloudflared.exe"
URL_FILE = "tunnel_url.txt"

# Kill any existing cloudflared
os.system("taskkill /F /IM cloudflared.exe /T 2>nul")

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200

proc = subprocess.Popen(
    [CF, "tunnel", "--url", "http://127.0.0.1:7860"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
)

print(f"cloudflared PID={proc.pid}", flush=True)
print("Scanning output for URL...", flush=True)

found = False
count = 0
for line in proc.stdout:
    line = line.strip()
    count += 1
    print(line, flush=True)
    m = re.search(r'https://[a-z0-9\-]+\.trycloudflare\.com', line)
    if m:
        url = m.group(0)
        print(f"\nURL_FOUND={url}", flush=True)
        with open(URL_FILE, "w") as f:
            f.write(url)
        found = True
        break
    if count > 200:
        print("Too many lines, stopping scan", flush=True)
        break

if not found:
    print("URL_NOT_FOUND", flush=True)

# Exit — cloudflared keeps running detached
print("Exiting grabber. cloudflared still running.", flush=True)
sys.exit(0)
