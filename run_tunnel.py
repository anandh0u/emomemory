import subprocess, re, sys, os

CLOUDFLARED = r".\tools\cloudflared.exe"
OUT_FILE = "tunnel_url.txt"

proc = subprocess.Popen(
    [CLOUDFLARED, "tunnel", "--url", "http://127.0.0.1:7860"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,   # merge stderr into stdout
    text=True,
    bufsize=1,
)

sys.stdout.write(f"cloudflared PID={proc.pid}\n")
sys.stdout.flush()

url = None
for line in proc.stdout:
    sys.stdout.write(line)
    sys.stdout.flush()
    m = re.search(r'https://[a-z0-9\-]+\.trycloudflare\.com', line)
    if m and not url:
        url = m.group(0)
        sys.stdout.write(f"\n>>> LIVE URL: {url}\n\n")
        sys.stdout.flush()
        with open(OUT_FILE, "w") as f:
            f.write(url + "\n")
        # Keep running so tunnel stays alive
