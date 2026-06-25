# Start web demo in background
$webProc = Start-Process -FilePath ".\.venv\Scripts\python.exe" `
    -ArgumentList "web_demo.py --host 127.0.0.1 --port 7860" `
    -NoNewWindow -PassThru `
    -RedirectStandardOutput "web_demo.log" `
    -RedirectStandardError "web_demo.err"

Write-Host "Web demo PID: $($webProc.Id)"
Write-Host "Waiting 20 seconds for web demo to start..."
Start-Sleep -Seconds 20

# Test if server is up
try {
    $resp = Invoke-WebRequest -Uri "http://127.0.0.1:7860" -TimeoutSec 5 -ErrorAction Stop
    Write-Host "Web demo is UP (status $($resp.StatusCode))"
} catch {
    Write-Host "Web demo may still be starting: $_"
}

Write-Host "Starting cloudflared tunnel..."
# Run cloudflared and capture output line by line
$pinfo = New-Object System.Diagnostics.ProcessStartInfo
$pinfo.FileName = ".\tools\cloudflared.exe"
$pinfo.Arguments = "tunnel --url http://127.0.0.1:7860"
$pinfo.RedirectStandardOutput = $true
$pinfo.RedirectStandardError = $true
$pinfo.UseShellExecute = $false
$pinfo.CreateNoWindow = $true

$p = New-Object System.Diagnostics.Process
$p.StartInfo = $pinfo
$p.Start() | Out-Null

Write-Host "Cloudflared PID: $($p.Id)"

# Read stderr (where cloudflared writes logs) for up to 60 seconds
$timeout = [DateTime]::Now.AddSeconds(60)
$urlFound = $false
while (-not $p.StandardError.EndOfStream -and [DateTime]::Now -lt $timeout) {
    $line = $p.StandardError.ReadLine()
    Write-Host $line
    if ($line -match 'https://[a-z0-9\-]+\.trycloudflare\.com') {
        $url = $matches[0]
        Write-Host "=== TUNNEL URL: $url ==="
        $url | Out-File -FilePath "tunnel_url.txt" -Encoding utf8
        $urlFound = $true
        break
    }
}

if (-not $urlFound) {
    Write-Host "=== URL NOT FOUND in timeout ==="
}

# Keep running
Write-Host "Tunnel running. Press Ctrl+C to stop."
$p.WaitForExit()
