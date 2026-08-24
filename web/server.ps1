<#
Minimal local static server for the DI web prototype.
Serves index.html on http://localhost:8787/ so the browser treats the page
as a secure context (required for microphone access) and so the Anthropic
API sees a real origin instead of a null file:// origin.
Leave this window open while using the app; closing it stops the server.
#>

$port = 8787
$root = $PSScriptRoot
$indexPath = Join-Path $root 'index.html'

if (-not (Test-Path $indexPath)) {
    Write-Host "index.html not found next to server.ps1 ($indexPath)" -ForegroundColor Red
    exit 1
}

$listener = New-Object System.Net.HttpListener
$listener.Prefixes.Add("http://localhost:$port/")

try {
    $listener.Start()
} catch {
    Write-Host "Could not start the server on port $port. Is it already running? $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host "DI web prototype running at http://localhost:$port/" -ForegroundColor Cyan
Write-Host "Leave this window open. Close it to stop the server." -ForegroundColor DarkGray

try {
    while ($listener.IsListening) {
        $context = $listener.GetContext()
        $response = $context.Response
        try {
            $bytes = [System.IO.File]::ReadAllBytes($indexPath)
            $response.ContentType = 'text/html; charset=utf-8'
            $response.ContentLength64 = $bytes.Length
            $response.OutputStream.Write($bytes, 0, $bytes.Length)
        } catch {
            $response.StatusCode = 500
        } finally {
            $response.OutputStream.Close()
        }
    }
} finally {
    $listener.Stop()
}
