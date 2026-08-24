<#
DI MVP - text chat prototype for the "digital immortalization" avatar project.

What this does:
- Renders DI/prompt-v2.txt with profile.json into a real system prompt.
- Runs a local chat loop against the Anthropic Messages API (Invoke-RestMethod,
  no SDK install needed) using the ANTHROPIC_API_KEY environment variable.
- Applies a keyword-based safety guardrail BEFORE the model ever sees a message
  that looks like acute distress / suicidal ideation - this bypasses the
  persona entirely and never depends on the model "choosing" to break character.
- Keeps a naive local memory log (memory.jsonl) and injects the last few turns
  as [HISTORICAL_CONTEXT], matching what prompt-v2.txt expects.

Known MVP limitations (see the audit artifact for the full list):
- Memory is "last N turns", not real semantic RAG.
- The safety guardrail is a keyword heuristic, not a trained classifier -
  it WILL miss things and WILL false-positive. It is a stand-in for the
  determinstic, out-of-band guardrail the review agents said this needs.
- profile.json ships with fictional example data - replace before real use,
  and only with data you have documented consent to use.
- No voice/video/biometric detection - text only.

Usage:
  $env:ANTHROPIC_API_KEY = "sk-ant-..."
  ./Start-Chat.ps1
#>

param(
    [string]$ProfilePath = (Join-Path $PSScriptRoot 'profile.json'),
    [string]$PromptPath = (Join-Path $PSScriptRoot '..\prompt-v2.txt'),
    [string]$MemoryPath = (Join-Path $PSScriptRoot 'memory.jsonl'),
    [string]$CrisisLogPath = (Join-Path $PSScriptRoot 'crisis-log.jsonl'),
    [string]$CrisisKeywordsPath = (Join-Path $PSScriptRoot 'crisis-keywords.json'),
    [int]$HistoryTurns = 6
)

$ErrorActionPreference = 'Stop'

if (-not $env:ANTHROPIC_API_KEY) {
    Write-Host "ANTHROPIC_API_KEY is not set. Set it in this shell first, e.g.:" -ForegroundColor Yellow
    Write-Host '  $env:ANTHROPIC_API_KEY = "sk-ant-..."' -ForegroundColor Yellow
    exit 1
}
if (-not (Test-Path $ProfilePath)) { throw "Profile not found: $ProfilePath" }
if (-not (Test-Path $PromptPath)) { throw "Prompt template not found: $PromptPath" }

$profileData = Get-Content $ProfilePath -Raw | ConvertFrom-Json
$crisisWords = @()
if (Test-Path $CrisisKeywordsPath) {
    $crisisWords = [string[]](Get-Content $CrisisKeywordsPath -Raw | ConvertFrom-Json)
}

function Get-FamilyBlock {
    param($family)
    if (-not $family -or $family.Count -eq 0) { return "(no family directory entries provided)" }
    ($family | ForEach-Object { "- $($_.name) | $($_.relationship) | $($_.story)" }) -join "`n"
}

function Get-RenderedSystemPrompt {
    param([string]$template, $p)
    $text = $template
    $text = $text.Replace('[LEGACY_FULL_NAME]', [string]$p.legacy_full_name)
    $text = $text.Replace('[INSERT_GENERAL_TONE: E.g., Warm, protective, joker, reflective, direct].', [string]$p.general_tone)
    $text = $text.Replace('[INSERT_DISTINCTIVE_PHRASES_AND_EXPRESSIONS: E.g., a list of 3-6 short phrases or verbal tics this person actually used].', ($p.distinctive_phrases -join '; '))
    $text = $text.Replace('[INSERT_FAVORITE_TOPICS: E.g., a short list].', ($p.favorite_topics -join ', '))
    $text = $text.Replace('[INSERT_HOBBIES: E.g., a short list].', ($p.hobbies -join ', '))
    $text = $text.Replace('[INSERT_SPORTS_AND_PREFERENCES: E.g., a short list].', [string]$p.sports_preferences)
    $familyBlock = Get-FamilyBlock $p.family_directory
    $text = $text + "`n`n[FAMILY DIRECTORY DATA]`n$familyBlock`n(Only reference the people and stories listed above - never invent a family member or anecdote not provided here.)"
    return $text
}

function Test-CrisisSignal {
    param([string]$text, [string[]]$keywords)
    $lower = $text.ToLowerInvariant()
    foreach ($k in $keywords) {
        if ($lower.Contains($k.ToLowerInvariant())) { return $k }
    }
    return $null
}

function Get-HistoricalContext {
    param([string]$path, [int]$turns)
    if (-not (Test-Path $path)) { return "" }
    $lines = Get-Content $path -ErrorAction SilentlyContinue | Select-Object -Last ($turns * 2)
    if (-not $lines) { return "" }
    $entries = $lines | ForEach-Object { $_ | ConvertFrom-Json }
    ($entries | ForEach-Object { "[$($_.role)] $($_.content)" }) -join "`n"
}

function Add-MemoryEntry {
    param([string]$path, [string]$role, [string]$content)
    $entry = [ordered]@{ role = $role; content = $content; ts = (Get-Date).ToString('o') }
    ($entry | ConvertTo-Json -Compress) | Add-Content -Path $path -Encoding utf8
}

function Invoke-Claude {
    param([string]$system, [string]$userText, [string]$model)
    $body = @{
        model      = $model
        max_tokens = 700
        system     = $system
        messages   = @(@{ role = 'user'; content = $userText })
    } | ConvertTo-Json -Depth 10

    $headers = @{
        'x-api-key'         = $env:ANTHROPIC_API_KEY
        'anthropic-version' = '2023-06-01'
        'content-type'      = 'application/json'
    }
    $resp = Invoke-RestMethod -Uri 'https://api.anthropic.com/v1/messages' -Method Post -Headers $headers -Body $body
    return ($resp.content | Where-Object { $_.type -eq 'text' } | Select-Object -First 1).text
}

$template = Get-Content $PromptPath -Raw
$baseSystemPrompt = Get-RenderedSystemPrompt -template $template -p $profileData
$model = if ($profileData.model) { $profileData.model } else { 'claude-sonnet-5' }

if ($profileData.crisis_line_name -like 'REPLACE_WITH*' -or $profileData.crisis_line_number -like 'REPLACE_WITH*') {
    Write-Host "WARNING: profile.json still has placeholder crisis-line info. Set a real local crisis line before using this with anyone for real." -ForegroundColor Yellow
}

Write-Host "================================================================" -ForegroundColor DarkGray
Write-Host " DI prototype - a digital simulation of $($profileData.legacy_full_name)" -ForegroundColor Cyan
Write-Host " This is an AI, built from information provided about this person." -ForegroundColor Cyan
Write-Host " It is not $($profileData.legacy_full_name). Type 'exit' to end the call." -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor DarkGray

while ($true) {
    $userInput = Read-Host "`nYou"
    if ([string]::IsNullOrWhiteSpace($userInput)) { continue }
    if ($userInput -eq 'exit') { break }

    $hit = Test-CrisisSignal -text $userInput -keywords $crisisWords
    if ($hit) {
        Add-MemoryEntry -path $CrisisLogPath -role 'crisis_trigger' -content "matched='$hit' | input='$userInput'"
        Write-Host "`n[safety protocol - not $($profileData.legacy_full_name) speaking]" -ForegroundColor Red
        Write-Host "I hear that you're in real pain right now. I'm an AI simulation, not $($profileData.legacy_full_name), and I'm not equipped to help you through this alone." -ForegroundColor Red
        Write-Host "Please reach out to $($profileData.crisis_line_name): $($profileData.crisis_line_number)." -ForegroundColor Red
        Add-MemoryEntry -path $MemoryPath -role 'user' -content $userInput
        Add-MemoryEntry -path $MemoryPath -role 'assistant' -content '[safety protocol triggered - see crisis-log.jsonl]'
        continue
    }

    $history = Get-HistoricalContext -path $MemoryPath -turns $HistoryTurns
    $systemWithHistory = if ($history) { "$baseSystemPrompt`n`n[HISTORICAL_CONTEXT]`n$history" } else { $baseSystemPrompt }

    try {
        $reply = Invoke-Claude -system $systemWithHistory -userText $userInput -model $model
    } catch {
        Write-Host "`n[error calling the model: $($_.Exception.Message)]" -ForegroundColor Red
        continue
    }

    Write-Host "`n$($profileData.legacy_full_name): $reply" -ForegroundColor Green
    Add-MemoryEntry -path $MemoryPath -role 'user' -content $userInput
    Add-MemoryEntry -path $MemoryPath -role 'assistant' -content $reply
}

Write-Host "`nCall ended." -ForegroundColor DarkGray
