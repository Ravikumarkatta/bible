# Set error action preference
$ErrorActionPreference = "Stop"

# Define paths
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootPath = Split-Path -Parent $scriptPath
$downloadsPath = Join-Path $rootPath "data\downloads"

# Bible source URL
$url = "https://www.gutenberg.org/files/10/10-0.txt"
$outputFile = "kjv_bible.txt"
$outputPath = Join-Path $downloadsPath $outputFile

# Create downloads directory
Write-Host "Creating downloads directory..."
New-Item -ItemType Directory -Force -Path $downloadsPath | Out-Null

try {
    # Download the Bible text
    Write-Host "Downloading KJV Bible from Project Gutenberg..."
    Invoke-WebRequest -Uri $url -OutFile $outputPath
    Write-Host "Successfully downloaded Bible text to: $outputPath"
}
catch {
    Write-Error "Failed to download Bible: $_"
    exit 1
}