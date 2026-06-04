param (
    [string]$BlenderVersion = ""
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$addonDir = Join-Path $repoRoot "addon"

if (-not (Test-Path $addonDir)) {
    Write-Host "Error: Could not find addon directory at $addonDir" -ForegroundColor Red
    exit 1
}

$blenderAppData = Join-Path $env:APPDATA "Blender Foundation\Blender"

if (-not (Test-Path $blenderAppData)) {
    Write-Host "Blender AppData directory not found at $blenderAppData" -ForegroundColor Yellow
    Write-Host "You may need to run Blender at least once, or specify a custom path."
    exit 1
}

$versionsToLink = @()

if ($BlenderVersion -ne "") {
    $versionsToLink += $BlenderVersion
} else {
    # Find all version directories (e.g. 3.0, 3.6, 4.0)
    $dirs = Get-ChildItem -Path $blenderAppData -Directory | Where-Object { $_.Name -match "^\d+\.\d+$" }
    foreach ($dir in $dirs) {
        $versionsToLink += $dir.Name
    }
}

if ($versionsToLink.Count -eq 0) {
    Write-Host "No Blender versions found in $blenderAppData" -ForegroundColor Yellow
    exit 1
}

foreach ($version in $versionsToLink) {
    $targetAddonsDir = Join-Path $blenderAppData "$version\scripts\addons"
    $targetLinkPath = Join-Path $targetAddonsDir "open_jaywalker"

    if (-not (Test-Path $targetAddonsDir)) {
        Write-Host "Creating $targetAddonsDir"
        New-Item -ItemType Directory -Force -Path $targetAddonsDir | Out-Null
    }

    if (Test-Path $targetLinkPath) {
        # Check if it's already a junction to the right place
        $existingTarget = (Get-Item $targetLinkPath).Target
        if ($existingTarget -eq $addonDir) {
            Write-Host "Symlink already exists for Blender $version ($targetLinkPath -> $addonDir)" -ForegroundColor Green
            continue
        } else {
            Write-Host "Removing existing open_jaywalker folder/link for Blender $version" -ForegroundColor Yellow
            Remove-Item -Recurse -Force $targetLinkPath
        }
    }

    Write-Host "Creating symlink for Blender $($version): $targetLinkPath -> $addonDir"
    # Using Junction for Windows compatibility without needing admin rights
    New-Item -ItemType Junction -Path $targetLinkPath -Value $addonDir | Out-Null
    Write-Host "Successfully created symlink for Blender $version" -ForegroundColor Green
}

Write-Host "Done!"
