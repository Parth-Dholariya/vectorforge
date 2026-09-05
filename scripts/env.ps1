# Pins the 64-bit mingw-w64 toolchain for this shell.
#
# Why this exists: C:\MinGW\bin holds a 32-bit MinGW.org GCC 6.3 that shadows
# the good compiler on PATH. A 32-bit toolchain caps the address space at 2 GB,
# which makes GIST-1M (3.84 GB raw) unloadable and mmap of large index files
# impossible. Prepending the WinLibs bin dir wins the lookup.
#
# Usage:  . .\scripts\env.ps1

$MINGW64 = Join-Path $env:LOCALAPPDATA `
    'Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin'

if (-not (Test-Path $MINGW64)) {
    Write-Error "mingw-w64 not found at $MINGW64. Run: winget install -e --id BrechtSanders.WinLibs.POSIX.UCRT"
    return
}

$env:PATH = "$MINGW64;$env:PATH"
$env:CC   = Join-Path $MINGW64 'gcc.exe'
$env:CXX  = Join-Path $MINGW64 'g++.exe'

$target = & g++ -dumpmachine
if ($target -ne 'x86_64-w64-mingw32') {
    Write-Error "expected x86_64-w64-mingw32, got '$target' - a 32-bit gcc is still shadowing PATH"
    return
}

Write-Host "toolchain: $(& g++ --version | Select-Object -First 1)" -ForegroundColor Green
Write-Host "target   : $target" -ForegroundColor Green
