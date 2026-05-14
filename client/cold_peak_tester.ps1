# cold_and_warm_peak_tester.ps1
# Two-stage stress test: cold burst (empty cluster) + warm burst (partially scaled cluster).
# Designed to validate placement decisions in both transient and steady-state scenarios.

param(
    [string]$PythonExe = "python"
)

Write-Host "=== Cold + Warm Peak Test starting ===" -ForegroundColor Cyan
$startTime = Get-Date

# Phase 1: Minimal warmup - only triggers initial pod creation
Write-Host "`n[Phase 1/5] Minimal warmup (10s, 2 RPS)" -ForegroundColor Yellow
& $PythonExe load_simulator.py --total 20 --rate 2

# Phase 2: Cold peak - empty cluster, scheduler must scale up rapidly
Write-Host "`n[Phase 2/5] Cold peak (90s, 30 RPS) - empty cluster scale-up" -ForegroundColor Yellow
& $PythonExe load_simulator.py --total 2700 --rate 30

# Phase 3: Cooldown - partial scale-down (90s Knative delay prevents full drain)
Write-Host "`n[Phase 3/5] Cooldown (45s)" -ForegroundColor Yellow
Start-Sleep -Seconds 45

# Phase 4: Warm peak - some pods still alive, scale-up happens with warm baseline
Write-Host "`n[Phase 4/5] Warm peak (60s, 25 RPS) - partially warm cluster" -ForegroundColor Yellow
& $PythonExe load_simulator.py --total 1500 --rate 25

# Phase 5: Verification - confirms cluster returns to baseline
Write-Host "`n[Phase 5/5] Verification (20s, 10 RPS)" -ForegroundColor Yellow
& $PythonExe load_simulator.py --total 200 --rate 10

$elapsedSec = [math]::Round(((Get-Date) - $startTime).TotalSeconds, 1)
Write-Host "`n=== Cold + Warm Peak Test completed in $elapsedSec seconds ===" -ForegroundColor Green