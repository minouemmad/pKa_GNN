param(
    [int[]]$Seeds = @(42, 1, 7, 123),
    [int]$DsIdx = 2,        # radius 9 = index 2 in [7,8,9,10,11]
    [string[]]$OnlyVariants = @()
)

$ErrorActionPreference = 'Continue'
$env:PYTHONIOENCODING = 'utf-8'

$root = 'C:\Users\maemm\OneDrive\Desktop\FFX\pKa_GNN\tinker_pipeline'
Set-Location $root

$feat = 'Graph_pKa/Features_FFX138'
$dsBase = 'Graph_pKa/Features_FFX138'

# Variants:
#   Charge       : paper-exact (charge in nodes, no dipole)         input_dim=24
#   NodeDip      : paper-minus-charge (dipole in nodes, no charge)  input_dim=26
#   ChargeDip    : both in nodes                                     input_dim=27
#   EdgeOnlyDip  : charge in nodes; dipole ONLY as edge features    input_dim=24 + edge_dim=4
$variants = @(
    @{ name='Charge';      drop='Dipole_X,Dipole_Y,Dipole_Z'; edge=$false; edim=0 },
    @{ name='NodeDip';     drop='atomic_charge';              edge=$false; edim=0 },
    @{ name='ChargeDip';   drop='';                           edge=$false; edim=0 },
    @{ name='EdgeOnlyDip'; drop='Dipole_X,Dipole_Y,Dipole_Z'; edge=$true;  edim=4 }
)

if ($OnlyVariants.Count -gt 0) {
    $variants = $variants | Where-Object { $OnlyVariants -contains $_.name }
}

# 1) Build datasets
foreach ($v in $variants) {
    $out = "$dsBase/Datasets_$($v.name)_138"
    if (Test-Path $out) { Remove-Item -Recurse -Force $out }
    Write-Host "==== Building dataset: $($v.name)  drop='$($v.drop)'  edge=$($v.edge) ===="
    $env:DROP_FEATURE_COLS = $v.drop
    if ($v.edge) { $env:EDGE_DIPOLE_FEATURES = '1' } else { Remove-Item env:EDGE_DIPOLE_FEATURES -ErrorAction SilentlyContinue }
    python 03_create_datasets.py --feat-dir $feat --out-dir $out *> "data/03_$($v.name)_138.log"
    Get-Content "data/03_$($v.name)_138.log" | Select-String 'input_dim|Radius  9' | Select-Object -First 2 | ForEach-Object { Write-Host "  $_" }
    Remove-Item env:DROP_FEATURE_COLS -ErrorAction SilentlyContinue
}
Remove-Item env:EDGE_DIPOLE_FEATURES -ErrorAction SilentlyContinue

# 2) Train
$results = New-Object System.Collections.Generic.List[object]
foreach ($v in $variants) {
    $ds = "$dsBase/Datasets_$($v.name)_138"
    foreach ($s in $Seeds) {
        $tag = "FFX138_$($v.name)_seed$s"
        $out = "Graph_pKa/Net_FFX138/Training_$tag"
        Write-Host "==== Train $tag ===="
        $cmd = @(
            '05_train.py',
            '--dataset-dir', $ds,
            '--results-dir', $out,
            '--dataset', "$DsIdx",
            '--seed',    "$s"
        )
        if ($v.edim -gt 0) { $cmd += @('--edge-dim', "$($v.edim)") }
        python @cmd *> "data/05_$tag.log"

        $perFold = "$out\predictions\dataset_${DsIdx}_all_folds.csv"
        if (Test-Path $perFold) {
            $rows = Import-Csv $perFold
            $errs = $rows | ForEach-Object { [math]::Abs([double]$_.Predicted_pKa - [double]$_.True_pKa) }
            $mae  = ($errs | Measure-Object -Average).Average
            $rmse = [math]::Sqrt(($errs | ForEach-Object { $_*$_ } | Measure-Object -Average).Average)
            $results.Add([pscustomobject]@{ Variant=$v.name; Seed=$s; MAE=[math]::Round($mae,4); RMSE=[math]::Round($rmse,4); N=$rows.Count })
            Write-Host "  -> MAE=$([math]::Round($mae,4)) RMSE=$([math]::Round($rmse,4)) N=$($rows.Count)"
        } else {
            Write-Host "  -> MISSING $perFold"
        }
    }
}

$results | Format-Table -AutoSize | Out-String | Write-Host
$results | Export-Csv -Path 'Graph_pKa/Net_FFX138/sweep_charge_dipole_138.csv' -NoTypeInformation

Write-Host "`n=== Mean across seeds (radius 9, FFX 138 PDBs) ==="
$agg = $results | Group-Object Variant | ForEach-Object {
    $maes = @($_.Group | ForEach-Object { [double]$_.MAE })
    $mean = ($maes | Measure-Object -Average).Average
    $std  = if ($maes.Count -gt 1) { [math]::Sqrt((($maes | ForEach-Object { ($_-$mean)*($_-$mean) }) | Measure-Object -Sum).Sum / ($maes.Count-1)) } else { 0 }
    [pscustomobject]@{ Variant=$_.Name; MeanMAE=[math]::Round($mean,4); StdMAE=[math]::Round($std,4); N=$_.Count }
}
$agg | Sort-Object MeanMAE | Format-Table -AutoSize | Out-String | Write-Host
$agg | Export-Csv -Path 'Graph_pKa/Net_FFX138/sweep_charge_dipole_138_mean.csv' -NoTypeInformation
