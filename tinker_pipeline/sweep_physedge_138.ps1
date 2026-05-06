param(
    [int[]]$Seeds = @(42, 1, 7, 123, 2026, 17, 99, 314),
    [int]$DsIdx = 2,
    [string[]]$OnlyVariants = @()
)

$ErrorActionPreference = 'Continue'
$env:PYTHONIOENCODING = 'utf-8'

$root = 'C:\Users\maemm\OneDrive\Desktop\FFX\pKa_GNN\tinker_pipeline'
Set-Location $root

$feat = 'Graph_pKa/Features_FFX138'
$dsBase = 'Graph_pKa/Features_FFX138'
$resBase = 'Graph_pKa/Net_FFX138_PhysEdge'

# Common drop list shared by every variant: lab-frame helpers are dropped by
# 03_create_datasets.py automatically, but raw local-frame dipoles are NOT.
# We always drop the raw 3-vector local-frame dipoles in this sweep — they
# carry no rotation-invariant information and have already been shown to be
# noise to the GAT.  What we keep / add varies per variant.
$rawDipDrop = 'Dipole_X,Dipole_Y,Dipole_Z,PermDipole_X,PermDipole_Y,PermDipole_Z'

# Invariant scalar columns produced by 02_prepare_features.py:
$invariantCols = 'Dipole_norm,Dipole_align_z,Dipole_field_proj,PermDipole_norm,PermDipole_align_z,PermDipole_field_proj'

# 4 variants. All use atomic_charge in node features (paper baseline).
#   Charge             : drop raw dipoles AND invariant scalars; no edges     (24 dim)
#   Invariant          : keep invariants only (drop raw dipoles); no edges    (30 dim)
#   PhysEdge           : Charge-only nodes + Coulomb+qd+dd edges               (24 + edge=6)
#   InvariantPhysEdge  : Invariant nodes + Coulomb+qd+dd edges                 (30 + edge=6)
$variants = @(
    @{ name='Charge';            drop="$rawDipDrop,$invariantCols"; coulomb=$false; qd=$false; dd=$false; edim=0 },
    @{ name='Invariant';         drop="$rawDipDrop";                 coulomb=$false; qd=$false; dd=$false; edim=0 },
    @{ name='PhysEdge';          drop="$rawDipDrop,$invariantCols"; coulomb=$true;  qd=$true;  dd=$true;  edim=6 },
    @{ name='InvariantPhysEdge'; drop="$rawDipDrop";                 coulomb=$true;  qd=$true;  dd=$true;  edim=6 }
)

if ($OnlyVariants.Count -gt 0) {
    $variants = $variants | Where-Object { $OnlyVariants -contains $_.name }
}

# 1) Build datasets
foreach ($v in $variants) {
    $out = "$dsBase/Datasets_$($v.name)_138"
    if (Test-Path $out) { Remove-Item -Recurse -Force $out }
    Write-Host "==== Building dataset: $($v.name) coulomb=$($v.coulomb) qd=$($v.qd) dd=$($v.dd) ===="
    $env:DROP_FEATURE_COLS = $v.drop
    if ($v.coulomb) { $env:COULOMB_EDGE = '1' }        else { Remove-Item env:COULOMB_EDGE         -ErrorAction SilentlyContinue }
    if ($v.qd)      { $env:CHARGE_DIPOLE_EDGE = '1' }  else { Remove-Item env:CHARGE_DIPOLE_EDGE   -ErrorAction SilentlyContinue }
    if ($v.dd)      { $env:DIPOLE_DIPOLE_EDGE = '1' }  else { Remove-Item env:DIPOLE_DIPOLE_EDGE   -ErrorAction SilentlyContinue }
    python 03_create_datasets.py --feat-dir $feat --out-dir $out *> "data/03_$($v.name)_138.log"
    Get-Content "data/03_$($v.name)_138.log" | Select-String 'input_dim' | Select-Object -First 1 | ForEach-Object { Write-Host "  $_" }
    Remove-Item env:DROP_FEATURE_COLS -ErrorAction SilentlyContinue
}
Remove-Item env:COULOMB_EDGE,env:CHARGE_DIPOLE_EDGE,env:DIPOLE_DIPOLE_EDGE -ErrorAction SilentlyContinue

# 2) Train: variants x seeds
$results = New-Object System.Collections.Generic.List[object]
foreach ($v in $variants) {
    $ds = "$dsBase/Datasets_$($v.name)_138"
    foreach ($s in $Seeds) {
        $tag = "$($v.name)_seed$s"
        $out = "$resBase/Training_$tag"
        $perFold = "$out\predictions\dataset_${DsIdx}_all_folds.csv"
        if (Test-Path $perFold) {
            Write-Host "==== Skipping $tag (already done) ===="
        } else {
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
        }

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

if (-not (Test-Path $resBase)) { New-Item -ItemType Directory -Path $resBase | Out-Null }
$results | Format-Table -AutoSize | Out-String | Write-Host
$results | Export-Csv -Path "$resBase/sweep_physedge_138.csv" -NoTypeInformation
Write-Host "`nWrote $resBase/sweep_physedge_138.csv"
