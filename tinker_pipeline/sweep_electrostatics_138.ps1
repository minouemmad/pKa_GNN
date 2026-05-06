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
$resBase = 'Graph_pKa/Net_FFX138_Electro'

# 6 variants. All use atomic_charge in node features (paper baseline).
#   Charge          : drop both dipole sets, no edge features            (24 dim)
#   InducedDip      : keep induced dipole only (drop Perm)               (27 dim)
#   PermDip         : keep permanent dipole only (drop Induced)          (27 dim)
#   BothDip         : keep both dipoles                                   (30 dim)
#   CoulombEdge     : Charge nodes + Coulomb edge [r, q_iq_j/r]          (24 + edge=2)
#   CoulombEdgeBoth : BothDip nodes + Coulomb edge                       (30 + edge=2)
$variants = @(
    @{ name='Charge';          drop='Dipole_X,Dipole_Y,Dipole_Z,PermDipole_X,PermDipole_Y,PermDipole_Z'; coulomb=$false; dipEdge=$false; edim=0  },
    @{ name='InducedDip';      drop='PermDipole_X,PermDipole_Y,PermDipole_Z';                              coulomb=$false; dipEdge=$false; edim=0  },
    @{ name='PermDip';         drop='Dipole_X,Dipole_Y,Dipole_Z';                                          coulomb=$false; dipEdge=$false; edim=0  },
    @{ name='BothDip';         drop='';                                                                    coulomb=$false; dipEdge=$false; edim=0  },
    @{ name='CoulombEdge';     drop='Dipole_X,Dipole_Y,Dipole_Z,PermDipole_X,PermDipole_Y,PermDipole_Z'; coulomb=$true;  dipEdge=$false; edim=2  },
    @{ name='CoulombEdgeBoth'; drop='';                                                                    coulomb=$true;  dipEdge=$false; edim=2  }
)

if ($OnlyVariants.Count -gt 0) {
    $variants = $variants | Where-Object { $OnlyVariants -contains $_.name }
}

# 1) Build datasets
foreach ($v in $variants) {
    $out = "$dsBase/Datasets_$($v.name)_138"
    if (Test-Path $out) { Remove-Item -Recurse -Force $out }
    Write-Host "==== Building dataset: $($v.name)  drop='$($v.drop)'  coulomb=$($v.coulomb) ===="
    $env:DROP_FEATURE_COLS = $v.drop
    if ($v.coulomb) { $env:COULOMB_EDGE = '1' } else { Remove-Item env:COULOMB_EDGE -ErrorAction SilentlyContinue }
    if ($v.dipEdge) { $env:EDGE_DIPOLE_FEATURES = '1' } else { Remove-Item env:EDGE_DIPOLE_FEATURES -ErrorAction SilentlyContinue }
    python 03_create_datasets.py --feat-dir $feat --out-dir $out *> "data/03_$($v.name)_138.log"
    Get-Content "data/03_$($v.name)_138.log" | Select-String 'input_dim' | Select-Object -First 1 | ForEach-Object { Write-Host "  $_" }
    Remove-Item env:DROP_FEATURE_COLS -ErrorAction SilentlyContinue
}
Remove-Item env:COULOMB_EDGE -ErrorAction SilentlyContinue
Remove-Item env:EDGE_DIPOLE_FEATURES -ErrorAction SilentlyContinue

# 2) Train: 6 variants x 8 seeds = 48 trainings
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
$results | Export-Csv -Path "$resBase/sweep_electrostatics_138.csv" -NoTypeInformation
Write-Host "`nWrote $resBase/sweep_electrostatics_138.csv"
