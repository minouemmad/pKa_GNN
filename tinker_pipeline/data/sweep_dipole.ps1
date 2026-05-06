cd C:\Users\maemm\OneDrive\Desktop\FFX\pKa_GNN\tinker_pipeline
$configs = @(
  @{tag='target_dip';      dir='Datasets';       extra=@('--readout','target_atom')},
  @{tag='target_nodip';    dir='Datasets_NoDip'; extra=@('--readout','target_atom')},
  @{tag='2L_mean_dip';     dir='Datasets';       extra=@('--num-layers','2')},
  @{tag='2L_mean_nodip';   dir='Datasets_NoDip'; extra=@('--num-layers','2')},
  @{tag='dropout02_dip';   dir='Datasets';       extra=@('--dropout','0.2')},
  @{tag='dropout02_nodip'; dir='Datasets_NoDip'; extra=@('--dropout','0.2')}
)
foreach ($c in $configs) {
  $resDir = "Graph_pKa\Results\Training_FFX_$($c.tag)"
  $args = @('05_train.py','--dataset','2','--dataset-dir',"Graph_pKa\Features_Paper_FFX\$($c.dir)",'--results-dir',$resDir) + $c.extra
  & python @args *> "data\sweep_$($c.tag).log"
  $line = Get-Content "$resDir\predictions\summary_metrics.csv" | Select-Object -Last 1
  Add-Content "data\sweep_dipole_results.txt" "$($c.tag.PadRight(18)) :: $line"
  Write-Host "$($c.tag.PadRight(18)) :: $line"
}
Write-Host "DONE"
