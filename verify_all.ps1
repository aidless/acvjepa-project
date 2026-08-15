# verify_all.ps1 — AC-VJEPA project offline verification baseline (takeover 2026-08-15)
# Runs every self-contained smoke test / unit test / config validator that can run
# on this machine (CPU, 1x RTX 3060 Laptop 6GB, Docker CLI). Cluster/RDMA/K8s items
# are recorded as SKIP/BLOCKED in VERIFY_RESULTS.md, not attempted here.
$ErrorActionPreference = 'Continue'
$root = 'F:\deepseek\acvjepa-project'
Set-Location $root

$artifacts = Join-Path $root 'verify_artifacts'
if (-not (Test-Path $artifacts)) { New-Item -ItemType Directory -Path $artifacts | Out-Null }
$summary = Join-Path $artifacts '_summary.tsv'
"STATUS`tCHECK`tNOTE" | Set-Content $summary -Encoding UTF8

function Run-Check {
    param(
        [string]$Name,
        [string[]]$ArgList,
        [int]$TimeoutSec = 300,
        [string]$WorkDir = $root,
        [string]$SkipReason = '',
        [string]$FilePath = 'python'
    )
    $safe = ($Name -replace '[^\w\-\.]', '_')
    $outFile = Join-Path $artifacts "$safe.out.txt"
    $errFile = Join-Path $artifacts "$safe.err.txt"
    if ($SkipReason -ne '') {
        "SKIP`t$Name`t$SkipReason" | Add-Content $summary -Encoding UTF8
        Write-Host "[SKIP] $Name -- $SkipReason"
        return
    }
    Write-Host "[RUN ] $Name"
    Push-Location $WorkDir
    try {
        & $FilePath @ArgList *> $outFile
        $code = $LASTEXITCODE
    } catch {
        $code = -1
        $_ | Out-String | Add-Content $outFile
    } finally {
        Pop-Location
    }
    if ($code -eq 0) {
        "PASS`t$Name" | Add-Content $summary -Encoding UTF8
        Write-Host "[PASS] $Name"
    } else {
        "FAIL`t$Name`texit=$code" | Add-Content $summary -Encoding UTF8
        Write-Host "[FAIL] $Name (exit=$code)"
    }
}

# --- 0. monitoring dir expected by update_production_dashboard.py -----------------
if (-not (Test-Path 'monitoring')) { New-Item -ItemType Directory -Path 'monitoring' | Out-Null }
if (-not (Test-Path 'monitoring/grafana_acvjepa_elastic_dashboard.json')) {
    Copy-Item 'grafana_acvjepa_elastic_dashboard.json' 'monitoring/grafana_acvjepa_elastic_dashboard.json' -Force
}

# --- A. unit tests (unittest runner; avoids broken deepeval pytest plugin) --------
Run-Check 'unit.test_heterogeneous_microbatch_failpoints' @('-m','unittest','-v','test_heterogeneous_microbatch_failpoints.py')
Run-Check 'unit.test_shadow_canary_gate' @('-m','unittest','-v','test_shadow_canary_gate.py')
Run-Check 'unit.test_shadow_rca_and_pointcloud_pipeline' @('-m','unittest','-v','test_shadow_rca_and_pointcloud_pipeline.py')
Run-Check 'unit.test_resumable_ledger_and_training_input' @('-m','unittest','-v','test_resumable_ledger_and_training_input.py')
Run-Check 'unit.ac_vjepa_fault_injection_tests' @('-m','unittest','-v','ac_vjepa_fault_injection_tests.py')

# --- B. standalone smoke tests -----------------------------------------------------
# NOTE: ac_vjepa_core smoke is run on CPU (CUDA_VISIBLE_DEVICES=-1) because the
# original delivery was validated on CPU PyTorch and this laptop's CUDA runtime
# reports "unknown error" for this model's ops; logic is identical.
# (torch 2.5.1 on Windows ignores CUDA_VISIBLE_DEVICES='' so use an invalid id.)
$env:CUDA_VISIBLE_DEVICES = '-1'
Run-Check 'smoke.ac_vjepa_core' @('ac_vjepa_core.py')
Run-Check 'smoke.elastic_data_cursor_ledger' @('elastic_data_cursor_ledger.py')
Run-Check 'smoke.verified_checkpoint_cache' @('verified_checkpoint_cache.py')
Run-Check 'smoke.threadsafe_checkpoint_load_gate' @('threadsafe_checkpoint_load_gate.py')
Run-Check 'smoke.multimodal_hitl_tamper_evident_ledger' @('multimodal_hitl_tamper_evident_ledger.py')
Run-Check 'smoke.rapid_recovery_alert_drill' @('rapid_recovery_alert_drill.py')
Run-Check 'smoke.mixed_precision_elastic_recovery' @('mixed_precision_elastic_recovery.py')
Run-Check 'smoke.rdma_rail_chaos_guard' @('rdma_rail_chaos_guard.py')
Run-Check 'smoke.recovery_deployment_arbiter' @('recovery_deployment_arbiter.py')
Run-Check 'smoke.heterogeneous_microbatch_chaos_framework' @('heterogeneous_microbatch_chaos_framework.py')
Run-Check 'smoke.checkpoint_cache_load_shedding_simulator' @('checkpoint_cache_load_shedding_simulator.py')
Run-Check 'smoke.checkpoint_integrity_corruption_demo' @('checkpoint_integrity_corruption_demo.py')
Run-Check 'smoke.distributed_training_observability' @('distributed_training_observability.py')
Run-Check 'smoke.run_failpoint_observability_drill' @('run_failpoint_observability_drill.py','--report','verify_artifacts/failpoint_drill_report.json','--metrics','verify_artifacts/failpoint_metrics.prom')
Run-Check 'smoke.dr_policy_tuner' @('dr_policy_tuner.py')
Run-Check 'smoke.spsc_robot_pipeline' @('spsc_robot_pipeline.py')
Run-Check 'smoke.shadow_canary_gate' @('shadow_canary_gate.py')
Run-Check 'smoke.shadow_degradation_rca' @('shadow_degradation_rca.py')
Run-Check 'smoke.hitl_quarantine_review' @('hitl_quarantine_review.py')
Run-Check 'smoke.sim2real_hard_example_compiler' @('sim2real_hard_example_compiler.py','--demo','--output','verify_artifacts/sim2real_demo_out')
Run-Check 'smoke.sim2real_pointcloud_video_pipeline' @('sim2real_pointcloud_video_pipeline.py','--demo','--output','verify_artifacts/pc_demo_out')

# --- C. --smoke-test modules --------------------------------------------------------
Run-Check 'smoke.dynamic_nccl_update_plan_train' @('dynamic_nccl_update_plan_train.py','--smoke-test')

# --- D. config validators ------------------------------------------------------------
Run-Check 'validate.monitoring_config' @('validate_monitoring_config.py')
Run-Check 'validate.local_compose' @('validate_local_compose.py')
Run-Check 'validate.kubernetes_chaos_lab' @('validate_kubernetes_chaos_lab.py')
Run-Check 'validate.kubernetes_chaos_ci' @('validate_kubernetes_chaos_ci.py')
Run-Check 'validate.failpoint_ci_config' @('validate_failpoint_ci_config.py')
Run-Check 'validate.update_production_dashboard' @('update_production_dashboard.py')

# --- E. Gloo CPU multi-process semantic regression (manual 2-proc runner) -----------
# torchrun's elastic agent holds an extra full torch import -> WinError 1455 (pagefile)
# on this machine. scripts/manual_gloo_runner.py spawns two plain python processes
# (RANK/WORLD_SIZE env, gloo backend, USE_LIBUV=0 for the Windows wheel), which fits
# in the available commit budget. Same two-process semantic regression as on Linux.
$env:USE_LIBUV = '0'
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:PYTORCH_NO_CUDA = '1'
$demoOut = 'verify_artifacts/demo_ddp_data'
Run-Check 'ddp.make_demo_data' @('make_demo_ddp_data.py') 
$manifest = 'F:\home\ubuntu\lecun_analysis\demo_ddp_data\manifest.jsonl'
if (Test-Path $manifest) {
    Run-Check 'ddp.train_ac_vjepa_gloo_2proc' @('scripts/manual_gloo_runner.py','train_ac_vjepa_ddp.py','--manifest',$manifest,'--output','verify_artifacts/ddp_train_out','--epochs','1','--gradient-accumulation','1') -TimeoutSec 600
    Run-Check 'ddp.topology_aware_update_plan_2proc' @('scripts/manual_gloo_runner.py','topology_aware_update_plan.py','--smoke-test') -TimeoutSec 600
    Run-Check 'ddp.test_dynamic_nccl_full_state_equivalence' @('scripts/manual_gloo_runner.py','test_dynamic_nccl_full_state_equivalence.py') -TimeoutSec 600
    Run-Check 'ddp.test_dynamic_nccl_acvjepa_integration' @('scripts/manual_gloo_runner.py','test_dynamic_nccl_acvjepa_integration.py') -TimeoutSec 600
} else {
    "SKIP`tddp.*`tmanifest not produced (demo data path differs on Windows)" | Add-Content $summary -Encoding UTF8
    Write-Host "[SKIP] ddp.* (manifest missing: $manifest)"
}

# --- F. Docker static + optional local demo -------------------------------------------
$dockerInfo = docker info 2>&1 | Out-String
if ($LASTEXITCODE -eq 0) {
    Run-Check 'docker.compose_config' @('compose','-f','docker-compose.local-chaos.yml','config','-q') -FilePath 'docker'
    Run-Check 'docker.local_chaos_demo' @('compose','-f','docker-compose.local-chaos.yml','up','-d','--wait') -TimeoutSec 300 -FilePath 'docker'
    if ($LASTEXITCODE -eq 0) {
        docker compose -f docker-compose.local-chaos.yml down 2>&1 | Out-Null
    }
} else {
    "SKIP`tdocker.*`tdaemon unavailable: $($dockerInfo.Trim().Split("`n")[0])" | Add-Content $summary -Encoding UTF8
    Write-Host "[SKIP] docker.* (daemon unavailable)"
}

# --- G. project's own offline contract shell script (best effort via git-bash) --------
Run-Check 'contract.offline_chaos_contract_sh' @('scripts/run_offline_chaos_contract.sh') -TimeoutSec 600 -FilePath 'bash'

# --- H. Real V-JEPA2 HF weights -> training entry (single proc, CPU) -----------------
# Requires the downloaded safetensors (weights/vjepa2.1-vitb-fpc64-384/model.safetensors,
# see DATA_MANIFEST.md A-layer). Runs the actual `--init-from vjepa2hf:` path through
# train_ac_vjepa_ddp.py: 384px demo windows, frozen backbone, 1 epoch, then checks the
# checkpoint file landed. SKIPs when weights are absent.
$hfCkpt = Join-Path $root 'weights\vjepa2.1-vitb-fpc64-384\model.safetensors'
if (Test-Path $hfCkpt) {
    $env:USE_LIBUV = '0'
    $env:CUDA_VISIBLE_DEVICES = '-1'
    $env:PYTORCH_NO_CUDA = '1'
    $env:RANK = '0'
    $env:LOCAL_RANK = '0'
    $env:WORLD_SIZE = '1'
    $env:MASTER_ADDR = '127.0.0.1'
    $env:MASTER_PORT = '29701'
    $hfDemo = 'verify_artifacts/hf_demo'
    $hfOut = 'verify_artifacts/hf_train_out'
    Run-Check 'hf.make_demo_data_384' @('make_demo_ddp_data.py','--img-size','384','--root',(Join-Path $root $hfDemo))
    $hfManifest = Join-Path $root (Join-Path $hfDemo 'manifest.jsonl')
    if (Test-Path $hfManifest) {
        Run-Check 'hf.train_vjepa2hf_frozen_1proc' @('train_ac_vjepa_ddp.py','--manifest',$hfManifest,'--output',(Join-Path $root $hfOut),'--epochs','1','--per-rank-batch-size','1','--gradient-accumulation','1','--num-workers','0','--init-from',"vjepa2hf:${hfCkpt}:frozen",'--init-img-size','384','--latent-dim','64') -TimeoutSec 600
        $hfLast = Join-Path $root (Join-Path $hfOut 'last.pt')
        if (Test-Path $hfLast) {
            "PASS`thf.checkpoint_saved`tlast.pt exists ($([math]::Round((Get-Item $hfLast).Length/1MB,1)) MB)" | Add-Content $summary -Encoding UTF8
            Write-Host "[PASS] hf.checkpoint_saved"
        } else {
            "FAIL`thf.checkpoint_saved`tlast.pt missing" | Add-Content $summary -Encoding UTF8
            Write-Host "[FAIL] hf.checkpoint_saved (last.pt missing)"
        }
        Remove-Item (Join-Path $root $hfOut) -Recurse -Force -ErrorAction SilentlyContinue
    }
    Remove-Item (Join-Path $root $hfDemo) -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item Env:RANK,Env:LOCAL_RANK,Env:WORLD_SIZE,Env:MASTER_ADDR,Env:MASTER_PORT -ErrorAction SilentlyContinue
} else {
    "SKIP`thf.*`tweights not present: $hfCkpt" | Add-Content $summary -Encoding UTF8
    Write-Host "[SKIP] hf.* (weights missing)"
}

Write-Host ''
Write-Host '=== SUMMARY ==='
Get-Content $summary
Write-Host "Artifacts: $artifacts"
