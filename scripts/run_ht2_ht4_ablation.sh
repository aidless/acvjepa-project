#!/usr/bin/env bash
# H-T2/H-T4 预注册消融执行脚本
# 预注册: 决策记录 2026-08-15; HYPOTHESES H-T2/H-T4 (preregistered)
# H-T2: head 宽度 32/64/128/256 x seeds; 判据 = 边际收益比 Δ(128->256)/Δ(32->64) < 0.2 且 d<0.2 判饱和 (饱和点 <768 即支持)
# H-T4: ema vs sync x seeds (固定 head 64); 判据 = |Δ| = |mean_A - mean_B|/mean_B < 5% 支持
# 指标口径: 一步潜在预测误差 loss_latent_nll (训练日志 JSON 行)
#
# 用法:
#   bash scripts/run_ht2_ht4_ablation.sh <repo_root> <manifest> <weights_safetensors> <out_root> [epochs] [seeds]
# 环境变量: BATCH_SIZE (默认 2, 共享实例显存受限时用), CONDA_ENV (默认 acvjepa)
set -euo pipefail

REPO="$1"; MANIFEST="$2"; WEIGHTS="$3"; OUT="$4"; EPOCHS="${5:-30}"; SEEDS="${6:-2026 2027 2028}"
BATCH_SIZE="${BATCH_SIZE:-2}"
CONDA_ENV="${CONDA_ENV:-acvjepa}"

if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

mkdir -p "$OUT"
SUMMARY="$OUT/summary.tsv"
[ -f "$SUMMARY" ] || echo -e "tag\tfinal_latent_nll\twindows" > "$SUMMARY"

run_one () {
  local tag="$1"; shift
  local log="$OUT/$tag.log"
  echo "[$(date +%H:%M:%S)] START $tag"
  if python "$REPO/train_p1_domain_adapt.py" \
      --manifest "$MANIFEST" --output "$OUT/$tag" --epochs "$EPOCHS" \
      --init-from "vjepa2hf:$WEIGHTS:frozen" --init-img-size 384 \
      --per-rank-batch-size "$BATCH_SIZE" --gradient-accumulation 4 \
      "$@" > "$log" 2>&1; then
    local nll wins
    # 首行可能是 [vjepa_backbone] 非 JSON 行; 用 JSON header 行定位 windows
    nll=$(grep '"loss_latent_nll"' "$log" | tail -1 | python -c "import sys,json;print(json.loads(sys.stdin.read())['loss_latent_nll'])" 2>/dev/null || echo NA)
    wins=$(grep '"mode": "p1_domain_adapt"' "$log" | head -1 | python -c "import sys,json;print(json.loads(sys.stdin.read())['windows'])" 2>/dev/null || echo NA)
    echo -e "$tag\t$nll\t$wins" >> "$SUMMARY"
    echo "[$(date +%H:%M:%S)] DONE $tag nll=$nll"
  else
    echo -e "$tag\tFAIL\tNA" >> "$SUMMARY"
    echo "[$(date +%H:%M:%S)] FAIL $tag (see $log)"
  fi
}

# H-T2: 投影头容量消融 (latent_dim = head 宽度)
for wd in 32 64 128 256; do
  for s in $SEEDS; do
    run_one "ht2_w${wd}_s${s}" --latent-dim "$wd" --seed "$s"
  done
done

# H-T4: EMA vs 同步目标 A/B (固定 head 64)
for arm in ema sync; do
  for s in $SEEDS; do
    run_one "ht4_${arm}_s${s}" --latent-dim 64 --ema-target "$arm" --seed "$s"
  done
done

echo "ALL DONE -> $SUMMARY"
cat "$SUMMARY"
