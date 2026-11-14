# --- acceptance_backfill (EVAL_ACCEPTANCE_STANDARD.md P4) ---
# 证据来源: failpoint-chaos-ci.yml
# 命令逐字取自该来源，未增删任何参数。
.PHONY: eval

eval:
	ARTIFACT_DIR=artifacts bash scripts/run_offline_chaos_contract.sh
