"""
manager.py — AdaptiveManager（MAPE-K 自适应控制器）

职责：
  1. 日志记录（system_metrics.csv + DuckDB 双写）
  2. 模型决策（滑动窗口 + 时间衰减 + 冷却滞后）
  3. 成本计算与资源监控

参考：DESIGN.md §4 / §11.1 / §16
"""

import csv
import math
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    COST_LIMIT_USD,
    CSV_COLUMNS,
    FEEDBACK_COLUMNS,
    FEEDBACK_FILE,
    LOG_FILE,
    METRIC_WEIGHTS,
    MODEL_COOLDOWN_RUNS,
    MODEL_TIERS,
    PRICING_TABLE,
    RESOURCE_MODERATE,
    RESOURCE_SCARCE,
    SATISFACTION_ACCEPTABLE,
    SATISFACTION_EXCELLENT,
    SATISFACTION_GOOD,
    SATISFACTION_POOR,
    SWITCH_MARGIN,
    TIME_DECAY_LAMBDA,
    WINDOW_SIZE,
)

from db import get_conn, generate_report_id, insert_report, insert_eval


class AdaptiveManager:
    """自适应模型管理器 — 基于 Objective Metrics 驱动 MAPE-K 闭环。"""

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def __init__(self) -> None:
        self._ensure_csv(LOG_FILE, CSV_COLUMNS)
        self._ensure_csv(FEEDBACK_FILE, FEEDBACK_COLUMNS)

    @staticmethod
    def _ensure_csv(path: str, columns: list) -> None:
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(columns)

    # ------------------------------------------------------------------
    # 成本计算
    # ------------------------------------------------------------------
    def calculate_cost(
        self, model_name: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        clean = model_name.replace("gemini/", "")
        info = PRICING_TABLE.get(clean, {})
        if not info:
            return 0.0

        if clean == "gemini-2.5-pro":
            rate = (
                info["input_low"]
                if prompt_tokens <= info["cutoff"]
                else info["input_high"]
            )
            input_cost = (prompt_tokens / 1_000_000) * rate
        else:
            input_cost = (prompt_tokens / 1_000_000) * info.get("input", 0)

        output_cost = (completion_tokens / 1_000_000) * info.get("output", 0)
        return round(input_cost + output_cost, 6)

    # ------------------------------------------------------------------
    # 日志写入（主轨）
    # ------------------------------------------------------------------
    def log_execution(
        self,
        model_used: str,
        latency: float,
        total_tokens: int,
        cost: float,
        report_path: str,
        transcript_path: str = "",
        factual_consistency: Optional[float] = None,
        completeness: Optional[float] = None,
        coherence: Optional[float] = None,
        chat_depth: int = 0,
        re_asks: int = 0,
        copy_clicks: int = 0,
        expand_clicks: int = 0,
        time_spent_sec: float = 0.0,
        user_feedback_score: Optional[float] = None,
        summary_preview: str = "",
        rag_info: Optional[dict] = None,
    ) -> Optional[str]:
        """写入 system_metrics.csv + DuckDB（双写）。返回 report_id。"""
        weighted = self.calculate_weighted_score(
            factual_consistency, completeness, coherence
        )
        is_fb = 1 if user_feedback_score is not None else 0

        # ---------- CSV 写入（兼容旧流程） ----------
        try:
            with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    datetime.now().isoformat(),
                    model_used,
                    round(latency, 2),
                    total_tokens,
                    round(cost, 6),
                    report_path,
                    transcript_path,
                    factual_consistency,
                    completeness,
                    coherence,
                    weighted,
                    chat_depth,
                    re_asks,
                    copy_clicks,
                    expand_clicks,
                    round(time_spent_sec, 1),
                    is_fb,
                    user_feedback_score,
                ])
            print(f"✅ Logged to CSV. Run count: {self.get_run_count()}")
        except Exception as e:
            print(f"Error logging CSV: {e}")

        # ---------- DuckDB 写入 ----------
        report_id = None
        try:
            report_id = generate_report_id()
            conn = get_conn()
            insert_report(
                conn,
                report_id=report_id,
                model_used=model_used,
                latency_sec=latency,
                total_tokens=total_tokens,
                cost_usd=cost,
                report_path=report_path,
                transcript_path=transcript_path,
                summary_preview=summary_preview,
            )
            # 提取 RAG 信息
            _rag = rag_info or {}
            insert_eval(
                conn,
                report_id=report_id,
                model_used=model_used,
                factual_consistency=factual_consistency,
                completeness=completeness,
                coherence=coherence,
                objective_weighted_score=weighted,
                chat_depth=chat_depth,
                re_asks=re_asks,
                copy_clicks=copy_clicks,
                expand_clicks=expand_clicks,
                time_spent_sec=time_spent_sec,
                is_feedback_collected=bool(is_fb),
                user_feedback_score=user_feedback_score,
                rag_used=bool(_rag.get("rag_used", False)),
                rag_total_chunks=_rag.get("n_chunks_total"),
                rag_used_chunks=_rag.get("n_chunks_used"),
                rag_coverage_pct=_rag.get("rag_coverage_pct"),
            )
            conn.close()
            print(f"✅ Logged to DuckDB. report_id={report_id[:8]}...")
        except Exception as e:
            print(f"⚠️ DuckDB write failed (CSV still OK): {e}")

        return report_id

    # ------------------------------------------------------------------
    # 反馈写入（副轨）
    # ------------------------------------------------------------------
    def log_feedback(
        self,
        model_used: str,
        report_path: str,
        user_feedback_score: float,
        user_feedback_text: str = "",
        objective_metrics: Optional[dict] = None,
        user_action: str = "neutral",
    ) -> None:
        """写入 feedback_log.csv（DESIGN §3.2）。"""
        om = objective_metrics or {}
        fc = om.get("factual_consistency")
        cp = om.get("completeness")
        ch = om.get("coherence")
        ws = self.calculate_weighted_score(fc, cp, ch)

        try:
            with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    datetime.now().isoformat(),
                    model_used,
                    report_path,
                    user_feedback_score,
                    user_feedback_text,
                    fc, cp, ch, ws,
                    user_action,
                ])
            print("✅ Feedback logged.")
        except Exception as e:
            print(f"Error logging feedback: {e}")

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    def get_run_count(self) -> int:
        if not os.path.exists(LOG_FILE):
            return 0
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                return max(0, sum(1 for _ in f) - 1)
        except Exception:
            return 0

    @staticmethod
    def calculate_weighted_score(
        fc: Optional[float],
        cp: Optional[float],
        ch: Optional[float],
    ) -> Optional[float]:
        """加权计算 objective satisfaction（DESIGN §4.1）。"""
        if all(x is not None for x in (fc, cp, ch)):
            return round(
                METRIC_WEIGHTS["factual_consistency"] * fc
                + METRIC_WEIGHTS["completeness"] * cp
                + METRIC_WEIGHTS["coherence"] * ch,
                4,
            )
        return None

    # ------------------------------------------------------------------
    # 资源状态
    # ------------------------------------------------------------------
    def get_resource_status(self):
        """返回 (remaining_ratio, used_cost, limit)。"""
        if not os.path.exists(LOG_FILE):
            return 1.0, 0.0, COST_LIMIT_USD
        try:
            df = pd.read_csv(LOG_FILE)
            if df.empty:
                return 1.0, 0.0, COST_LIMIT_USD
            total_cost = pd.to_numeric(df["cost_usd"], errors="coerce").sum()
            remaining = max(0.0, 1.0 - total_cost / COST_LIMIT_USD)
            return remaining, total_cost, COST_LIMIT_USD
        except Exception:
            return 1.0, 0.0, COST_LIMIT_USD

    # ------------------------------------------------------------------
    # MAPE-K 决策核心（DESIGN §4.2 + §11.1）
    # ------------------------------------------------------------------
    def decide_model(self) -> str:
        """
        根据滑动窗口 + 时间衰减 satisfaction 与资源约束选择模型。
        包含冷却滞后（Hysteresis）防止震荡。
        """
        print("\n========== MAPE-K Decision ==========")

        # ---------- 无历史 → 默认 lite ----------
        if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) < 50:
            print("No history. Default → Tier 0")
            return MODEL_TIERS[0]

        try:
            df = pd.read_csv(LOG_FILE)
            if df.empty:
                return MODEL_TIERS[0]

            run_count = len(df)

            # ---------- 解析最近模型 ----------
            last_model = str(df["model_used"].iloc[-1]).strip().replace("gemini/", "")
            if last_model not in MODEL_TIERS:
                last_model = MODEL_TIERS[0]
            current_idx = MODEL_TIERS.index(last_model)

            # ---------- 冷却检查（Hysteresis） ----------
            if run_count >= 2:
                recent_models = (
                    df["model_used"]
                    .iloc[-MODEL_COOLDOWN_RUNS:]
                    .str.replace("gemini/", "", regex=False)
                    .str.strip()
                    .tolist()
                )
                if len(set(recent_models)) > 1:
                    # 最近 N 条中发生过切换 → 冷却，维持当前
                    print(
                        f"❄️ COOLDOWN: Model switch detected in last "
                        f"{MODEL_COOLDOWN_RUNS} runs. Maintaining {last_model}"
                    )
                    return last_model

            # ---------- 滑动窗口 + 时间衰减 ----------
            for col in ["factual_consistency", "completeness", "coherence"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            satisfaction = self._compute_windowed_satisfaction(df)
            if satisfaction is None:
                print("⚠️ No valid objective metrics. Default → maintain.")
                return last_model

            # ---------- 资源 ----------
            total_cost = pd.to_numeric(df["cost_usd"], errors="coerce").sum()
            resource_available = max(0.0, 1.0 - total_cost / COST_LIMIT_USD)

            print(
                f"📊 SAT={satisfaction:.3f} | RES={resource_available:.1%} "
                f"| Model={last_model} (idx={current_idx})"
            )

            # ---------- 决策逻辑 ----------
            next_idx = current_idx
            reason = "MAINTAIN"

            if resource_available < RESOURCE_SCARCE:
                # 破产
                if current_idx > 0:
                    next_idx = current_idx - 1
                    reason = "BANKRUPTCY_DOWNGRADE"

            elif resource_available < RESOURCE_MODERATE:
                # 紧张
                if satisfaction <= SATISFACTION_ACCEPTABLE and current_idx < len(MODEL_TIERS) - 1:
                    next_idx = current_idx + 1
                    reason = "SCARCE_UPGRADE"
                elif satisfaction > SATISFACTION_GOOD and current_idx > 0:
                    next_idx = current_idx - 1
                    reason = "SCARCE_DOWNGRADE"
                else:
                    reason = "SCARCE_MAINTAIN"

            else:
                # 充足
                if satisfaction <= SATISFACTION_POOR and current_idx < len(MODEL_TIERS) - 1:
                    next_idx = current_idx + 1
                    reason = "CRITICAL_UPGRADE"
                elif satisfaction <= SATISFACTION_ACCEPTABLE and current_idx < len(MODEL_TIERS) - 1:
                    next_idx = current_idx + 1
                    reason = "POOR_UPGRADE"
                elif satisfaction > SATISFACTION_EXCELLENT and current_idx > 0:
                    next_idx = current_idx - 1
                    reason = "EXCELLENT_DOWNGRADE"
                elif satisfaction > SATISFACTION_GOOD and current_idx > 0:
                    next_idx = current_idx - 1
                    reason = "GOOD_DOWNGRADE"

            # ---------- Switch Margin 保护 ----------
            if next_idx != current_idx:
                if next_idx > current_idx:
                    threshold = SATISFACTION_ACCEPTABLE
                    if satisfaction > threshold - SWITCH_MARGIN:
                        next_idx = current_idx
                        reason = "MARGIN_HOLD"
                elif next_idx < current_idx:
                    threshold = (
                        SATISFACTION_EXCELLENT
                        if resource_available >= RESOURCE_MODERATE
                        else SATISFACTION_GOOD
                    )
                    if satisfaction < threshold + SWITCH_MARGIN:
                        next_idx = current_idx
                        reason = "MARGIN_HOLD"

            new_model = MODEL_TIERS[next_idx]
            print(
                f"🎯 DECISION: {last_model} → {new_model} "
                f"(reason={reason})"
            )
            print("======================================\n")
            return new_model

        except Exception as e:
            import traceback
            print(f"CRITICAL ERROR in Manager: {e}")
            print(traceback.format_exc())
            return MODEL_TIERS[0]

    # ------------------------------------------------------------------
    # 滑动窗口 + 时间衰减（DESIGN §11.1 A/B/D）
    # ------------------------------------------------------------------
    def _compute_windowed_satisfaction(self, df: pd.DataFrame) -> Optional[float]:
        """
        从最近 WINDOW_SIZE 条有效记录计算 time-decayed satisfaction。
        异常值自动降权（|z| > 2.5）。
        """
        needed = ["factual_consistency", "completeness", "coherence"]
        if not all(c in df.columns for c in needed):
            return None

        valid = df.dropna(subset=needed).tail(WINDOW_SIZE)
        if valid.empty:
            return None

        scores = []
        for _, row in valid.iterrows():
            s = (
                METRIC_WEIGHTS["factual_consistency"] * float(row["factual_consistency"])
                + METRIC_WEIGHTS["completeness"] * float(row["completeness"])
                + METRIC_WEIGHTS["coherence"] * float(row["coherence"])
            )
            scores.append(s)

        scores_arr = np.array(scores)
        n = len(scores_arr)

        # 时间衰减权重：最新的权重最高
        decay_weights = np.array([
            math.exp(-TIME_DECAY_LAMBDA * (n - 1 - i)) for i in range(n)
        ])

        # 异常值降权
        if n >= 3:
            median = np.median(scores_arr)
            mad = np.median(np.abs(scores_arr - median)) + 1e-9
            z_scores = np.abs(scores_arr - median) / (mad * 1.4826)
            outlier_mask = z_scores > 2.5
            decay_weights[outlier_mask] *= 0.1  # 降权而非丢弃

        # 归一化加权平均
        total_w = decay_weights.sum()
        if total_w < 1e-12:
            return None

        weighted_sat = float(np.dot(scores_arr, decay_weights) / total_w)
        print(
            f"📈 Window({n}): scores={[round(s, 3) for s in scores_arr]} "
            f"→ weighted_sat={weighted_sat:.4f}"
        )
        return weighted_sat