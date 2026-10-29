"""Champion / Challenger Model Registry.

Promotes a new model to production Champion only if:
1. Out-of-Sample (OOS) Sharpe ratio improves by > 0.15 over the reigning Champion.
2. OOS Max Drawdown does not deteriorate by > 2.0%.
Persists versioned metadata, feature sets, hyperparameters, and evaluation metrics.
"""
from __future__ import annotations
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    model_name: str
    version: str
    feature_list: list[str]
    hyperparameters: dict[str, Any]
    oos_sharpe: float
    oos_brier_score: float
    oos_max_drawdown_pct: float
    training_date_start: str
    training_date_end: str
    created_at: str
    is_champion: bool = False
    notes: str = ""


class ChampionRegistry:
    """Production Model Registry managing Champion and Challenger promotions."""

    def __init__(self, registry_file: str = "models/registry.json"):
        self.registry_path = Path(registry_file)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.history: list[ModelMetadata] = []
        self.champion: Optional[ModelMetadata] = None
        self._load()

    def _load(self) -> None:
        """Load registry from disk if exists."""
        if not self.registry_path.exists():
            return
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.history = [ModelMetadata(**m) for m in data.get("history", [])]
            champ_data = data.get("champion")
            if champ_data:
                self.champion = ModelMetadata(**champ_data)
        except Exception as e:
            logger.error(f"Failed to load registry from {self.registry_path}: {e}")

    def save(self) -> None:
        """Persist registry state to disk."""
        data = {
            "last_updated": datetime.now().isoformat(),
            "champion": asdict(self.champion) if self.champion else None,
            "history": [asdict(m) for m in self.history]
        }
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def evaluate_challenger(
        self,
        model_name: str,
        feature_list: list[str],
        hyperparameters: dict[str, Any],
        oos_sharpe: float,
        oos_brier_score: float,
        oos_max_drawdown_pct: float,
        training_date_start: str,
        training_date_end: str,
        min_sharpe_improvement: float = 0.15,
        max_drawdown_slack_pct: float = 2.0
    ) -> tuple[bool, str, ModelMetadata]:
        """Evaluate whether a challenger model deserves promotion to Champion."""
        version = f"v{len(self.history) + 1}.0"
        challenger = ModelMetadata(
            model_name=model_name,
            version=version,
            feature_list=feature_list,
            hyperparameters=hyperparameters,
            oos_sharpe=round(oos_sharpe, 3),
            oos_brier_score=round(oos_brier_score, 4),
            oos_max_drawdown_pct=round(oos_max_drawdown_pct, 2),
            training_date_start=training_date_start,
            training_date_end=training_date_end,
            created_at=datetime.now().isoformat(),
            is_champion=False
        )

        # First model automatically becomes initial baseline champion
        if self.champion is None:
            challenger.is_champion = True
            challenger.notes = "Initial Champion baseline."
            self.champion = challenger
            self.history.append(challenger)
            self.save()
            return True, "Initial Champion established", challenger

        # Rigorous promotion criteria:
        sharpe_diff = challenger.oos_sharpe - self.champion.oos_sharpe
        dd_diff = challenger.oos_max_drawdown_pct - self.champion.oos_max_drawdown_pct

        sharpe_ok = sharpe_diff >= min_sharpe_improvement
        dd_ok = dd_diff <= max_drawdown_slack_pct  # Max drawdown must not increase by > slack

        if sharpe_ok and dd_ok:
            challenger.is_champion = True
            challenger.notes = (
                f"Promoted to Champion! Sharpe improved by +{sharpe_diff:.2f} "
                f"(vs {self.champion.model_name}:{self.champion.oos_sharpe:.2f}), "
                f"DD diff: {dd_diff:+.2f}%"
            )
            # Demote old champion in history
            self.champion.is_champion = False
            self.champion = challenger
            self.history.append(challenger)
            self.save()
            return True, challenger.notes, challenger
        else:
            reason = (
                f"Rejected: Sharpe diff {sharpe_diff:+.2f} (required >= +{min_sharpe_improvement:.2f}), "
                f"DD diff {dd_diff:+.2f}% (max allowed <= +{max_drawdown_slack_pct:.2f}%)"
            )
            challenger.notes = reason
            self.history.append(challenger)
            self.save()
            return False, reason, challenger
