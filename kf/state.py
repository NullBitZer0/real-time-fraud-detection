"""Static velocity cache — placeholder for Phase 5.

Loads the last 7 days of `fraudTrain.csv` at startup and indexes by
`cc_num`. The consumer looks up the most recent transactions for a given
card and overrides the zero-filled velocity features with real values.

Until Phase 5, the consumer leaves velocity = 0 and the cache is a
no-op pass-through.
"""
import os
import sys
import time
from collections import defaultdict, deque
from typing import Optional

import pandas as pd

from src.utils.logger import logging
from src.utils.exception import CustomException


class VelocityState:

    def __init__(self, history_path: str = "data/raw/fraudTrain.csv",
                       lookback_hours: int = 168,        # 7 days
                       window_hours: tuple = (1, 24, 168)):
        self.history_path   = history_path
        self.lookback_hours = lookback_hours
        self.windows        = window_hours
        # cc_num -> deque of (timestamp, amount) sorted ascending
        self._cache: dict[int, deque] = defaultdict(deque)
        self._loaded = False

    def load(self) -> None:
        """Index the last `lookback_hours` of fraudTrain by cc_num."""
        if self._loaded:
            return
        if not os.path.exists(self.history_path):
            logging.warning(f"Velocity history not found: {self.history_path} — velocity stays 0")
            return
        logging.info(f"Loading velocity history from {self.history_path}")
        df = pd.read_csv(self.history_path)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
        cutoff = df["trans_date_trans_time"].max() - pd.Timedelta(hours=self.lookback_hours)
        df = df[df["trans_date_trans_time"] >= cutoff]

        for _, row in df.iterrows():
            self._cache[int(row["cc_num"])].append((row["trans_date_trans_time"], float(row["amt"])))
        logging.info(f"Velocity cache ready: {len(self._cache):,} cards indexed")
        self._loaded = True

    def lookup(self, cc_num: int, current_ts: pd.Timestamp) -> dict:
        """Return velocity features for `cc_num` at time `current_ts`.

        Returns:
            dict with keys: txn_last_1h, txn_last_24h, txn_last_168h,
                             amt_sum_last_1h, amt_sum_last_24h, amt_sum_last_168h
        """
        if not self._loaded:
            self.load()
        history = self._cache.get(int(cc_num), deque())
        counts  = {w: 0 for w in self.windows}
        sums    = {w: 0.0 for w in self.windows}
        for ts, amt in history:
            if ts >= current_ts:
                continue
            hours_ago = (current_ts - ts).total_seconds() / 3600.0
            for w in self.windows:
                if hours_ago <= w:
                    counts[w] += 1
                    sums[w]   += amt
        return {
            "txn_last_1h":   counts[1],
            "txn_last_24h":  counts[24],
            "txn_last_168h": counts[168],
            "amt_sum_last_1h":   sums[1],
            "amt_sum_last_24h":  sums[24],
            "amt_sum_last_168h": sums[168],
        }
