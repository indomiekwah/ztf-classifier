"""Small ALeRCE TAP client for one-object local inference."""

from __future__ import annotations

import random

import pandas as pd
import pyvo

TAP_URL = "https://tap.alerce.online/tap"
DETECTION_COLUMNS = (
    "oid, candid, mjd, fid, magpsf, sigmapsf, magpsf_corr, sigmapsf_corr, "
    "corrected"
)


class AlerceTapClient:
    def __init__(self, tap_url: str = TAP_URL):
        self.service = pyvo.dal.TAPService(tap_url)

    def _query(self, query: str, maxrec: int) -> pd.DataFrame:
        try:
            return self.service.run_sync(query, maxrec=maxrec).to_table().to_pandas()
        except Exception as exc:
            raise RuntimeError(f"Query TAP gagal: {exc}") from exc

    @staticmethod
    def _literal(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    def fetch_object(self, oid: str) -> pd.DataFrame:
        safe_oid = self._literal(oid)
        query = f"""
            SELECT oid, ndet, meanra, meandec, firstmjd, lastmjd
            FROM ztf.object
            WHERE oid = {safe_oid}
        """
        return self._query(query, maxrec=2)

    def fetch_random_oid(self) -> str | None:
        """Sample an OID from ztf.object without sorting the full table."""
        prefixes = [f"ZTF{year}" for year in range(17, 26)]
        for _ in range(3):
            prefix = random.choice(prefixes)
            safe_prefix = self._literal(f"{prefix}%")
            query = f"""
            SELECT TOP 1 oid
            FROM ztf.object
            WHERE oid LIKE {safe_prefix}
            """
            objects = self._query(query, maxrec=1)
            values = objects["oid"].dropna().astype(str).tolist()
            if values:
                return values[0]
        return None

    def fetch_detections(self, oid: str) -> pd.DataFrame:
        safe_oid = self._literal(oid)
        query = f"""
            SELECT {DETECTION_COLUMNS}
            FROM ztf.detection
            WHERE oid = {safe_oid}
            ORDER BY mjd ASC, candid ASC
        """
        return self._query(query, maxrec=10000)
