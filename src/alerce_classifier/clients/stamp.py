"""Optional ALeRCE stamp retrieval for Streamlit visualization."""

from __future__ import annotations

from typing import Any


class AlerceStampClient:
    def __init__(self) -> None:
        try:
            from alerce.core import Alerce
        except ImportError as exc:
            raise RuntimeError(
                "Dependency 'alerce' belum terpasang; stamp tidak tersedia."
            ) from exc
        self.client = Alerce()

    def get_stamps(self, oid: str, candid: int) -> Any:
        try:
            return self.client.get_stamps(
                oid=oid, candid=int(candid), survey="ztf", format="numpy"
            )
        except Exception as exc:
            raise RuntimeError(f"Pengambilan stamp gagal: {exc}") from exc
