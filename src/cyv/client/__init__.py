"""Client for the cyvd daemon (thin: stdlib + numpy only)."""

from cyv.client.core import Client, DaemonError

__all__ = ["Client", "DaemonError"]
