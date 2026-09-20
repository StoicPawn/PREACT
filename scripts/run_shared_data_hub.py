"""Run the shared provider gateway HTTP service."""

from __future__ import annotations

import os

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "preact.data_hub.api:app",
        host=os.getenv("SHARED_DATA_HUB_HOST", "127.0.0.1"),
        port=int(os.getenv("SHARED_DATA_HUB_PORT", "8787")),
        workers=1,
    )
