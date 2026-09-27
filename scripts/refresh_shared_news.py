"""Refresh ACEPC Shared News feeds and project them into the local archive."""

from __future__ import annotations

import json
import os

from preact.data_hub.news_feeds import refresh_shared_news


if __name__ == "__main__":
    root = os.getenv("SHARED_DATA_HUB_ROOT", "data/shared_hub")
    print(json.dumps(refresh_shared_news(root), indent=2, ensure_ascii=False, default=str))
