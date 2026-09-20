"""Google News RSS adapter for the shared provider gateway."""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlparse

from .gateway import ProviderResponse, SharedProviderGateway

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
MEDIA_NS = "http://search.yahoo.com/mrss/"
TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(value: str | None) -> str | None:
    if not value:
        return None
    text = html.unescape(TAG_RE.sub(" ", value))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:800] or None


def parse_google_news_rss(
    payload: str,
    *,
    max_records: int = 60,
    language: str | None = None,
) -> list[dict[str, Any]]:
    root = ET.fromstring(payload)
    articles: list[dict[str, Any]] = []
    for item in root.findall("./channel/item")[: max(1, int(max_records))]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        if not title or not link:
            continue

        published_raw = (item.findtext("pubDate") or "").strip()
        published = None
        if published_raw:
            try:
                stamp = parsedate_to_datetime(published_raw)
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=timezone.utc)
                published = stamp.astimezone(timezone.utc).isoformat()
            except (TypeError, ValueError, OverflowError):
                published = published_raw

        source_node = item.find("source")
        publisher = (source_node.text or "").strip() if source_node is not None else ""
        source_url = (source_node.attrib.get("url") or "").strip() if source_node is not None else ""
        domain = (
            urlparse(source_url).netloc.lower().removeprefix("www.")
            if source_url
            else publisher
        )

        image_url = None
        for tag in (f"{{{MEDIA_NS}}}content", f"{{{MEDIA_NS}}}thumbnail"):
            media = item.find(tag)
            if media is not None and media.attrib.get("url"):
                image_url = media.attrib["url"].strip() or None
                if image_url:
                    break

        articles.append(
            {
                "url": link,
                "title": title,
                "seendate": published,
                "domain": domain or publisher or "news.google.com",
                "publisher": publisher or domain,
                "language": language,
                "socialimage": image_url,
                "snippet": _strip_html(item.findtext("description")),
            }
        )
    return articles


def google_news_search(
    gateway: SharedProviderGateway,
    *,
    query: str,
    hl: str,
    gl: str,
    ceid: str,
    max_records: int = 60,
    ttl_seconds: int = 900,
) -> ProviderResponse:
    result = gateway.get_text(
        source_id="google_news_rss",
        operation="rss_search",
        url=GOOGLE_NEWS_RSS,
        params={"q": query, "hl": hl, "gl": gl, "ceid": ceid},
        ttl_seconds=max(0, int(ttl_seconds)),
        minimum_interval_seconds=0.25,
        timeout_seconds=30.0,
    )
    language = hl.split("-", 1)[0] if hl else None
    articles = parse_google_news_rss(
        str(result.payload),
        max_records=max_records,
        language=language,
    )
    return ProviderResponse(
        source_id=result.source_id,
        operation=result.operation,
        payload={"articles": articles},
        retrieved_at=result.retrieved_at,
        cached=result.cached,
        request_fingerprint=result.request_fingerprint,
        snapshot_checksum=result.snapshot_checksum,
    )
