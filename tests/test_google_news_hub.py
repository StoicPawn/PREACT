from datetime import datetime, timezone

from preact.data_hub.google_news import parse_google_news_rss


def test_google_news_rss_parser_normalizes_metadata() -> None:
    xml = """<?xml version="1.0"?>
    <rss version="2.0">
      <channel>
        <item>
          <title>Example headline</title>
          <link>https://news.google.com/rss/articles/x</link>
          <pubDate>Sun, 20 Sep 2026 06:00:00 GMT</pubDate>
          <source url="https://example.com">Example News</source>
          <description><![CDATA[<b>Summary</b> text]]></description>
        </item>
      </channel>
    </rss>
    """
    rows = parse_google_news_rss(xml, language="en")
    assert len(rows) == 1
    assert rows[0]["title"] == "Example headline"
    assert rows[0]["publisher"] == "Example News"
    assert rows[0]["domain"] == "example.com"
    assert rows[0]["snippet"] == "Summary text"
    assert rows[0]["language"] == "en"
