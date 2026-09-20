"""Product-specific projections from shared raw provider data."""

from .gdelt_history import gdelt_article_documents, gdelt_event_records

__all__ = ["gdelt_article_documents", "gdelt_event_records"]
