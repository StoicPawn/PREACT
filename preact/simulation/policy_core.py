"""Compatibility layer exposing the policy core under the documented name.

The Markdown guides reference a ``policy_core`` module.  The functional
implementation lives in :mod:`preact.simulation.policy`; this shim simply
re-exports the public API so that both import paths are valid.
"""

from .policy import PolicyCore, PolicyParameters, TaxBracket

__all__ = ["PolicyCore", "PolicyParameters", "TaxBracket"]
