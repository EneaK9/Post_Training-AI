"""Evaluator integration: the Meta client protocol, the fake, campaign structure, shipping,
insights and comment sync.

Production code depends only on `MetaClient` (meta.base). The real client (Phase 4) and the
fake (meta.fake) both implement it. The real client is constructed only when DRY_RUN is
false, the kill switch is open, and the account is active.
"""
