"""Execution Router package scaffold.

This package will house the IBKR-native execution router that supersedes the
legacy `ExecutionManager`/`PositionManager` stack. Modules introduced here must
remain stateless with respect to execution state, relying on IBKR streams as the
source of truth and emitting normalized lifecycle events for downstream
consumers.
"""

__all__: list[str] = []
