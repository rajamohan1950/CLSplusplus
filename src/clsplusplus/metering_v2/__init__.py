"""CLS++ Metering v2 — append-only event log + pay-as-you-go pricing.

Rollout defined in docs/adr/0001-metering-data-lake.md. Steps landed:

    step 1   — schema + feature flag (merged)
    step 2   — dual-write + dead-letter notifier (merged)
    step 2.5 — pay-as-you-go pricer (merged)
    step 3   — daily reconciliation vs Redis (this package)
"""

from clsplusplus.metering_v2.notifier import MeteringNotifier
from clsplusplus.metering_v2.pricing import MeteringPricer, compute_unit_cost_cents
from clsplusplus.metering_v2.reconciler import (
    DriftFinding,
    MeteringReconciler,
    ReconciliationResult,
)
from clsplusplus.metering_v2.schema import (
    apply_if_enabled,
    apply_schema,
    drop_schema,
    read_ddl,
)
from clsplusplus.metering_v2.writer import MeteringWriter, UsageEvent, VALID_ACTOR_KINDS

__all__ = [
    "apply_if_enabled",
    "apply_schema",
    "drop_schema",
    "read_ddl",
    "MeteringWriter",
    "MeteringNotifier",
    "MeteringPricer",
    "MeteringReconciler",
    "DriftFinding",
    "ReconciliationResult",
    "UsageEvent",
    "VALID_ACTOR_KINDS",
    "compute_unit_cost_cents",
]
