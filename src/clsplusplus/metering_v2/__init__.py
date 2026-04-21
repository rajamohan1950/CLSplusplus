"""CLS++ Metering v2 — append-only event log + pay-as-you-go pricing.

Step 1 of the rollout defined in docs/adr/0001-metering-data-lake.md.
Only the schema bootstrapper lives here today. No writers, no readers.
"""

from clsplusplus.metering_v2.schema import (
    apply_if_enabled,
    apply_schema,
    drop_schema,
    read_ddl,
)

__all__ = ["apply_if_enabled", "apply_schema", "drop_schema", "read_ddl"]
