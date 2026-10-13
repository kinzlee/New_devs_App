"""
Tests that pin down the three bugs reported by Sunset Properties and Ocean Rentals,
and prove the fixes.

Run from the backend/ directory:
    pip install pytest pytest-asyncio
    pytest tests/test_bug_fixes.py -v
"""
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

import pytest


# ---------------------------------------------------------------------------
# Bug 1: Cross-tenant cache leak (Ocean Rentals saw other companies' numbers)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cache_key_is_scoped_per_tenant():
    """
    Two different tenants with the same property_id must NEVER share a cache slot.
    Before the fix the key was 'revenue:{property_id}'; now it is
    'revenue:{tenant_id}:{property_id}'.
    """
    from app.services import cache

    captured_keys = []

    class FakeRedis:
        async def get(self, key):
            captured_keys.append(("get", key))
            return None  # force a miss so we go through to the calc

        async def setex(self, key, ttl, value):
            captured_keys.append(("setex", key))

    with patch.object(cache, "redis_client", FakeRedis()), \
         patch("app.services.reservations.calculate_total_revenue",
               new=AsyncMock(return_value={
                   "property_id": "prop-001", "tenant_id": "t",
                   "total": "100.00", "currency": "USD", "count": 1,
               })):
        await cache.get_revenue_summary("prop-001", "tenant_a")
        await cache.get_revenue_summary("prop-001", "tenant_b")

    keys = [k for _, k in captured_keys]
    # Each tenant's key must contain its own tenant id and must be distinct.
    assert any("tenant_a" in k and "prop-001" in k for k in keys), keys
    assert any("tenant_b" in k and "prop-001" in k for k in keys), keys
    assert len(set(keys)) >= 2, f"tenant_a and tenant_b collided on cache key: {keys}"


# ---------------------------------------------------------------------------
# Bug 2: Off-by-cents (finance team: "totals look slightly off")
# ---------------------------------------------------------------------------

def test_dashboard_source_no_longer_floats_the_total():
    """
    Rather than importing dashboard.py (which drags in the whole auth stack),
    read the source directly and confirm the lossy `float(...)` conversion
    is gone and the Decimal string flows straight through to the response.
    """
    import pathlib
    source = pathlib.Path(__file__).parent.parent.joinpath(
        "app/api/v1/dashboard.py"
    ).read_text()

    assert "float(revenue_data" not in source, (
        "float() still being applied to the revenue total — reintroduces "
        "IEEE-754 drift on money values"
    )
    assert '"total_revenue": revenue_data[\'total\']' in source or \
           '"total_revenue": revenue_data["total"]' in source, (
        "endpoint must pass the Decimal string through unchanged"
    )


def test_float_vs_decimal_demonstrates_the_bug():
    """
    Lock in WHY the fix matters: adding many sub-cent values as floats drifts;
    the same addition as Decimal is exact.
    """
    # 10,000 reservations of $0.1 each should be exactly $1000.
    as_float = sum([0.1] * 10_000)
    as_decimal = sum([Decimal("0.1")] * 10_000)

    assert as_float != 1000.0            # floats drift
    assert as_decimal == Decimal("1000.0")  # Decimal is exact


# ---------------------------------------------------------------------------
# Bug 3: Timezone-dependent monthly totals (Sunset Properties' March mismatch)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_monthly_revenue_window_uses_property_timezone():
    """
    For a Paris property, March must span
        2025-03-01 00:00 Europe/Paris  ->  2025-04-01 00:00 Europe/Paris
    (i.e. 2025-02-28 23:00 UTC -> 2025-03-31 22:00 UTC in winter / 21:00 in summer),
    NOT 2025-03-01 00:00 UTC -> 2025-04-01 00:00 UTC.
    """
    from app.services.reservations import calculate_monthly_revenue

    with patch("builtins.print") as mock_print:
        await calculate_monthly_revenue(
            property_id="prop-001",
            tenant_id="tenant_a",
            month=3,
            year=2025,
            property_timezone="Europe/Paris",
        )

    # The function print()s the start/end datetimes — grab them and assert.
    debug_line = next(
        (args[0] for args, _ in mock_print.call_args_list if "Querying revenue" in str(args[0])),
        "",
    )
    assert "Europe/Paris" in debug_line or "+01:00" in debug_line or "+02:00" in debug_line, (
        f"Expected timezone-aware window, got: {debug_line!r}"
    )


def test_naive_vs_tz_aware_month_boundary():
    """
    Lock in WHY the fix matters: a check-in at 23:30 Paris on March 31 is still
    March locally, but UTC says it's already April 1 at 21:30 — the OLD naive
    window would have miscounted it.
    """
    paris = ZoneInfo("Europe/Paris")
    checkin_local = datetime(2025, 3, 31, 23, 30, tzinfo=paris)
    checkin_utc = checkin_local.astimezone(timezone.utc)

    # Local month is March; UTC instant is still March 31 at 21:30 UTC in summer.
    # Flip to a case that would clearly cross: April 1 at 01:00 Paris:
    cross_local = datetime(2025, 4, 1, 1, 0, tzinfo=paris)
    cross_utc = cross_local.astimezone(timezone.utc)

    # Old (buggy) window end: naive datetime(2025, 4, 1) interpreted as UTC.
    old_end_utc = datetime(2025, 4, 1, tzinfo=timezone.utc)
    # New (fixed) window end: 2025-04-01 00:00 Paris -> 2025-03-31 22:00 UTC in summer.
    new_end_utc = datetime(2025, 4, 1, tzinfo=paris).astimezone(timezone.utc)

    # The late-March Paris check-in belongs in MARCH under both windows:
    assert checkin_utc < old_end_utc
    assert checkin_utc < new_end_utc

    # The April 1 at 01:00 Paris check-in belongs in APRIL locally,
    # but the OLD UTC-based window would wrongly bucket it into March
    # (it's still 2025-03-31 23:00 UTC in summer, i.e. BEFORE old_end_utc):
    assert cross_utc < old_end_utc, "old buggy behavior: puts April check-in in March"
    # The NEW tz-aware window correctly excludes it from March:
    assert cross_utc >= new_end_utc, "fixed behavior: April check-in no longer counted in March"
