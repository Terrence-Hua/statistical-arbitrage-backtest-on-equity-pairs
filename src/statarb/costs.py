"""Transaction cost and slippage model.

Separates the cost calculation from the backtest engine so parameters
can be changed without touching signal logic.

The model has two components:
- **Commission**: fixed bps per notional per leg.
- **Slippage**: half-spread cost, also in bps per leg.

Both are applied at each fill (entry and exit).

Usage
-----
    from statarb.costs import CostModel, estimate_round_trip_cost

    model = CostModel(commission_bps=5.0, slippage_bps=3.0)
    cost = model.fill_cost(price_a=150.0, price_b=200.0,
                           hedge_ratio=0.8, notional=100_000)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostModel:
    """Parameters for the transaction cost model.

    Attributes
    ----------
    commission_bps : float
        Commission charged per leg, in basis points (default 5 bps = 0.05 %).
    slippage_bps : float
        One-way bid-ask slippage per leg, in basis points (default 3 bps).
    """

    commission_bps: float = 5.0
    slippage_bps: float = 3.0

    @property
    def total_bps(self) -> float:
        """Combined one-way cost per leg in basis points."""
        return self.commission_bps + self.slippage_bps

    def fill_cost(
        self,
        price_a: float,
        price_b: float,
        hedge_ratio: float,
        notional: float,
    ) -> float:
        """One-way fill cost for a spread position.

        Cost is charged on both legs.  The B leg notional is:
        ``|hedge_ratio| * notional * price_b / price_a`` adjusted to maintain
        approximate dollar-neutrality.

        Parameters
        ----------
        price_a, price_b : float
            Current mid prices of asset A and B.
        hedge_ratio : float
            OLS beta (shares of B per share of A).
        notional : float
            Dollar notional for the A leg.

        Returns
        -------
        float
            One-way cost in dollars.
        """
        frac = self.total_bps / 10_000.0
        cost_a = notional * frac
        notional_b = abs(hedge_ratio) * notional
        cost_b = notional_b * frac
        return cost_a + cost_b

    def round_trip_cost(
        self,
        price_a: float,
        price_b: float,
        hedge_ratio: float,
        notional: float,
    ) -> float:
        """Full round-trip cost (entry + exit) for a spread position.

        Parameters
        ----------
        price_a, price_b : float
            Mid prices (assumed equal at entry and exit for estimation).
        hedge_ratio : float
            OLS beta.
        notional : float
            Dollar notional for the A leg.

        Returns
        -------
        float
            Round-trip cost in dollars.
        """
        return 2.0 * self.fill_cost(price_a, price_b, hedge_ratio, notional)


def estimate_round_trip_cost(
    price_a: float,
    price_b: float,
    hedge_ratio: float,
    notional: float,
    commission_bps: float = 5.0,
    slippage_bps: float = 3.0,
) -> float:
    """Convenience function for round-trip cost estimation.

    Parameters
    ----------
    price_a, price_b : float
        Current mid prices.
    hedge_ratio : float
        OLS beta.
    notional : float
        Dollar notional for the A leg.
    commission_bps : float
        Commission per leg in basis points.
    slippage_bps : float
        One-way slippage per leg in basis points.

    Returns
    -------
    float
        Round-trip cost in dollars.
    """
    model = CostModel(commission_bps=commission_bps, slippage_bps=slippage_bps)
    return model.round_trip_cost(price_a, price_b, hedge_ratio, notional)
