"""Transaction cost and slippage model.

Models the round-trip cost of entering and exiting a pairs trade.
Each leg pays commission (fixed bps) plus market-impact slippage
(proportional to order size relative to ADV).

Usage
-----
    from statarb.costs import CostModel

    model = CostModel(commission_bps=5.0, slippage_bps=3.0)
    cost = model.one_way(notional=100_000)
    rt_cost = model.round_trip(notional=100_000)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostModel:
    """Per-trade transaction cost parameters.

    Attributes
    ----------
    commission_bps : float
        Broker commission per leg, in basis points (e.g. 5 = 0.05 %).
    slippage_bps : float
        One-way market-impact slippage per leg, in basis points.
        Approximates half of the effective bid-ask spread plus price impact.
    """

    commission_bps: float = 5.0
    slippage_bps: float = 3.0

    @property
    def total_one_way_bps(self) -> float:
        """Combined one-way cost in basis points."""
        return self.commission_bps + self.slippage_bps

    def one_way(self, notional: float) -> float:
        """One-way dollar cost for a single leg.

        Parameters
        ----------
        notional : float
            Dollar value traded on this leg.

        Returns
        -------
        float
            Cost in dollars.
        """
        return notional * self.total_one_way_bps / 10_000.0

    def round_trip(self, notional: float) -> float:
        """Full round-trip cost for a single leg (entry + exit).

        Parameters
        ----------
        notional : float
            Dollar value traded on this leg.

        Returns
        -------
        float
            Round-trip cost in dollars.
        """
        return 2.0 * self.one_way(notional)

    def pair_one_way(
        self,
        notional_a: float,
        notional_b: float,
    ) -> float:
        """One-way cost for both legs of a pairs trade.

        Parameters
        ----------
        notional_a : float
            Dollar notional on the A leg.
        notional_b : float
            Dollar notional on the B leg (absolute value).

        Returns
        -------
        float
            Total one-way cost for both legs in dollars.
        """
        return self.one_way(notional_a) + self.one_way(abs(notional_b))

    def pair_round_trip(
        self,
        notional_a: float,
        notional_b: float,
    ) -> float:
        """Round-trip cost for both legs of a pairs trade.

        Parameters
        ----------
        notional_a : float
            Dollar notional on the A leg.
        notional_b : float
            Dollar notional on the B leg (absolute value).

        Returns
        -------
        float
            Total round-trip cost for both legs in dollars.
        """
        return 2.0 * self.pair_one_way(notional_a, abs(notional_b))

    def spread_fill_cost(
        self,
        price_a: float,
        price_b: float,
        hedge_ratio: float,
        notional: float,
    ) -> float:
        """One-way cost for entering/exiting a spread position.

        The A leg trades *notional* dollars.  The B leg trades
        ``notional * hedge_ratio * price_b / price_a`` dollars, scaled
        so the hedge is dollar-neutral.

        Parameters
        ----------
        price_a, price_b : float
            Current prices of asset A and B.
        hedge_ratio : float
            OLS beta (shares of B per share of A).
        notional : float
            Dollar notional allocated to the A leg.

        Returns
        -------
        float
            One-way transaction cost in dollars.
        """
        notional_b = abs(hedge_ratio) * notional * price_b / max(price_a, 1e-9)
        return self.pair_one_way(notional, notional_b)

    def breakeven_half_life(self, notional: float, daily_sigma_dollars: float) -> float:
        """Minimum half-life (days) at which the strategy breaks even after costs.

        A mean-reversion trade over one half-life expects to capture
        roughly ``daily_sigma_dollars * sqrt(half_life)`` in gross P&L.
        This returns the smallest half-life where expected gross P&L
        equals the round-trip cost for both legs.

        Parameters
        ----------
        notional : float
            Dollar notional per leg.
        daily_sigma_dollars : float
            Daily P&L standard deviation in dollars (proxy for edge per period).

        Returns
        -------
        float
            Breakeven half-life in days.  Returns inf if daily_sigma_dollars <= 0.
        """
        if daily_sigma_dollars <= 0:
            return float("inf")
        rt = self.round_trip(notional) * 2  # both legs
        return (rt / daily_sigma_dollars) ** 2
