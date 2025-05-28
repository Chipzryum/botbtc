import pandas as pd
import logging

logger = logging.getLogger(__name__)

class StrategyWrapper:
    """
    Wrapper class that adapts our strategy interface to the backtesting.py library requirements.
    This allows us to use our strategies with the backtesting.py library.
    """

    def __init__(self, strategy_class, strategy_params=None):
        """
        Initialize the strategy wrapper.

        Args:
            strategy_class: The strategy class to use (must implement BaseStrategy interface)
            strategy_params (dict, optional): Strategy-specific parameters
        """
        self.strategy_class = strategy_class
        self.strategy_params = strategy_params or {}

    def create_backtesting_strategy(self, debug_logging=False):
        """
        Create a strategy class compatible with the backtesting.py library.

        Args:
            debug_logging (bool): Whether to enable debug logging

        Returns:
            type: A Strategy subclass for the backtesting.py library
        """
        strategy_class = self.strategy_class
        strategy_params = self.strategy_params

        # Import here to avoid circular imports
        from backtesting import Strategy

        class BacktestingStrategy(Strategy):
            """
            Strategy adapter for the backtesting.py library.
            This dynamically created class wraps our strategy implementation.
            """

            def init(self):
                """Initialize the strategy with indicators and signals."""
                logger.info(f"Initializing {strategy_class.__name__} for backtesting")

                # Create strategy instance with parameters
                self.strategy = strategy_class(strategy_params)

                # Create DataFrame from backtesting.py's data
                current_data_df = pd.DataFrame({
                    'open': self.data.Open,
                    'high': self.data.High,
                    'low': self.data.Low,
                    'close': self.data.Close,
                    'volume': self.data.Volume if hasattr(self.data, 'Volume') else None
                }, index=self.data.index)

                # Drop volume column if it's None
                if current_data_df['volume'] is None:
                    current_data_df.drop('volume', axis=1, inplace=True)

                # Process data with the strategy
                _, self.signal_actions, self.signal_sizes = self.strategy.prepare_for_backtest(current_data_df)

                # Log signal summary
                logger.info(f"Strategy initialized. Buy signals: {(self.signal_actions == 'BUY').sum()}, "
                           f"Sell signals: {(self.signal_actions == 'SELL').sum()}")

            def next(self):
                """Execute trading logic for the current bar."""
                current_bar_iloc = len(self.data.Close) - 1

                if current_bar_iloc < 0 or current_bar_iloc >= len(self.signal_actions):
                    logger.warning(
                        f"current_bar_iloc {current_bar_iloc} is out of bounds for signal series "
                        f"(len {len(self.signal_actions)}). Skipping trade logic for this bar."
                    )
                    return

                current_action = self.signal_actions.iloc[current_bar_iloc]
                current_size_percentage = self.signal_sizes.iloc[current_bar_iloc]
                current_price = self.data.Close[-1]

                if debug_logging and current_bar_iloc > 0 and current_bar_iloc % 500 == 0:
                    position_status = 'Flat'
                    if self.position.is_long:
                        position_status = 'Long'
                    elif self.position.is_short:
                        position_status = 'Short'
                    logger.debug(
                        f"Step: {current_bar_iloc}, Price: {current_price:.2f}, "
                        f"Action: {current_action}, Size%: {current_size_percentage:.4f}, "
                        f"Position: {position_status}, Equity: {self.equity:.2f}"
                    )

                if current_action == "BUY" and current_size_percentage > 0:
                    if not self.position.is_long:
                        self.buy(size=current_size_percentage)
                        if debug_logging:
                            logger.debug(
                                f"Executed BUY: Price={current_price:.2f}, "
                                f"Size%={current_size_percentage * 100:.2f}, Equity={self.equity:.2f}"
                            )

                elif current_action == "SELL" and current_size_percentage > 0:
                    if not self.position.is_short:
                        self.sell(size=current_size_percentage)
                        if debug_logging:
                            logger.debug(
                                f"Executed SELL: Price={current_price:.2f}, "
                                f"Size%={current_size_percentage * 100:.2f}, Equity={self.equity:.2f}"
                            )

        return BacktestingStrategy
