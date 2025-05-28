import pandas as pd
import logging

logger = logging.getLogger(__name__)

class BaseStrategy:
    """
    Base Strategy class defining the interface that all strategies must implement.
    This ensures that the backtest engine can work with any strategy that follows this interface.
    """
    
    def __init__(self, params=None):
        """
        Initialize the strategy with optional parameters.
        
        Args:
            params (dict, optional): Strategy-specific parameters.
        """
        self.params = params or {}
        self.indicators = {}
    
    def compute_indicators(self, data_df):
        """
        Compute technical indicators needed for the strategy.
        This method must be implemented by subclasses.
        
        Args:
            data_df (pd.DataFrame): DataFrame with price data (must contain OHLCV columns).
            
        Returns:
            pd.DataFrame: DataFrame with added indicator columns.
        """
        raise NotImplementedError("Subclasses must implement compute_indicators()")
    
    def generate_signals(self, data_df):
        """
        Generate trading signals based on indicators.
        This method must be implemented by subclasses.
        
        Args:
            data_df (pd.DataFrame): DataFrame with price data and indicators.
            
        Returns:
            tuple: (signal_actions, signal_sizes) - Series of actions and sizes.
        """
        raise NotImplementedError("Subclasses must implement generate_signals()")
    
    def prepare_for_backtest(self, ohlcv_df):
        """
        Prepare data for backtest by computing indicators and generating signals.
        
        Args:
            ohlcv_df (pd.DataFrame): DataFrame with OHLCV data.
            
        Returns:
            tuple: (df_with_indicators, signal_actions, signal_sizes)
        """
        logger.info(f"Preparing strategy: Computing indicators and generating signals...")
        
        # Standardize column names if needed
        df = ohlcv_df.copy()
        if 'Open' in df.columns and 'open' not in df.columns:
            df.rename(columns={
                'Open': 'open', 
                'High': 'high', 
                'Low': 'low', 
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)
        
        # Compute indicators
        df_with_indicators = self.compute_indicators(df)
        
        if df_with_indicators is None:
            logger.error("Indicator computation returned None. Strategy will not generate trade signals.")
            signal_actions = pd.Series("HOLD", index=df.index)
            signal_sizes = pd.Series(0.0, index=df.index)
        else:
            # Generate signals
            signal_actions, signal_sizes = self.generate_signals(df_with_indicators)
            
            # Ensure indices match
            if not signal_actions.index.equals(df.index):
                logger.warning("Signal actions index does not match data index. Reindexing to align.")
                signal_actions = signal_actions.reindex(df.index, fill_value="HOLD")
            if not signal_sizes.index.equals(df.index):
                logger.warning("Signal sizes index does not match data index. Reindexing to align.")
                signal_sizes = signal_sizes.reindex(df.index, fill_value=0.0)
        
        logger.info(
            f"Strategy prepared. Total data points: {len(df)}. "
            f"Buy signals found: {(signal_actions == 'BUY').sum()}, "
            f"Sell signals found: {(signal_actions == 'SELL').sum()}"
        )
        
        return df_with_indicators, signal_actions, signal_sizes
