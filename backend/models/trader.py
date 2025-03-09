import logging
from typing import Dict, List, Any, Optional
import uuid
from datetime import datetime
import random

from ..config import STARTING_CAPITAL, TARGET_CAPITAL, TRADING_FEE

logger = logging.getLogger(__name__)


class Trader:
    """
    Base class for AI trader agents.
    """
    
    def __init__(self, name: str, personality: str, trading_style: str, backstory: str):
        """
        Initialize a trader agent.
        
        Args:
            name: The trader's name
            personality: The trader's personality traits
            trading_style: The trader's trading style
            backstory: The trader's backstory
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.personality = personality
        self.trading_style = trading_style
        self.backstory = backstory
        
        # Portfolio
        self.wallet = {
            'USDT': STARTING_CAPITAL,  # Starting with USDT
        }
        
        # Trading history
        self.trades = []
        
        # Communication history
        self.messages = []
        
        # Status
        self.active = True
        self.goal_reached = False
        self.start_time = datetime.now()
        self.goal_reached_time = None
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate the total portfolio value in USDT.
        
        Args:
            current_prices: Dictionary mapping symbols to their current prices
            
        Returns:
            The total portfolio value in USDT
        """
        total_value = self.wallet.get('USDT', 0)
        
        for symbol, amount in self.wallet.items():
            if symbol != 'USDT':
                # For crypto assets, convert to USDT
                price_key = f"{symbol}USDT"
                if price_key in current_prices:
                    total_value += amount * current_prices[price_key]
        
        return total_value
    
    def check_goal_reached(self, current_prices: Dict[str, float]) -> bool:
        """
        Check if the trader has reached the target capital.
        
        Args:
            current_prices: Dictionary mapping symbols to their current prices
            
        Returns:
            True if the goal has been reached, False otherwise
        """
        if self.goal_reached:
            return True
        
        portfolio_value = self.get_portfolio_value(current_prices)
        
        if portfolio_value >= TARGET_CAPITAL:
            self.goal_reached = True
            self.goal_reached_time = datetime.now()
            logger.info(f"Trader {self.name} reached the goal of ${TARGET_CAPITAL}!")
            return True
        
        return False
    
    def buy(self, symbol: str, amount_usdt: float, price: float) -> bool:
        """
        Buy a cryptocurrency.
        
        Args:
            symbol: The cryptocurrency symbol (e.g., 'BTC')
            amount_usdt: The amount of USDT to spend
            price: The current price of the cryptocurrency
            
        Returns:
            True if the trade was successful, False otherwise
        """
        # Check if the trader has enough USDT
        if self.wallet.get('USDT', 0) < amount_usdt:
            logger.warning(f"Trader {self.name} doesn't have enough USDT to buy {symbol}")
            return False
        
        # Calculate the amount of cryptocurrency to buy
        fee = amount_usdt * TRADING_FEE
        amount_crypto = (amount_usdt - fee) / price
        
        # Update the wallet
        self.wallet['USDT'] = self.wallet.get('USDT', 0) - amount_usdt
        self.wallet[symbol] = self.wallet.get(symbol, 0) + amount_crypto
        
        # Record the trade
        trade = {
            'id': str(uuid.uuid4()),
            'type': 'buy',
            'symbol': symbol,
            'amount_usdt': amount_usdt,
            'amount_crypto': amount_crypto,
            'price': price,
            'fee': fee,
            'timestamp': datetime.now().isoformat()
        }
        self.trades.append(trade)
        
        logger.info(f"Trader {self.name} bought {amount_crypto} {symbol} for {amount_usdt} USDT")
        return True
    
    def sell(self, symbol: str, amount_crypto: float, price: float) -> bool:
        """
        Sell a cryptocurrency.
        
        Args:
            symbol: The cryptocurrency symbol (e.g., 'BTC')
            amount_crypto: The amount of cryptocurrency to sell
            price: The current price of the cryptocurrency
            
        Returns:
            True if the trade was successful, False otherwise
        """
        # Check if the trader has enough cryptocurrency
        if self.wallet.get(symbol, 0) < amount_crypto:
            logger.warning(f"Trader {self.name} doesn't have enough {symbol} to sell")
            return False
        
        # Calculate the amount of USDT to receive
        amount_usdt = amount_crypto * price
        fee = amount_usdt * TRADING_FEE
        amount_usdt_after_fee = amount_usdt - fee
        
        # Update the wallet
        self.wallet[symbol] = self.wallet.get(symbol, 0) - amount_crypto
        self.wallet['USDT'] = self.wallet.get('USDT', 0) + amount_usdt_after_fee
        
        # Record the trade
        trade = {
            'id': str(uuid.uuid4()),
            'type': 'sell',
            'symbol': symbol,
            'amount_crypto': amount_crypto,
            'amount_usdt': amount_usdt_after_fee,
            'price': price,
            'fee': fee,
            'timestamp': datetime.now().isoformat()
        }
        self.trades.append(trade)
        
        logger.info(f"Trader {self.name} sold {amount_crypto} {symbol} for {amount_usdt_after_fee} USDT")
        return True
    
    def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the market data and make trading decisions.
        This method should be overridden by subclasses.
        
        Args:
            market_data: The current market data
            
        Returns:
            A dictionary containing the analysis results
        """
        raise NotImplementedError("Subclasses must implement analyze_market()")
    
    def generate_message(self, market_data: Dict[str, Any], other_traders: List['Trader']) -> Optional[str]:
        """
        Generate a message to share with other traders.
        This method should be overridden by subclasses.
        
        Args:
            market_data: The current market data
            other_traders: List of other traders
            
        Returns:
            A message string, or None if no message is generated
        """
        raise NotImplementedError("Subclasses must implement generate_message()")
    
    def respond_to_message(self, message: Dict[str, Any], market_data: Dict[str, Any]) -> Optional[str]:
        """
        Respond to a message from another trader.
        This method should be overridden by subclasses.
        
        Args:
            message: The message to respond to
            market_data: The current market data
            
        Returns:
            A response message string, or None if no response is generated
        """
        raise NotImplementedError("Subclasses must implement respond_to_message()")
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the trader.
        
        Returns:
            A dictionary containing information about the trader
        """
        return {
            'id': self.id,
            'name': self.name,
            'personality': self.personality,
            'trading_style': self.trading_style,
            'backstory': self.backstory,
            'wallet': self.wallet,
            'trades': self.trades[-10:] if len(self.trades) > 10 else self.trades,  # Last 10 trades
            'active': self.active,
            'goal_reached': self.goal_reached,
            'start_time': self.start_time.isoformat(),
            'goal_reached_time': self.goal_reached_time.isoformat() if self.goal_reached_time else None,
            'messages': self.messages[-10:] if len(self.messages) > 10 else self.messages  # Last 10 messages
        } 