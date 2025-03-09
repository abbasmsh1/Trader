import json
import os
import gzip
import shutil
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .config.persistence import (
    TRADE_DATA_DIR, STATE_DATA_DIR, BACKUP_DIR, ANALYTICS_DIR,
    MAX_BACKUPS, BACKUP_INTERVAL_HOURS, MAX_TRADE_HISTORY_DAYS,
    COMPRESS_OLD_DATA, COMPRESSION_THRESHOLD_DAYS
)

logger = logging.getLogger(__name__)

class TradeStore:
    """Handles persistence of trade data and trader states with advanced features."""
    
    def __init__(self):
        """Initialize the trade store with advanced persistence features."""
        self._ensure_directories()
        self._last_backup_time = self._load_last_backup_time()
        self._executor = ThreadPoolExecutor(max_workers=2)
    
    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        for directory in [TRADE_DATA_DIR, STATE_DATA_DIR, BACKUP_DIR, ANALYTICS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_last_backup_time(self) -> datetime:
        """Load the timestamp of the last backup."""
        backup_time_file = BACKUP_DIR / "last_backup.txt"
        if backup_time_file.exists():
            try:
                with open(backup_time_file, 'r') as f:
                    return datetime.fromisoformat(f.read().strip())
            except Exception as e:
                logger.error(f"Error loading last backup time: {e}")
        return datetime.min
    
    def _save_last_backup_time(self, timestamp: datetime) -> None:
        """Save the timestamp of the last backup."""
        backup_time_file = BACKUP_DIR / "last_backup.txt"
        try:
            with open(backup_time_file, 'w') as f:
                f.write(timestamp.isoformat())
        except Exception as e:
            logger.error(f"Error saving last backup time: {e}")
    
    def save_trades(self, trader_id: str, trades: List[Dict[str, Any]]) -> None:
        """Save trades for a specific trader with automatic backup and compression."""
        try:
            # Save trades
            trade_file = TRADE_DATA_DIR / f"{trader_id}_trades.json"
            with open(trade_file, 'w') as f:
                json.dump(trades, f, indent=2)
            
            # Trigger async operations
            self._executor.submit(self._handle_async_operations, trader_id, trades)
            
            logger.info(f"Saved {len(trades)} trades for trader {trader_id}")
        except Exception as e:
            logger.error(f"Error saving trades: {e}")
    
    def _handle_async_operations(self, trader_id: str, trades: List[Dict[str, Any]]) -> None:
        """Handle asynchronous operations like backup, compression, and analytics."""
        try:
            # Check if backup is needed
            self._check_and_create_backup()
            
            # Compress old data
            if COMPRESS_OLD_DATA:
                self._compress_old_data(trader_id)
            
            # Update analytics
            self._update_analytics(trader_id, trades)
            
            # Clean up old data
            self._cleanup_old_data(trader_id)
        except Exception as e:
            logger.error(f"Error in async operations: {e}")
    
    def _check_and_create_backup(self) -> None:
        """Create a backup if enough time has passed since the last backup."""
        now = datetime.now()
        if (now - self._last_backup_time).total_seconds() > BACKUP_INTERVAL_HOURS * 3600:
            try:
                # Create backup directory with timestamp
                backup_dir = BACKUP_DIR / now.strftime("%Y%m%d_%H%M%S")
                backup_dir.mkdir(exist_ok=True)
                
                # Backup trade data
                shutil.copytree(TRADE_DATA_DIR, backup_dir / "trades", dirs_exist_ok=True)
                shutil.copytree(STATE_DATA_DIR, backup_dir / "states", dirs_exist_ok=True)
                
                # Remove old backups if exceeding MAX_BACKUPS
                self._cleanup_old_backups()
                
                # Update last backup time
                self._save_last_backup_time(now)
                self._last_backup_time = now
                
                logger.info(f"Created backup at {backup_dir}")
            except Exception as e:
                logger.error(f"Error creating backup: {e}")
    
    def _cleanup_old_backups(self) -> None:
        """Remove old backups exceeding MAX_BACKUPS."""
        try:
            backups = sorted(BACKUP_DIR.glob("*"), key=os.path.getctime, reverse=True)
            for backup in backups[MAX_BACKUPS:]:
                if backup.is_dir() and backup.name != "last_backup.txt":
                    shutil.rmtree(backup)
                    logger.info(f"Removed old backup: {backup}")
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")
    
    def _compress_old_data(self, trader_id: str) -> None:
        """Compress old trade data to save space."""
        try:
            trade_file = TRADE_DATA_DIR / f"{trader_id}_trades.json"
            if not trade_file.exists():
                return
                
            # Check if file is older than threshold
            if (datetime.now() - datetime.fromtimestamp(trade_file.stat().st_mtime)).days > COMPRESSION_THRESHOLD_DAYS:
                # Compress file
                with open(trade_file, 'rb') as f_in:
                    with gzip.open(f"{trade_file}.gz", 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # Remove original file
                trade_file.unlink()
                logger.info(f"Compressed old trade data for trader {trader_id}")
        except Exception as e:
            logger.error(f"Error compressing old data: {e}")
    
    def _update_analytics(self, trader_id: str, trades: List[Dict[str, Any]]) -> None:
        """Update trading analytics."""
        try:
            # Convert trades to DataFrame for analysis
            df = pd.DataFrame(trades)
            if df.empty:
                return
                
            # Calculate analytics
            analytics = {
                'total_trades': len(trades),
                'buy_trades': len(df[df['type'] == 'buy']),
                'sell_trades': len(df[df['type'] == 'sell']),
                'total_volume_usdt': df['amount_usdt'].sum(),
                'total_fees': df['fee'].sum(),
                'avg_trade_size': df['amount_usdt'].mean(),
                'most_traded_symbol': df['symbol'].mode().iloc[0] if not df['symbol'].empty else None,
                'last_updated': datetime.now().isoformat()
            }
            
            # Save analytics
            analytics_file = ANALYTICS_DIR / f"{trader_id}_analytics.json"
            with open(analytics_file, 'w') as f:
                json.dump(analytics, f, indent=2)
            
            logger.info(f"Updated analytics for trader {trader_id}")
        except Exception as e:
            logger.error(f"Error updating analytics: {e}")
    
    def _cleanup_old_data(self, trader_id: str) -> None:
        """Clean up old trade data exceeding retention period."""
        try:
            trade_file = TRADE_DATA_DIR / f"{trader_id}_trades.json"
            if not trade_file.exists():
                return
                
            with open(trade_file, 'r') as f:
                trades = json.load(f)
            
            # Filter out old trades
            cutoff_date = datetime.now() - timedelta(days=MAX_TRADE_HISTORY_DAYS)
            current_trades = [
                trade for trade in trades 
                if datetime.fromisoformat(trade['timestamp']) > cutoff_date
            ]
            
            # Save filtered trades
            if len(current_trades) < len(trades):
                with open(trade_file, 'w') as f:
                    json.dump(current_trades, f, indent=2)
                logger.info(f"Cleaned up {len(trades) - len(current_trades)} old trades for trader {trader_id}")
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
    
    def load_trades(self, trader_id: str) -> List[Dict[str, Any]]:
        """Load trades for a specific trader, handling both compressed and uncompressed data."""
        try:
            # Check for uncompressed file
            trade_file = TRADE_DATA_DIR / f"{trader_id}_trades.json"
            if trade_file.exists():
                with open(trade_file, 'r') as f:
                    return json.load(f)
            
            # Check for compressed file
            compressed_file = TRADE_DATA_DIR / f"{trader_id}_trades.json.gz"
            if compressed_file.exists():
                with gzip.open(compressed_file, 'rt') as f:
                    return json.load(f)
            
            return []
        except Exception as e:
            logger.error(f"Error loading trades: {e}")
            return []
    
    def get_analytics(self, trader_id: str) -> Dict[str, Any]:
        """Get analytics for a specific trader."""
        try:
            analytics_file = ANALYTICS_DIR / f"{trader_id}_analytics.json"
            if analytics_file.exists():
                with open(analytics_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading analytics: {e}")
            return {}
    
    def save_trader_state(self, trader_id: str, state: Dict[str, Any]) -> None:
        """Save the current state of a trader with versioning."""
        try:
            # Add version information
            state['version'] = {
                'timestamp': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            state_file = STATE_DATA_DIR / f"{trader_id}_state.json"
            
            # Create backup of current state if it exists
            if state_file.exists():
                backup_file = STATE_DATA_DIR / f"{trader_id}_state.backup.json"
                shutil.copy2(state_file, backup_file)
            
            # Save new state
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Saved state for trader {trader_id}")
        except Exception as e:
            logger.error(f"Error saving trader state: {e}")
    
    def load_trader_state(self, trader_id: str) -> Dict[str, Any]:
        """Load the state for a specific trader with fallback to backup."""
        try:
            state_file = STATE_DATA_DIR / f"{trader_id}_state.json"
            if state_file.exists():
                try:
                    with open(state_file, 'r') as f:
                        return json.load(f)
                except json.JSONDecodeError:
                    # Try loading from backup
                    backup_file = STATE_DATA_DIR / f"{trader_id}_state.backup.json"
                    if backup_file.exists():
                        with open(backup_file, 'r') as f:
                            return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading trader state: {e}")
            return {}
