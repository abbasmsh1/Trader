import asyncio
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import logging
from datetime import datetime
import os
import sys
from typing import Dict, Any
from pydantic import BaseModel

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.api.binance_api import BinanceAPI
from backend.models.simulation import Simulation
from backend.models.market import Market
from backend.utils.websocket_manager import WebSocketManager
from backend.config import WEBSOCKET_UPDATE_INTERVAL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="AI Crypto Trader Simulation")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize WebSocket manager
websocket_manager = WebSocketManager()

# Initialize Binance API client
binance_api = BinanceAPI()

# Initialize market and simulation
market = Market(binance_api)
simulation = Simulation(market)

class SellRequest(BaseModel):
    symbol: str
    amount: float

@app.on_event("startup")
async def startup_event():
    """Start the simulation when the application starts."""
    logger.info("Starting AI Crypto Trader Simulation")
    asyncio.create_task(simulation.run())
    asyncio.create_task(broadcast_updates())


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when the application shuts down."""
    logger.info("Shutting down AI Crypto Trader Simulation")
    await simulation.stop()


@app.get("/")
async def root():
    """Root endpoint that returns basic information about the API."""
    return {
        "name": "AI Crypto Trader Simulation API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/traders")
async def get_traders():
    """Get information about all traders."""
    return {"traders": simulation.get_traders_info()}


@app.get("/traders/{trader_id}")
async def get_trader(trader_id: str):
    """Get detailed information about a specific trader."""
    trader_info = simulation.get_trader_info(trader_id)
    if trader_info:
        return trader_info
    raise HTTPException(status_code=404, detail=f"Trader with ID {trader_id} not found")


@app.get("/market")
async def get_market_data():
    """Get current market data for all trading pairs."""
    return {"market_data": market.get_current_data()}


@app.get("/market/{trading_pair}")
async def get_pair_data(trading_pair: str):
    """Get current market data for a specific trading pair."""
    data = market.get_pair_data(trading_pair)
    if data:
        return {"market_data": data}
    raise HTTPException(status_code=404, detail=f"Trading pair {trading_pair} not found")


@app.get("/communications")
async def get_communications():
    """Get recent communications between traders."""
    return {"communications": simulation.get_recent_communications()}


@app.post("/traders/{trader_id}/sell")
async def sell_asset(trader_id: str, request: SellRequest):
    """Sell an asset for a specific trader."""
    trader = None
    for t in simulation.traders:
        if t.id == trader_id:
            trader = t
            break
    
    if not trader:
        raise HTTPException(status_code=404, detail=f"Trader with ID {trader_id} not found")
    
    # Get current market data
    market_data = market.get_current_data()
    symbol_pair = f"{request.symbol}USDT"
    
    if symbol_pair not in market_data:
        raise HTTPException(status_code=400, detail=f"Trading pair {symbol_pair} not found")
    
    current_price = market_data[symbol_pair]['price']
    
    # Execute the sell
    success = trader.sell(request.symbol, request.amount, current_price)
    
    if success:
        return {"success": True, "message": f"Successfully sold {request.amount} {request.symbol}"}
    else:
        raise HTTPException(status_code=400, detail="Failed to sell asset")


@app.get("/analytics")
async def get_all_analytics():
    """Get analytics for all traders."""
    analytics = {}
    for trader in simulation.traders:
        trader_analytics = trader.trade_store.get_analytics(trader.id)
        if trader_analytics:
            analytics[trader.id] = {
                'trader_name': trader.name,
                'analytics': trader_analytics
            }
    return {"analytics": analytics}


@app.get("/analytics/{trader_id}")
async def get_trader_analytics(trader_id: str):
    """Get detailed analytics for a specific trader."""
    trader = None
    for t in simulation.traders:
        if t.id == trader_id:
            trader = t
            break
    
    if not trader:
        raise HTTPException(status_code=404, detail=f"Trader with ID {trader_id} not found")
    
    analytics = trader.trade_store.get_analytics(trader.id)
    if not analytics:
        raise HTTPException(status_code=404, detail=f"No analytics found for trader {trader_id}")
    
    return {
        "trader_name": trader.name,
        "analytics": analytics
    }


@app.get("/simulation/status")
async def get_simulation_status():
    """Get detailed simulation status including performance metrics."""
    active_traders = [t for t in simulation.traders if t.active]
    
    # Get current market data for portfolio calculations
    market_data = market.get_current_data()
    
    # Calculate simulation metrics
    metrics = {
        'start_time': simulation.start_time.isoformat() if simulation.start_time else None,
        'duration': str(datetime.now() - simulation.start_time) if simulation.start_time else None,
        'active_traders': len(active_traders),
        'total_traders': len(simulation.traders),
        'winner': {
            'name': simulation.winner.name,
            'id': simulation.winner.id,
            'time_to_win': str(simulation.winner.goal_reached_time - simulation.start_time)
        } if simulation.winner else None,
        'traders': []
    }
    
    # Add detailed trader information
    for trader in simulation.traders:
        portfolio_value = trader.get_portfolio_value(market_data)
        analytics = trader.trade_store.get_analytics(trader.id)
        
        trader_info = {
            'id': trader.id,
            'name': trader.name,
            'active': trader.active,
            'portfolio_value': portfolio_value,
            'goal_reached': trader.goal_reached,
            'total_trades': analytics.get('total_trades', 0),
            'win_rate': analytics.get('win_rate', 0),
            'profit_loss': portfolio_value - STARTING_CAPITAL
        }
        metrics['traders'].append(trader_info)
    
    return metrics


@app.post("/simulation/backup")
async def create_backup():
    """Manually trigger a backup of all simulation data."""
    try:
        for trader in simulation.traders:
            # Force immediate backup
            trader.trade_store._check_and_create_backup()
        return {"message": "Backup created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/news")
async def get_news():
    """Get current news data and sentiment analysis."""
    if not simulation:
        raise HTTPException(status_code=400, detail="Simulation not running")
    
    return simulation.get_news_data()


@app.get("/api/traders/{trader_id}/news-impact")
async def get_trader_news_impact(trader_id: str):
    """Get news impact analysis for a specific trader."""
    if not simulation:
        raise HTTPException(status_code=400, detail="Simulation not running")
    
    news_impact = simulation.get_trader_news_impact(trader_id)
    if not news_impact:
        raise HTTPException(status_code=404, detail="Trader not found")
    
    return news_impact


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket_manager.connect(websocket)
    try:
        while True:
            # Get current data
            market_data = market.get_current_data()
            traders_info = simulation.get_traders_info()
            communications = simulation.get_recent_communications()
            
            # Send update
            await websocket_manager.broadcast({
                "market": market_data,
                "traders": traders_info,
                "communications": communications
            })
            
            # Wait before next update
            await asyncio.sleep(WEBSOCKET_UPDATE_INTERVAL)
    
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


async def broadcast_updates():
    """Broadcast updates to all connected WebSocket clients."""
    while True:
        # Get current state
        data = {
            "timestamp": datetime.now().isoformat(),
            "traders": simulation.get_traders_info(),
            "market": market.get_current_data(),
            "communications": simulation.get_recent_communications()
        }
        
        # Broadcast to all connected clients
        await websocket_manager.broadcast(json.dumps(data))
        
        # Wait before sending the next update
        await asyncio.sleep(1)


@app.post("/reset")
async def reset_simulation():
    """Reset all trader accounts and restart the simulation."""
    global simulation, market
    
    # Stop the current simulation
    await simulation.stop()
    
    # Reinitialize market and simulation
    market = Market(binance_api)
    simulation = Simulation(market)
    
    # Start the new simulation
    asyncio.create_task(simulation.run())
    
    return {"success": True, "message": "Simulation reset successfully"}


if __name__ == "__main__":
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True) 