import asyncio
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import logging
from datetime import datetime

from api.binance_api import BinanceAPI
from models.simulation import Simulation
from models.market import Market
from utils.websocket_manager import WebSocketManager

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
    return {"error": f"Trader with ID {trader_id} not found"}


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
    return {"error": f"Trading pair {trading_pair} not found"}


@app.get("/communications")
async def get_communications():
    """Get recent communications between traders."""
    return {"communications": simulation.get_recent_communications()}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket_manager.connect(websocket)
    try:
        while True:
            # Keep the connection alive
            await websocket.receive_text()
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


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 