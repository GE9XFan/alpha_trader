"""
Discord Bot - Implementation Plan Week 7-8
Publishes paper trades with Alpha Vantage data
"""
import discord
from discord.ext import commands
import asyncio
from datetime import datetime

from src.core.config import config
from src.core.logger import get_logger
from src.trading.paper_trader import paper_trader


logger = get_logger(__name__)


class TradingBot(commands.Bot):
    """
    Discord bot - PUBLISHES PAPER TRADES WITH ALPHA VANTAGE DATA
    Implementation Plan Week 7-8
    """
    
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='!', intents=intents)
        
        self.paper = paper_trader  # Reuse paper trader!
        self.config = config.community['discord']
        self.channels = {}
    
    async def on_ready(self):
        """Bot ready event"""
        logger.info(f'Bot connected as {self.user}')
        
        # Get channels
        if self.config['channels']:
            self.channels['signals'] = self.get_channel(self.config['channels'].get('signals'))
            self.channels['performance'] = self.get_channel(self.config['channels'].get('performance'))
            self.channels['analytics'] = self.get_channel(self.config['channels'].get('analytics'))
        
        # Start publishing loops
        if self.channels.get('signals'):
            self.loop.create_task(self.publish_trades())
        if self.channels.get('analytics'):
            self.loop.create_task(self.publish_av_analytics())
    
    async def publish_trades(self):
        """Publish paper trades to Discord with Alpha Vantage Greeks"""
        last_trade_id = 0
        
        while True:
            try:
                # Get new trades from database
                with self.paper.db.get_db() as conn:
                    cur = conn.cursor()
                    cur.execute("""
                        SELECT * FROM trades 
                        WHERE id > %s AND mode = 'paper'
                        ORDER BY id
                        LIMIT 10
                    """, (last_trade_id,))
                    
                    columns = [desc[0] for desc in cur.description]
                    new_trades = [dict(zip(columns, row)) for row in cur.fetchall()]
                
                for trade in new_trades:
                    # Format trade as embed with AV data
                    embed = discord.Embed(
                        title=f"📈 Paper Trade: {trade['symbol']}",
                        color=discord.Color.green() if trade['action'] == 'BUY' else discord.Color.red(),
                        timestamp=trade['timestamp']
                    )
                    
                    embed.add_field(name="Action", value=trade['action'], inline=True)
                    embed.add_field(name="Option", value=f"{trade['option_type']} ${trade['strike']}", inline=True)
                    embed.add_field(name="Quantity", value=f"{trade['quantity']} contracts", inline=True)
                    embed.add_field(name="Price", value=f"${trade.get('fill_price', 0):.2f}", inline=True)
                    
                    # Add Greeks from Alpha Vantage
                    if trade.get('entry_delta') is not None:
                        embed.add_field(
                            name="Greeks (AV)", 
                            value=f"Δ={trade['entry_delta']:.3f} Γ={trade.get('entry_gamma', 0):.3f} "
                                  f"Θ={trade.get('entry_theta', 0):.3f} IV={trade.get('entry_iv', 0):.1%}",
                            inline=False
                        )
                    
                    await self.channels['signals'].send(embed=embed)
                    
                    last_trade_id = trade['id']
                
            except Exception as e:
                logger.error(f"Error publishing trades: {e}")
            
            await asyncio.sleep(5)  # Check every 5 seconds
    
    async def publish_av_analytics(self):
        """Publish Alpha Vantage analytics periodically"""
        while True:
            try:
                # Get sentiment from Alpha Vantage
                sentiment = await self.paper.av.get_news_sentiment(config.trading.symbols)
                
                if sentiment and 'feed' in sentiment:
                    embed = discord.Embed(
                        title="📰 Market Sentiment (Alpha Vantage)",
                        color=discord.Color.blue(),
                        timestamp=datetime.now()
                    )
                    
                    for article in sentiment['feed'][:3]:
                        embed.add_field(
                            name=article.get('title', 'News')[:100],
                            value=f"Sentiment: {article.get('overall_sentiment_score', 'N/A')}",
                            inline=False
                        )
                    
                    await self.channels['analytics'].send(embed=embed)
                
            except Exception as e:
                logger.error(f"Error publishing analytics: {e}")
            
            await asyncio.sleep(900)  # Every 15 minutes


# BOT REUSES PAPER TRADER WITH ALPHA VANTAGE DATA
bot = TradingBot()
