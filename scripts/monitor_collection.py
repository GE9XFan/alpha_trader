#!/usr/bin/env python3
"""
Enhanced Real-time monitoring dashboard for data collection
Shows actual collection activity, frequencies, and statistics
"""
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.logger import get_logger

logger = get_logger(__name__)


class CollectionMonitor:
    """Enhanced monitor for data collection with real-time metrics"""
    
    def __init__(self):
        self.connection_params = {
            'host': 'localhost',
            'port': 5432,
            'database': 'alphatrader',
            'user': 'michaelmerrick'
        }
        
        # Track previous counts for growth calculation
        self.previous_counts = {}
        self.start_counts = {}
        
        # Define collection schedules (in seconds)
        self.schedules = {
            'options_data': {'realtime': 30, 'historical': 86400},  # 30 sec / daily
            'technical_indicators': 300,  # 5 minutes
            'market_data': 10,  # Streaming (continuous)
            'news_sentiment': 1800,  # 30 minutes
            'analytics': 900,  # 15 minutes
            'fundamentals': 86400,  # Daily
            'economic_indicators': 86400  # Daily
        }
    
    def get_table_stats(self):
        """Get enhanced statistics for all tables"""
        stats = {}
        
        try:
            with psycopg2.connect(**self.connection_params) as conn:
                cur = conn.cursor(cursor_factory=RealDictCursor)
                
                tables = [
                    ('options_data', '📦', 'Options'),
                    ('technical_indicators', '📊', 'Indicators'),
                    ('market_data', '📈', 'Market Bars'),
                    ('news_sentiment', '📰', 'News'),
                    ('analytics', '🔬', 'Analytics'),
                    ('fundamentals', '📋', 'Fundamentals'),
                    ('economic_indicators', '💹', 'Economic')
                ]
                
                for table, emoji, display_name in tables:
                    # Get total count
                    cur.execute(f"SELECT COUNT(*) as count FROM {table}")
                    count = cur.fetchone()['count']
                    
                    # Get latest timestamp
                    cur.execute(f"""
                        SELECT MAX(timestamp) as latest 
                        FROM {table}
                        WHERE timestamp IS NOT NULL
                    """)
                    result = cur.fetchone()
                    latest = result['latest'] if result else None
                    
                    # Get data from last 5 minutes for activity indicator
                    cur.execute(f"""
                        SELECT COUNT(*) as recent
                        FROM {table}
                        WHERE timestamp > NOW() - INTERVAL '5 minutes'
                    """)
                    recent = cur.fetchone()['recent']
                    
                    # Get unique counts for better metrics
                    if table == 'technical_indicators':
                        cur.execute("""
                            SELECT COUNT(DISTINCT indicator) as indicators,
                                   COUNT(DISTINCT symbol) as symbols
                            FROM technical_indicators
                        """)
                        unique = cur.fetchone()
                        extra_info = f"{unique['indicators']} types, {unique['symbols']} symbols"
                    elif table == 'options_data':
                        cur.execute("""
                            SELECT COUNT(DISTINCT data_type) as types
                            FROM options_data
                        """)
                        types = cur.fetchone()['types']
                        extra_info = f"{types} types (RT/Hist)"
                    else:
                        extra_info = None
                    
                    stats[table] = {
                        'count': count,
                        'emoji': emoji,
                        'display_name': display_name,
                        'latest': latest,
                        'recent_5min': recent,
                        'extra_info': extra_info
                    }
                
        except Exception as e:
            logger.error(f"Error getting table stats: {e}")
        
        return stats
    
    def calculate_growth(self, current, previous):
        """Calculate growth between two measurements"""
        if previous == 0:
            return 0
        return current - previous
    
    def format_time_ago(self, timestamp):
        """Format timestamp as time ago with color"""
        if not timestamp:
            return "Never", "⚫"
        
        if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo:
            timestamp = timestamp.replace(tzinfo=None)
        
        delta = datetime.now() - timestamp
        seconds = delta.total_seconds()
        
        # Color based on recency
        if seconds < 60:
            time_str = f"{int(seconds)}s ago"
            color = "🟢"  # Green - very recent
        elif seconds < 300:
            time_str = f"{int(seconds / 60)}m ago"
            color = "🟡"  # Yellow - recent
        elif seconds < 3600:
            time_str = f"{int(seconds / 60)}m ago"
            color = "🟠"  # Orange - getting old
        elif seconds < 86400:
            time_str = f"{int(seconds / 3600)}h ago"
            color = "🔴"  # Red - old
        else:
            time_str = f"{int(delta.days)}d ago"
            color = "⚫"  # Black - very old
        
        return time_str, color
    
    def get_collection_status(self, table_name, last_update):
        """Determine if collection is active based on schedule"""
        if not last_update:
            return "⚫ IDLE"
        
        if hasattr(last_update, 'tzinfo') and last_update.tzinfo:
            last_update = last_update.replace(tzinfo=None)
        
        seconds_since = (datetime.now() - last_update).total_seconds()
        
        # Get expected schedule
        if table_name == 'options_data':
            # During market hours: 30 seconds
            now = datetime.now()
            if now.weekday() < 5 and 9 <= now.hour < 16:
                expected = 30
            else:
                expected = 86400
        elif table_name == 'market_data':
            expected = 10 if self.is_market_hours() else float('inf')
        else:
            expected = self.schedules.get(table_name, 3600)
        
        # Determine status
        if seconds_since < expected * 1.5:
            return "🟢 ACTIVE"
        elif seconds_since < expected * 3:
            return "🟡 DELAYED"
        else:
            return "🔴 STALE"
    
    def is_market_hours(self):
        """Check if market is open"""
        now = datetime.now()
        return now.weekday() < 5 and 9 <= now.hour < 16
    
    def print_dashboard(self, stats):
        """Print enhanced monitoring dashboard"""
        # Clear screen
        print("\033[H\033[J", end="")
        
        print("="*100)
        print(f"📊 ALPHATRADER DATA COLLECTION MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*100)
        
        # Market status
        market_status = "🟢 MARKET OPEN" if self.is_market_hours() else "🔴 MARKET CLOSED"
        
        # Calculate totals
        total_records = sum(s['count'] for s in stats.values())
        total_growth = sum(
            self.calculate_growth(s['count'], self.previous_counts.get(table, s['count']))
            for table, s in stats.items()
        )
        session_growth = sum(
            s['count'] - self.start_counts.get(table, s['count'])
            for table, s in stats.items()
        )
        
        print(f"\n{market_status}  |  📈 TOTAL: {total_records:,} records  |  "
              f"🔄 Last 30s: +{total_growth:,}  |  📊 Session: +{session_growth:,}")
        
        # Collection Schedule
        print("\n" + "─"*100)
        print("📅 COLLECTION SCHEDULE:")
        print("  • Options: 30 sec (market) / Daily (off-hours)  • Indicators: 5 min  • Bars: Streaming")
        print("  • News: 30 min  • Analytics: 15 min  • Fundamentals/Economic: Daily")
        
        print("\n" + "─"*100)
        print("DATA TYPE               RECORDS    STATUS      LAST UPDATE    GROWTH    ACTIVITY")
        print("─"*100)
        
        # Sort tables by recent activity
        sorted_tables = sorted(stats.items(), 
                              key=lambda x: x[1].get('recent_5min', 0), 
                              reverse=True)
        
        for table, data in sorted_tables:
            count = data['count']
            emoji = data['emoji']
            display_name = data.get('display_name', table)
            growth = self.calculate_growth(count, self.previous_counts.get(table, count))
            time_str, time_color = self.format_time_ago(data.get('latest'))
            status = self.get_collection_status(table, data.get('latest'))
            
            # Format display name with extra info
            name_display = f"{emoji} {display_name}"
            if data.get('extra_info'):
                name_display = f"{emoji} {display_name[:10]}({data['extra_info'][:15]})"
            
            # Growth indicator
            if growth > 0:
                growth_str = f"+{growth:,}"
                growth_indicator = "📈"
            elif growth == 0:
                growth_str = "  -"
                growth_indicator = "➖"
            else:
                growth_str = f"{growth:,}"
                growth_indicator = "📉"
            
            # Activity sparkline
            recent = data.get('recent_5min', 0)
            if recent > 50:
                activity = "████"
            elif recent > 20:
                activity = "███░"
            elif recent > 5:
                activity = "██░░"
            elif recent > 0:
                activity = "█░░░"
            else:
                activity = "░░░░"
            
            print(f"{name_display:28} {count:8,}  {status:10}  {time_color} {time_str:10}  "
                  f"{growth_indicator} {growth_str:7}  {activity}")
        
        print("─"*100)
        
        # Real-time activity indicators
        print("\n🔴 REAL-TIME ACTIVITY:")
        
        # Show what's actively collecting
        active_collections = []
        for table, data in stats.items():
            if data.get('recent_5min', 0) > 0:
                active_collections.append(f"{data['emoji']} {data['display_name']}: {data['recent_5min']} updates")
        
        if active_collections:
            print("  Active: " + " | ".join(active_collections[:3]))
        else:
            print("  ⏸️  No active collection detected")
        
        # Collection rates
        print("\n📊 COLLECTION RATES (per 30 seconds):")
        active_rates = []
        for table, data in stats.items():
            rate = self.calculate_growth(data['count'], self.previous_counts.get(table, data['count']))
            if rate > 0:
                active_rates.append((data['emoji'], data['display_name'], rate))
        
        if active_rates:
            for emoji, name, rate in sorted(active_rates, key=lambda x: x[2], reverse=True)[:5]:
                bar_length = min(20, rate // 5)
                bar = '█' * bar_length + '░' * (20 - bar_length)
                print(f"  {emoji} {name:15} {bar} {rate:3}/30s")
        else:
            print("  ⏸️  Waiting for next collection cycle...")
        
        print("\n" + "="*100)
        print("Press Ctrl+C to exit | Updates every 30 seconds | 🟢 Active 🟡 Delayed 🔴 Stale")
    
    def run(self):
        """Main monitoring loop"""
        print("\n🚀 Starting Data Collection Monitor...")
        print("   Connecting to database...")
        
        # Get initial stats
        stats = self.get_table_stats()
        if not stats:
            print("❌ Failed to connect to database")
            return
        
        # Store initial counts
        self.start_counts = {table: data['count'] for table, data in stats.items()}
        self.previous_counts = self.start_counts.copy()
        
        print("✅ Connected! Monitoring started...")
        time.sleep(2)
        
        try:
            while True:
                # Get current stats
                stats = self.get_table_stats()
                
                # Print dashboard
                self.print_dashboard(stats)
                
                # Update previous counts
                self.previous_counts = {table: data['count'] for table, data in stats.items()}
                
                # Wait before next update (30 seconds for better granularity)
                time.sleep(30)
                
        except KeyboardInterrupt:
            print("\n\n✅ Monitoring stopped")
            
            # Print final statistics
            final_stats = self.get_table_stats()
            session_total = sum(
                data['count'] - self.start_counts.get(table, data['count'])
                for table, data in final_stats.items()
            )
            
            # Print detailed session summary
            print(f"\n📊 SESSION SUMMARY:")
            print("─" * 50)
            
            session_details = {}
            for table, data in final_stats.items():
                added = data['count'] - self.start_counts.get(table, data['count'])
                if added > 0:
                    session_details[data.get('display_name', table)] = added
            
            if session_details:
                for name, total in sorted(session_details.items(), key=lambda x: x[1], reverse=True):
                    print(f"   {name}: +{total:,} records")
                print("─" * 50)
                print(f"   Total Records Added: {session_total:,}")
            
            print(f"   Session End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*100)


def main():
    """Main entry point"""
    monitor = CollectionMonitor()
    monitor.run()


if __name__ == "__main__":
    main()