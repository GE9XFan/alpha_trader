#!/bin/bash
# Complete Redis cleanup for QuantiCity Capital trading state

echo "==================================================="
echo "COMPLETE REDIS CLEANUP - QuantiCity Capital"
echo "==================================================="
echo ""
echo "This will DELETE:"
echo "  - All open/closed positions"
echo "  - All signal queues and data"
echo "  - All pending orders and fills"
echo "  - All signal tracking/deduplication data"
echo ""

# Redis CLI command
REDIS_CLI="redis-cli"

# Function to delete keys by pattern and show count
delete_pattern() {
    local pattern=$1
    local description=$2

    echo -n "Deleting $description ($pattern)... "

    # Get count first
    local count=$($REDIS_CLI --scan --pattern "$pattern" 2>/dev/null | wc -l)

    if [ "$count" -gt 0 ]; then
        # Delete using xargs for efficiency
        $REDIS_CLI --scan --pattern "$pattern" 2>/dev/null | xargs -r -L 100 $REDIS_CLI DEL > /dev/null 2>&1
        echo "✓ Deleted $count keys"
    else
        echo "✓ No keys found"
    fi
}

echo "Starting cleanup..."
echo ""

# 1. POSITIONS
echo "1. Cleaning Positions..."
delete_pattern "positions:open:*" "open positions"
delete_pattern "positions:by_symbol:*" "position indices"
delete_pattern "positions:closed:*" "closed positions"
delete_pattern "positions:*" "other position data"
echo ""

# 2. SIGNAL QUEUES
echo "2. Cleaning Signal Queues..."
delete_pattern "signals:execution:*" "execution queues"
delete_pattern "signals:pending:*" "pending queues"
delete_pattern "signals:out:*" "outbound signals"
delete_pattern "signals:latest:*" "latest signal cache"

# Direct queue deletions (not pattern-based)
echo -n "Deleting distribution queues... "
$REDIS_CLI DEL distribution:premium:queue distribution:basic:queue distribution:free:queue > /dev/null 2>&1
echo "✓ Done"
echo ""

# 3. SIGNAL TRACKING
echo "3. Cleaning Signal Tracking..."
delete_pattern "signals:emitted:*" "emitted signal IDs"
delete_pattern "signals:cooldown:*" "cooldown keys"
delete_pattern "signals:last_conf:*" "confidence tracking"
delete_pattern "signals:last_contract:*" "contract hysteresis"
delete_pattern "signals:audit:*" "audit trails"
delete_pattern "signals:debug:*" "debug data"
echo ""

# 4. ORDERS
echo "4. Cleaning Orders..."
delete_pattern "orders:pending:*" "pending orders"
delete_pattern "orders:fills:*" "fill history"
delete_pattern "orders:*" "other order data"
echo ""

# 5. EXECUTION STATE
echo "5. Cleaning Execution State..."
delete_pattern "execution:fills:*" "fill counters"
delete_pattern "execution:commission:*" "commission data"
delete_pattern "execution:rejections:*" "rejection data"
delete_pattern "execution:errors:*" "error counters"
delete_pattern "execution:timeouts:*" "timeout counters"
delete_pattern "contracts:qualified:*" "qualified contracts"
echo ""

# 6. VERIFICATION
echo "==================================================="
echo "VERIFICATION"
echo "==================================================="
echo ""

echo "Remaining relevant keys:"
echo -n "  Positions: "
$REDIS_CLI --scan --pattern "positions:*" 2>/dev/null | wc -l
echo -n "  Signals: "
$REDIS_CLI --scan --pattern "signals:*" 2>/dev/null | wc -l
echo -n "  Orders: "
$REDIS_CLI --scan --pattern "orders:*" 2>/dev/null | wc -l
echo -n "  Execution queues: "
for symbol in SPY QQQ IWM; do
    len=$($REDIS_CLI LLEN "signals:execution:$symbol" 2>/dev/null)
    if [ "$len" != "0" ] && [ "$len" != "" ]; then
        echo -n "$symbol:$len "
    fi
done
echo ""
echo ""

echo "==================================================="
echo "CLEANUP COMPLETE!"
echo "==================================================="
echo ""
echo "NOTE: This did NOT clear:"
echo "  - Market data (market:*)"
echo "  - Analytics (analytics:*)"
echo "  - Metrics (metrics:*)"
echo "  - Risk settings (risk:*)"
echo "  - Health/heartbeats (health:*)"
echo "  - Account data (account:*)"
echo ""
echo "The system can now accept new signals without position limits!"