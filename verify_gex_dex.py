#!/usr/bin/env python3
import os, sys, re, json
import redis

OCC_RE = re.compile(r'^(?P<root>[A-Z]{1,6})(?P<date>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$')

def occ_parse(key: str):
    # keys: options:{SYMBOL}:{OCC}:greeks
    try:
        occ = key.split(':', 2)[2]
        m = OCC_RE.match(occ)
        if not m: return None, 0.0
        cp = m.group('cp')
        strike = float(m.group('strike')) / 1000.0
        return ('call' if cp == 'C' else 'put'), strike
    except Exception:
        return None, 0.0

def zero_gamma_strike(gex_by_strike):
    if not gex_by_strike: return None
    srt = sorted(gex_by_strike.items())
    cum = prev = 0.0
    prev_k = None
    for k, v in srt:
        prev = cum
        cum += v
        if prev_k is not None and ((prev < 0 and cum > 0) or (prev > 0 and cum < 0)):
            denom = (cum - prev)
            if denom != 0:
                w = abs(prev) / abs(denom)
                return prev_k * (1 - w) + k * w
            return k
        prev_k = k
    return None

def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/verify_gex_dex_sync.py SYMBOL SPOT")
        sys.exit(1)
    symbol = sys.argv[1].upper()
    spot = float(sys.argv[2])

    r = redis.Redis(
        host=os.getenv("REDIS_HOST", "127.0.0.1"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        db=int(os.getenv("REDIS_DB", "0")),
        password=os.getenv("REDIS_PASSWORD"),
        decode_responses=True,
    )

    pattern = f"options:{symbol}:*:greeks"
    # Use scan_iter to avoid blocking
    keys = list(r.scan_iter(match=pattern, count=2000))

    contract_mult = 100
    total_dex = 0.0
    total_gex = 0.0
    dex_by_strike = {}
    gex_by_strike = {}
    oi_by_strike  = {}

    for k in keys:
        opt_type, strike = occ_parse(k)
        if not opt_type or strike <= 0:
            continue

        t = r.type(k)
        if t == "hash":
            h = r.hgetall(k)
        else:
            raw = r.get(k)
            if not raw: 
                continue
            try:
                h = json.loads(raw)
            except Exception:
                continue

        try:
            delta = float(h.get("delta", 0) or 0)
            gamma = float(h.get("gamma", 0) or 0)
            oi    = int(float(h.get("open_interest", 0) or 0))
        except Exception:
            continue

        if oi <= 0 or gamma <= 0:
            continue

        # DEX: dollar delta (uses signed delta)
        dex = delta * oi * contract_mult * spot
        total_dex += dex
        dex_by_strike[strike] = dex_by_strike.get(strike, 0.0) + dex

        # GEX: AV gamma per $1 per share; calls +, puts -
        sign = 1.0 if opt_type == 'call' else -1.0
        gex = sign * gamma * oi * contract_mult * spot * spot
        total_gex += gex
        gex_by_strike[strike] = gex_by_strike.get(strike, 0.0) + gex

        oi_by_strike[strike] = oi_by_strike.get(strike, 0) + oi

    MIN_OI = 5
    valid = [(s, v) for s, v in gex_by_strike.items() if oi_by_strike.get(s, 0) >= MIN_OI]
    max_gex_strike = max(valid, key=lambda kv: abs(kv[1]))[0] if valid else None

    # 3-strike cluster max (optional)
    cluster_max = None
    if valid:
        vdict = dict(valid)
        srt = sorted(vdict.keys())
        cluster = {}
        for i, s in enumerate(srt):
            wnd = srt[max(0, i-1):min(len(srt), i+2)]
            cluster[s] = sum(vdict[w] for w in wnd)
        cluster_max = max(cluster.items(), key=lambda kv: abs(kv[1]))[0]

    zgs = zero_gamma_strike(gex_by_strike)

    print(f"Symbol: {symbol}  Spot: {spot}")
    print(f"Total DEX: {total_dex:,.2f}   (dollar_delta)")
    print(f"Total GEX: {total_gex:,.2f}   (dollar)")
    print(f"Max |GEX| strike (OI≥{MIN_OI}): {max_gex_strike}")
    print(f"Cluster max (3-strike): {cluster_max}")
    print(f"Zero-gamma strike (interp): {zgs}")

    top5 = sorted(gex_by_strike.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    print("\nTop 5 |GEX| strikes:")
    for s, v in top5:
        print(f"  {s:>8.2f} : {v:,.0f}")

if __name__ == "__main__":
    main()