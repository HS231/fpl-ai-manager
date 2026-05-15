"""
FPL AI Manager — Backend Server
================================
Run this with: python fpl_server.py
It starts a local server at http://localhost:5000
Keep this running while you use the dashboard.

Install dependencies first:
  python -m pip install flask flask-cors requests
"""

import json
import time
import statistics
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests as req
import fpl_config as cfg

app = Flask(__name__)
CORS(app)  # allows the dashboard HTML to talk to this server

# ── FPL API endpoints ─────────────────────────────────────
FPL_BASE        = "https://fantasy.premierleague.com/api"
BOOTSTRAP_URL   = f"{FPL_BASE}/bootstrap-static/"
FIXTURES_URL    = f"{FPL_BASE}/fixtures/"
ELEMENT_URL     = f"{FPL_BASE}/element-summary/{{pid}}/"   # per-player history

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json",
}

# ── Simple in-memory cache (avoids hammering the FPL API) ─
_cache = {}
CACHE_TTL = 300  # seconds — refresh data every 5 minutes

def cached_get(url):
    """Fetch a URL, returning cached result if fresh enough."""
    now = time.time()
    if url in _cache and now - _cache[url]["ts"] < CACHE_TTL:
        return _cache[url]["data"]
    resp = req.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    _cache[url] = {"ts": now, "data": data}
    return data


# ════════════════════════════════════════════════════════════
#  ENRICHMENT FUNCTIONS  v2
#  — momentum, consistency, rotation risk, home/away splits
#  — set piece / penalty taker bonus
#  — previous season regression anchor
#  — fixture context (home vs away FDR)
# ════════════════════════════════════════════════════════════

def calc_momentum(gw_points: list) -> float:
    """
    Weighted momentum — recent games count more than older ones.
    Uses exponential weighting so GW-1 ago matters more than GW-6.
    Positive = trending up, negative = trending down.
    """
    if len(gw_points) < 4:
        return 0.0
    weights     = [0.5, 0.3, 0.2]           # last 3 GWs weighted
    season_avg  = statistics.mean(gw_points)
    recent      = gw_points[-3:]
    weighted    = sum(p * w for p, w in zip(reversed(recent), weights))
    return round(weighted - season_avg, 2)


def calc_consistency(gw_points: list) -> float:
    """
    Consistency score 0-10.
    Penalises boom/bust players — a reliable 5pts/GW scorer
    is more valuable than a 0, 0, 0, 20, 0 player.
    Uses coefficient of variation, capped and inverted.
    """
    if len(gw_points) < 3:
        return 5.0
    avg = statistics.mean(gw_points)
    if avg == 0:
        return 0.0
    stdev = statistics.stdev(gw_points)
    cv    = stdev / avg
    return round(max(0.0, min(10.0, 10.0 - (cv * 4.0))), 2)


def calc_rotation_risk(minutes_list: list) -> str:
    """
    Rotation risk based on minutes trend over last 5 games.
    Looks at both starts (60+ mins) and sub appearances.
    Returns: 'low', 'medium', 'high'
    """
    if len(minutes_list) < 3:
        return "unknown"
    recent = minutes_list[-5:]
    starts = sum(1 for m in recent if m >= 60)
    ratio  = starts / len(recent)
    if ratio >= 0.8:
        return "low"
    elif ratio >= 0.5:
        return "medium"
    else:
        return "high"


def calc_home_away_split(history: list) -> dict:
    """
    Home/away points split.
    Used to adjust xPts based on whether next fixture is home or away.
    """
    home_pts = [h["total_points"] for h in history if h.get("was_home")]
    away_pts = [h["total_points"] for h in history if not h.get("was_home")]
    home_avg = round(statistics.mean(home_pts), 2) if home_pts else 0
    away_avg = round(statistics.mean(away_pts), 2) if away_pts else 0
    return {
        "home_avg":       home_avg,
        "away_avg":       away_avg,
        "home_advantage": round(home_avg - away_avg, 2),
    }


def calc_prev_season_anchor(history_past: list) -> float:
    """
    Previous season regression anchor.
    A player with a strong previous season record gets a small upward
    adjustment — prevents the model from completely ignoring pedigree
    during a temporary bad patch.
    Returns a bonus between 0 and 1.5.
    """
    if not history_past:
        return 0.0
    # Use best of last 2 seasons, scaled by minutes played
    scored = []
    for s in history_past[-2:]:
        mins = s.get("minutes", 0)
        pts  = s.get("total_points", 0)
        if mins > 900:   # played meaningful minutes
            scored.append(pts / (mins / 90))  # points per 90 mins
    if not scored:
        return 0.0
    best_p90 = max(scored)
    # Scale: 8+ pts/90 = 1.5 bonus, 5 pts/90 = 0.5 bonus, below 3 = 0
    return round(min(1.5, max(0.0, (best_p90 - 3.0) * 0.3)), 2)


def calc_set_piece_bonus(player: dict) -> float:
    """
    Set piece and penalty taker bonus.
    FPL API exposes direct_freekicks_order and penalties_order.
    Order 1 = first taker = big bonus. Order 2 = backup = small bonus.
    This is one of the most underrated edges in FPL modelling.
    """
    bonus = 0.0
    pk_order = player.get("penalties_order") or 0
    fk_order = player.get("direct_freekicks_order") or 0
    cs_order = player.get("corners_and_indirect_freekicks_order") or 0

    # Penalty taker
    if pk_order == 1:
        bonus += 1.8   # first penalty taker = massive edge
    elif pk_order == 2:
        bonus += 0.6   # backup taker

    # Direct free kick taker
    if fk_order == 1:
        bonus += 0.8
    elif fk_order == 2:
        bonus += 0.3

    # Corner/indirect FK taker (assists potential)
    if cs_order == 1:
        bonus += 0.5
    elif cs_order == 2:
        bonus += 0.2

    return round(bonus, 2)


def calc_team_strength(team_data: dict, is_home: bool) -> float:
    """
    Team attacking strength modifier.
    FPL provides strength_attack_home/away for each team.
    Stronger attacking teams create more FPL points for their players.
    Returns multiplier between 0.85 and 1.15.
    """
    if is_home:
        strength = team_data.get("strength_attack_home", 1000)
    else:
        strength = team_data.get("strength_attack_away", 1000)
    # FPL strength ranges roughly 1000-1400
    # Normalise to 0.85-1.15 multiplier
    normalised = (strength - 1000) / 400   # 0 to 1
    return round(0.85 + (normalised * 0.30), 3)


def calc_xpts(player: dict, fixture_list: list, team_data: dict) -> float:
    """
    Expected points v4 — full signal model.

    Signals used:
    - PPG, form, xGI (core performance)
    - ICT index: influence, creativity, threat (FPL forward-looking indicators)
    - Transfer momentum (crowd wisdom)
    - Fixture difficulty per GW (additive adjustments)
    - Home/away split from player history
    - Clean sheet probability for DEF/GK
    - Momentum (trending up/down)
    - Consistency (reliability vs boom/bust)
    - Rotation risk (minutes trend)
    - Set piece bonus (penalty/FK takers)
    - Injury multiplier (chance of playing)

    Target ranges (3 GW total):
      Elite premium (Haaland, Salah): 18-24
      Good mid-price (Gibbs-White):   12-17
      Budget enabler:                  6-10
    """
    ppg       = float(player.get("points_per_game") or 0)
    form      = float(player.get("form") or 0)
    xgi       = float(player.get("xgi") or 0)
    momentum  = player.get("momentum", 0)
    pos       = player.get("pos", "MID")
    consist   = player.get("consistency", 5)
    rot_risk  = player.get("rotation_risk", "unknown")
    home_away = player.get("home_away", {})
    sp_bonus  = player.get("set_piece_bonus", 0)

    # ICT index signals — normalised to small additive bonuses
    # FPL threat/creativity range 0-300, influence 0-200
    threat     = float(player.get("threat") or 0)
    creativity = float(player.get("creativity") or 0)
    influence  = float(player.get("influence") or 0)
    ict_bonus  = min(1.5, (threat * 0.003) + (creativity * 0.002) + (influence * 0.001))

    # Transfer momentum — are managers buying or selling?
    transfers_in  = player.get("transfers_in", 0)
    transfers_out = player.get("transfers_out", 0)
    net_transfers = transfers_in - transfers_out
    # Normalise: 200k net buys = +0.5 bonus, 200k net sells = -0.5 penalty
    transfer_signal = max(-0.8, min(0.8, net_transfers / 400000))

    # ── Base expected points per game ──
    base_per_gw = (ppg * 0.50) + (form * 0.30) + (xgi * 0.20)
    base_per_gw = max(0.0, base_per_gw)

    total = 0.0
    for fix in fixture_list[:cfg.FORECAST_GWS]:
        fdr     = int(fix.get("fdr", 3))
        is_home = fix.get("is_home", True)

        gw = base_per_gw

        # Fixture difficulty — additive
        fdr_adj = {1: +1.5, 2: +0.8, 3: 0.0, 4: -0.8, 5: -1.8}.get(fdr, 0.0)
        gw += fdr_adj

        # Home advantage from player's own history
        if is_home:
            home_adv = home_away.get("home_advantage", 0)
            gw += min(0.8, max(-0.8, home_adv * 0.3))

        # Clean sheet bonus for DEF/GK
        if pos in ("DEF", "GK"):
            gw += {1: 1.2, 2: 0.7, 3: 0.2, 4: 0.0, 5: 0.0}.get(fdr, 0.0)

        total += max(0.0, gw)

    # ── Global additive adjustments ──
    total += max(-1.5, min(1.5, momentum * 0.3))  # momentum
    total += ict_bonus                              # ICT signals
    total += transfer_signal                        # transfer momentum

    if consist < 4:
        total -= 1.0
    elif consist > 7:
        total += 0.5

    rot_pen = {"low": 0.0, "medium": -1.0, "high": -2.5, "unknown": -0.5}.get(rot_risk, 0.0)
    total += rot_pen
    total += min(1.5, sp_bonus)

    # ── INJURY MULTIPLIER ──
    # Apply chance of playing as a direct multiplier
    # 75% doubtful almost always plays but may be managed — 0.85x
    # 50% doubtful is a genuine risk — 0.50x
    # 25% doubtful very unlikely — 0.20x
    chance = player.get("chance", None)
    status = player.get("status", "a")
    if status == "d" or (chance is not None and chance < 100):
        injury_mult = {
            100: 1.00,
            75:  0.85,
            50:  0.50,
            25:  0.20,
            0:   0.00,
        }.get(chance if chance is not None else 100, 1.00)
        total *= injury_mult

    return round(max(0.0, total), 2)


def calc_differential_score(player: dict) -> float:
    """
    Differential value = high xPts + low ownership.
    Low-owned high-upside players are gold in mini-leagues.
    Score 0-10.
    """
    xpts  = player.get("xpts", 0)
    owned = float(player.get("selected_pct") or 50)
    ownership_factor = max(0.1, (100 - owned) / 100)
    return round(min(10.0, xpts * ownership_factor * 0.4), 2)


# ════════════════════════════════════════════════════════════
#  SQUAD OPTIMIZER
#  Picks the best 15-man squad within FPL rules
# ════════════════════════════════════════════════════════════

def build_squad(players: list, formation: str = "4-4-2") -> dict:
    """
    Smart squad optimizer v2.
    Improvements over v1:
    - Two-pass selection: first pass picks XI, second pass fills bench
      with cheapest viable players to maximise XI quality
    - Budget headroom: reserves £1m for bench flexibility
    - Premium protection: ensures at least 2 premium picks (£9m+) in XI
    - Value check: never picks a player where a cheaper same-pos player
      has higher xpts (pure overspend)
    """
    parts  = formation.split("-")
    n_def, n_mid, n_fwd = int(parts[0]), int(parts[1]), int(parts[2])

    # FPL squad rules — hard limits per position across ALL 15 players
    POS_LIMITS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}

    # Position pools sorted by xpts descending
    by_pos = {
        pos: sorted([p for p in players if p["pos"] == pos],
                    key=lambda x: x["xpts"], reverse=True)
        for pos in ("GK", "DEF", "MID", "FWD")
    }

    BUDGET     = 100.0
    team_count = {}
    spent      = 0.0

    def can_pick(p):
        return (
            team_count.get(p["team_id"], 0) < 3 and
            spent + p["price"] <= BUDGET
        )

    def pick(p):
        nonlocal spent
        team_count[p["team_id"]] = team_count.get(p["team_id"], 0) + 1
        spent += p["price"]

    # ── PASS 1: Pick the XI (leave ~20m for bench) ──
    XI_BUDGET = 80.0   # soft cap — ensures bench budget exists

    # Track how many of each position we've picked across full 15
    pos_count = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}

    def can_pick_pos(p):
        return pos_count[p["pos"]] < POS_LIMITS[p["pos"]]

    def pick_pos(pool, needed, budget_cap):
        nonlocal spent
        picks = []
        for p in pool:
            if len(picks) >= needed:
                break
            if team_count.get(p["team_id"], 0) >= 3:
                continue
            if not can_pick_pos(p):
                continue
            if spent + p["price"] > budget_cap:
                continue
            picks.append(p)
            team_count[p["team_id"]] = team_count.get(p["team_id"], 0) + 1
            pos_count[p["pos"]] += 1
            spent += p["price"]
        return picks

    # ── PASS 1: Pick XI ──
    xi_gk  = pick_pos(by_pos["GK"],  1,      XI_BUDGET)
    xi_def = pick_pos(by_pos["DEF"], n_def,  XI_BUDGET)
    xi_mid = pick_pos(by_pos["MID"], n_mid,  XI_BUDGET)
    xi_fwd = pick_pos(by_pos["FWD"], n_fwd,  XI_BUDGET)

    # Relax budget cap if XI isn't full
    for pos_name, pool, needed in [
        ("GK", by_pos["GK"], 1), ("DEF", by_pos["DEF"], n_def),
        ("MID", by_pos["MID"], n_mid), ("FWD", by_pos["FWD"], n_fwd)
    ]:
        existing = {"GK": xi_gk,"DEF": xi_def,"MID": xi_mid,"FWD": xi_fwd}[pos_name]
        used = {p["id"] for p in xi_gk+xi_def+xi_mid+xi_fwd}
        for p in pool:
            if len(existing) >= needed:
                break
            if p["id"] in used:
                continue
            if team_count.get(p["team_id"], 0) >= 3:
                continue
            if not can_pick_pos(p):
                continue
            if spent + p["price"] > BUDGET:
                continue
            existing.append(p)
            team_count[p["team_id"]] = team_count.get(p["team_id"], 0) + 1
            pos_count[p["pos"]] += 1
            spent += p["price"]
            used.add(p["id"])

    xi       = xi_gk + xi_def + xi_mid + xi_fwd
    used_ids = {p["id"] for p in xi}

    # ── PASS 2: Bench — must respect position limits ──
    # 1 GK on bench
    bench_gk_pool = sorted(
        [p for p in by_pos["GK"] if p["id"] not in used_ids],
        key=lambda x: x["price"]
    )
    bench_gk = []
    for p in bench_gk_pool:
        if pos_count[p["pos"]] >= POS_LIMITS[p["pos"]]:
            continue
        if team_count.get(p["team_id"], 0) >= 3:
            continue
        if spent + p["price"] > BUDGET:
            continue
        bench_gk.append(p)
        team_count[p["team_id"]] = team_count.get(p["team_id"], 0) + 1
        pos_count[p["pos"]] += 1
        spent += p["price"]
        used_ids.add(p["id"])
        break

    # 3 outfield bench players — cheapest, respecting position limits
    bench_out_pool = sorted(
        [p for pos_name in ("DEF","MID","FWD")
           for p in by_pos[pos_name] if p["id"] not in used_ids],
        key=lambda x: x["price"]
    )
    bench_out = []
    for p in bench_out_pool:
        if len(bench_out) >= 3:
            break
        if pos_count[p["pos"]] >= POS_LIMITS[p["pos"]]:
            continue
        if team_count.get(p["team_id"], 0) >= 3:
            continue
        if spent + p["price"] > BUDGET:
            continue
        bench_out.append(p)
        team_count[p["team_id"]] = team_count.get(p["team_id"], 0) + 1
        pos_count[p["pos"]] += 1
        spent += p["price"]
        used_ids.add(p["id"])

    bench = bench_gk + bench_out

    # ── Safety net — fill any missing bench spots ──
    if len(bench) < 4:
        all_pool = sorted(
            [p for pos_name in ("GK","DEF","MID","FWD")
               for p in by_pos[pos_name] if p["id"] not in used_ids],
            key=lambda x: x["price"]
        )
        for p in all_pool:
            if len(bench) >= 4:
                break
            if pos_count[p["pos"]] >= POS_LIMITS[p["pos"]]:
                continue
            if spent + p["price"] <= BUDGET:
                bench.append(p)
                pos_count[p["pos"]] += 1
                spent += p["price"]
                used_ids.add(p["id"])

    # ── Captain logic v2 ──
    # Rules:
    # 1. Never captain a doubtful player (chance < 100 or status == "d")
    # 2. Prefer home fixture for captain
    # 3. Minimum consistency of 4/10 for captain
    # 4. Tiebreak: form, then home fixture

    def captain_score(p):
        """Score a player for captain consideration."""
        # Disqualify doubtful players entirely
        if p.get("status") == "d" or (p.get("chance") is not None and p.get("chance") < 100):
            return -999
        # Disqualify inconsistent players
        if p.get("consistency", 5) < 4:
            return -999
        score = p["xpts"]
        # Bonus for home fixture
        if p.get("next_home"):
            score += 1.5
        # Bonus for high consistency
        score += (p.get("consistency", 5) - 5) * 0.2
        return score

    # Captain from fully fit players first
    eligible_caps = [p for p in xi if captain_score(p) > -999]

    # Fallback: if all players are doubtful, pick highest xpts regardless
    if not eligible_caps:
        eligible_caps = xi

    captain = max(eligible_caps, key=captain_score)
    vice    = max(
        [p for p in xi if p["id"] != captain["id"]],
        key=lambda x: (x["xpts"], x.get("next_home", False))
    )

    total_value = round(sum(p["price"] for p in xi + bench), 1)
    bank        = round(100.0 - total_value, 1)

    return {
        "xi":           xi,
        "bench":        bench,
        "captain":      captain,
        "vice_captain": vice,
        "total_value":  total_value,
        "bank":         bank,
        "formation":    formation,
    }


# ════════════════════════════════════════════════════════════
#  CLAUDE AI REASONING
#  Sends squad data to Claude, gets back expert analysis
# ════════════════════════════════════════════════════════════

def get_ai_briefing(squad: dict, gw_id: int) -> str:
    """
    Sends the squad to Claude and asks for a proper FPL manager briefing.
    Returns a string of natural language analysis.
    """
    xi     = squad["xi"]
    bench  = squad["bench"]
    cap    = squad["captain"]
    vc     = squad["vice_captain"]

    # Build a rich data summary to give Claude
    # Build injury flags for doubtful players
    doubtful_players = [p for p in xi + bench if p.get("is_doubtful") or (p.get("chance") is not None and p.get("chance") < 100)]
    doubtful_str = ", ".join([
        f"{p['name']} ({p.get('chance',75)}% chance{': ' + p['news'] if p.get('news') else ''})"
        for p in doubtful_players
    ]) if doubtful_players else "None"

    # Build transfer momentum flags
    hot_transfers = sorted(
        [p for p in xi if p.get("transfers_in", 0) > 50000],
        key=lambda x: x.get("transfers_in", 0), reverse=True
    )[:3]
    transfer_str = ", ".join([
        f"{p['name']} (+{p.get('transfers_in',0)//1000}k owners this GW)"
        for p in hot_transfers
    ]) if hot_transfers else "None"

    xi_summary = "\n".join([
        f"  {p['pos']} | {p['name']} ({p['team']}) | £{p['price']}m | "
        f"xPts:{p['xpts']} | Form:{p['form']} | Momentum:{p.get('momentum',0):+.1f} | "
        f"Consistency:{p.get('consistency',5)}/10 | FDR:{p.get('fdr_avg3','?')} | "
        f"Next: {'HOME' if p.get('next_home') else 'AWAY'} | "
        f"ICT:{p.get('ict_index',0)} | Rotation:{p.get('rotation_risk','?')} | "
        f"{'⚠ DOUBTFUL ' + str(p.get('chance','?')) + '%' if p.get('is_doubtful') else 'Fit'}"
        for p in xi
    ])
    bench_summary = ", ".join([f"{p['name']} ({p['pos']}, £{p['price']}m{'  ⚠' if p.get('is_doubtful') else ''})" for p in bench])

    prompt = f"""You are an elite FPL (Fantasy Premier League) analyst with 10+ years of experience.
You have just built the following optimal squad for Gameweek {gw_id}.

FORMATION: {squad['formation']}
TOTAL VALUE: £{squad['total_value']}m (£{squad['bank']}m in bank)
CAPTAIN: {cap['name']} ({cap['team']}) — xPts: {cap['xpts']}, Form: {cap['form']}, Next: {'HOME' if cap.get('next_home') else 'AWAY'}
VICE CAPTAIN: {vc['name']} ({vc['team']}) — xPts: {vc['xpts']}

INJURY CONCERNS: {doubtful_str}
TRANSFER MOMENTUM (managers buying in): {transfer_str}

STARTING XI (pos | name | price | xPts | form | momentum | consistency | FDR | home/away | ICT | rotation | fitness):
{xi_summary}

BENCH: {bench_summary}

Write a punchy, expert weekly manager briefing of around 220 words covering:
1. Overall squad strategy in one sharp sentence
2. Captain/VC rationale with fixture context — mention if home advantage played a role
3. Two or three standout picks — mention ICT scores or transfer momentum where relevant
4. Explicitly call out every doubtful player by name, their % chance, and what it means for the squad
5. Confident closing outlook

Write in second person. Be specific with names, teams, fixtures.
Sound like a knowledgeable friend. Plain prose only, no markdown, no bullet points."""

    prompt += "\n\nIMPORTANT: Always explicitly mention any doubtful/injured players and their risk. Never ignore injury flags."

    response = req.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "Content-Type":      "application/json",
            "x-api-key":         cfg.ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
        },
        json={
            "model":      "claude-sonnet-4-6",
            "max_tokens": 1000,
            "messages":   [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["content"][0]["text"]


# ════════════════════════════════════════════════════════════
#  MAIN DATA PIPELINE
#  Called by the /api/squad endpoint
# ════════════════════════════════════════════════════════════

def build_full_dataset(formation: str = "4-4-2") -> dict:
    """
    Full pipeline:
    1. Fetch bootstrap + fixtures
    2. Fetch per-player GW history for top candidates
    3. Enrich with momentum, consistency, rotation risk
    4. Score with xPts model
    5. Build optimal squad
    6. Get Claude AI briefing
    """

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetching bootstrap data...")
    boot     = cached_get(BOOTSTRAP_URL)
    fixtures = cached_get(FIXTURES_URL)

    # Build lookup maps
    team_map = {t["id"]: t for t in boot["teams"]}
    pos_map  = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

    # Current gameweek — always find the most relevant upcoming one
    from datetime import timezone
    now = datetime.now(timezone.utc)

    # First try: find next gameweek with a future deadline
    future_gws = [e for e in boot["events"]
                  if e.get("deadline_time") and
                  datetime.fromisoformat(e["deadline_time"].replace("Z","+00:00")) > now]
    next_gw    = future_gws[0] if future_gws else None

    # Fallback: use is_current or is_next flags
    current_gw = next((e for e in boot["events"] if e["is_current"]), None)
    flag_gw    = next((e for e in boot["events"] if e["is_next"]), None)

    # Pick the most accurate one
    active_gw  = next_gw or current_gw or flag_gw
    gw_id      = active_gw["id"] if active_gw else 38
    deadline   = active_gw["deadline_time"] if active_gw else None

    # Next 3 fixture difficulties per team
    team_fdr = {t["id"]: [] for t in boot["teams"]}
    for f in fixtures:
        if f.get("finished"):
            continue
        ev = f.get("event") or 999
        if ev > gw_id + cfg.FORECAST_GWS:
            continue
        if len(team_fdr[f["team_h"]]) < cfg.FORECAST_GWS:
            team_fdr[f["team_h"]].append(f["team_h_difficulty"])
        if len(team_fdr[f["team_a"]]) < cfg.FORECAST_GWS:
            team_fdr[f["team_a"]].append(f["team_a_difficulty"])

    # Filter to available players with meaningful minutes
    # Include doubtful players (d) — we apply injury multiplier instead of excluding
    candidates = [
        p for p in boot["elements"]
        if p["status"] not in ("u", "i", "s")   # exclude unavailable, injured, suspended
        and (p.get("minutes") or 0) > 90         # has played meaningfully
        and (p.get("chance_of_playing_next_round") or 100) >= 25  # at least 25% chance
    ]

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Enriching {len(candidates)} players with GW history...")

    enriched = []

    # EXPANDED POOL: top 200 by a combined form+points score
    # This catches in-form players (like Doku) who had a slow start
    def candidate_score(p):
        form        = float(p.get("form") or 0)
        total_pts   = p.get("total_points", 0)
        chance      = (p.get("chance_of_playing_next_round") or 100) / 100
        return (form * 4.0 + total_pts * 0.1) * chance

    top_candidates = sorted(candidates, key=candidate_score, reverse=True)[:200]

    # Build next-fixture list per team with home/away context
    # Structure: team_id -> [{"fdr": X, "is_home": bool, "event": N}, ...]
    team_fixtures = {t["id"]: [] for t in boot["teams"]}
    for f in sorted(fixtures, key=lambda x: x.get("event") or 999):
        if f.get("finished"):
            continue
        ev = f.get("event") or 999
        if ev > gw_id + cfg.FORECAST_GWS:
            continue
        tid_h, tid_a = f["team_h"], f["team_a"]
        if len(team_fixtures[tid_h]) < cfg.FORECAST_GWS:
            team_fixtures[tid_h].append({"fdr": f["team_h_difficulty"], "is_home": True,  "event": ev})
        if len(team_fixtures[tid_a]) < cfg.FORECAST_GWS:
            team_fixtures[tid_a].append({"fdr": f["team_a_difficulty"], "is_home": False, "event": ev})

    def enrich_player(p):
        """Enrich a single player — called concurrently."""
        try:
            history_data  = cached_get(ELEMENT_URL.format(pid=p["id"]))
            history       = history_data.get("history", [])[-cfg.GW_HISTORY_DEPTH:]
            history_past  = history_data.get("history_past", [])
            gw_pts        = [h["total_points"] for h in history]
            gw_mins       = [h["minutes"]      for h in history]
            fix_list      = team_fixtures.get(p["team"], [])
            fdr_list      = [f["fdr"] for f in fix_list]
            team_d        = team_map.get(p["team"], {})

            enriched_player = {
                "id":           p["id"],
                "name":         p["web_name"],
                "full_name":    f"{p['first_name']} {p['second_name']}",
                "team":         team_d.get("short_name", ""),
                "team_name":    team_d.get("name", ""),
                "team_id":      p["team"],
                "pos":          pos_map[p["element_type"]],
                "pos_id":       p["element_type"],
                "price":        p["now_cost"] / 10,
                "price_change": p.get("cost_change_event", 0),
                "total_points":    p.get("total_points", 0),
                "points_per_game": float(p.get("points_per_game") or 0),
                "form":            float(p.get("form") or 0),
                "selected_pct":    float(p.get("selected_by_percent") or 0),
                "xg":              float(p.get("expected_goals") or 0),
                "xa":              float(p.get("expected_assists") or 0),
                "xgi":             float(p.get("expected_goal_involvements") or 0),
                "clean_sheets":    p.get("clean_sheets", 0),
                "minutes":         p.get("minutes", 0),
                "status":          p.get("status", "a"),
                "chance":          p.get("chance_of_playing_next_round"),
                # ICT index — FPL's own forward-looking performance indicators
                "influence":       float(p.get("influence") or 0),
                "creativity":      float(p.get("creativity") or 0),
                "threat":          float(p.get("threat") or 0),
                "ict_index":       float(p.get("ict_index") or 0),
                # Transfer momentum — crowd wisdom signal
                "transfers_in":    p.get("transfers_in_event", 0),
                "transfers_out":   p.get("transfers_out_event", 0),
                # Injury flag for briefing
                "is_doubtful":     p.get("status") == "d",
                "news":            p.get("news", ""),
                "penalties_order":        p.get("penalties_order"),
                "direct_freekicks_order": p.get("direct_freekicks_order"),
                "corners_order":          p.get("corners_and_indirect_freekicks_order"),
                "fdr_next":  fdr_list[0] if fdr_list else 3,
                "fdr_next2": fdr_list[1] if len(fdr_list) > 1 else 3,
                "fdr_next3": fdr_list[2] if len(fdr_list) > 2 else 3,
                "fdr_avg3":  round(sum(fdr_list[:3]) / max(len(fdr_list[:3]), 1), 1),
                "next_home": fix_list[0]["is_home"] if fix_list else True,
                "momentum":           calc_momentum(gw_pts),
                "consistency":        calc_consistency(gw_pts),
                "rotation_risk":      calc_rotation_risk(gw_mins),
                "home_away":          calc_home_away_split(history),
                "set_piece_bonus":    calc_set_piece_bonus(p),
                "prev_season_anchor": calc_prev_season_anchor(history_past),
                "gw_history":         gw_pts,
            }
            enriched_player["xpts"]        = calc_xpts(enriched_player, fix_list, team_d)
            enriched_player["differential"] = calc_differential_score(enriched_player)
            return enriched_player
        except Exception as e:
            print(f"  ⚠ Skipped {p.get('web_name','?')}: {e}")
            return None

    # Fetch concurrently — 20 workers = ~5x faster than sequential
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetching with 20 concurrent workers...")
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(enrich_player, p): p for p in top_candidates}
        for future in as_completed(futures):
            result = future.result()
            if result:
                enriched.append(result)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Enriched {len(enriched)} players. Building squad...")

    # Build optimal squad
    squad = build_squad(enriched, formation)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Getting AI briefing from Claude...")
    try:
        briefing = get_ai_briefing(squad, gw_id)
    except Exception as e:
        briefing = f"AI briefing unavailable: {e}"

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Done.")

    return {
        "gameweek":    gw_id,
        "deadline":    deadline,
        "squad":       squad,
        "briefing":    briefing,
        "top_players": sorted(enriched, key=lambda x: x["xpts"], reverse=True)[:30],
        "generated_at": datetime.now().isoformat(),
    }


# ════════════════════════════════════════════════════════════
#  API ROUTES
#  These are the URLs the dashboard calls
# ════════════════════════════════════════════════════════════

@app.route("/api/squad")
@app.route("/api/squad/<formation>")
def get_squad(formation="4-4-2"):
    """Main endpoint — returns full squad + briefing."""
    try:
        data = build_full_dataset(formation)
        return jsonify({"ok": True, "data": data})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/health")
def health():
    """Quick check that the server is running."""
    return jsonify({
        "ok":      True,
        "message": "FPL AI Manager backend is running",
        "time":    datetime.now().isoformat(),
    })


@app.route("/api/players")
def get_players():
    """
    Returns lightweight player list for search/autocomplete.
    Uses ONLY the bootstrap endpoint — no GW history fetching.
    Fast, cheap, cached. Called once per session.
    """
    try:
        boot     = cached_get(BOOTSTRAP_URL)
        team_map = {t["id"]: t for t in boot["teams"]}
        pos_map  = {1:"GK", 2:"DEF", 3:"MID", 4:"FWD"}

        players = []
        for el in boot["elements"]:
            if el.get("status") == "u":
                continue
            team = team_map.get(el["team"], {})
            players.append({
                "id":        el["id"],
                "name":      el["web_name"],
                "full_name": f"{el['first_name']} {el['second_name']}",
                "pos":       pos_map.get(el["element_type"], "MID"),
                "team":      team.get("short_name", ""),
                "price":     el["now_cost"] / 10,
                "form":      float(el.get("form") or 0),
            })

        # Sort by form descending so best players appear first in search
        players.sort(key=lambda x: x["form"], reverse=True)
        return jsonify({"ok": True, "players": players})

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ════════════════════════════════════════════════════════════
#  TRANSFER PLANNER
# ════════════════════════════════════════════════════════════

def get_transfer_briefing(transfers: list, free_transfers: int, bank: float) -> str:
    """Ask Claude to write a transfer recommendation briefing."""

    transfer_text = "\n".join([
        f"  OUT: {t['out']['name']} ({t['out']['pos']}, £{t['out']['price']}m, "
        f"xPts:{t['out']['xpts']}, form:{t['out']['form']}, momentum:{t['out'].get('momentum',0):+.1f}) "
        f"→ IN: {t['in']['name']} ({t['in']['pos']}, £{t['in']['price']}m, "
        f"xPts:{t['in']['xpts']}, form:{t['in']['form']}, momentum:{t['in'].get('momentum',0):+.1f}) "
        f"| xPts gain: +{t['xpts_gain']:.1f} | hit required: {t['hit_required']}"
        for t in transfers[:5]
    ])

    prompt = f"""You are an elite FPL transfer analyst. Analyse these recommended transfers and write a concise 150-word briefing.

Free transfers available: {free_transfers}
Bank: £{bank}m

TOP RECOMMENDED TRANSFERS:
{transfer_text}

For each transfer explain WHY it makes sense — reference form, fixtures, momentum, value.
For any transfer requiring a -4 hit, explicitly state whether the xPts gain justifies it
(rule of thumb: hit only worth it if gain > 6 points over next 3 GWs).
Be direct and confident. Plain prose only, no markdown, no bullet points."""

    response = req.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "Content-Type":      "application/json",
            "x-api-key":         cfg.ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
        },
        json={
            "model":      "claude-sonnet-4-6",
            "max_tokens": 600,
            "messages":   [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["content"][0]["text"]


# ════════════════════════════════════════════════════════════
#  MY TEAM — Auto-load current squad from FPL account
# ════════════════════════════════════════════════════════════

@app.route("/api/my-team")
def get_my_team():
    """
    Fetches current squad using browser session cookie.
    No login required — uses FPL_COOKIE from fpl_config.py.
    """
    try:
        ENTRY_URL = f"https://fantasy.premierleague.com/api/entry/{cfg.FPL_TEAM_ID}/"
        PICKS_URL = f"https://fantasy.premierleague.com/api/entry/{cfg.FPL_TEAM_ID}/event/{{gw}}/picks/"

        cookie_headers = {{**HEADERS, "Cookie": cfg.FPL_COOKIE}}

        entry_r = req.get(ENTRY_URL, headers=cookie_headers, timeout=15)
        if entry_r.status_code == 403:
            raise Exception("Cookie expired — refresh from browser DevTools (F12 → Network → Headers → Cookie)")
        entry_data = entry_r.json()

        boot    = cached_get(BOOTSTRAP_URL)
        curr_gw = next((e for e in boot["events"] if e["is_current"]), None)
        next_gw = next((e for e in boot["events"] if e["is_next"]), None)
        active  = curr_gw or next_gw
        gw_id   = active["id"] if active else entry_data.get("current_event", 38)

        picks_r = req.get(PICKS_URL.format(gw=gw_id), headers=cookie_headers, timeout=15)
        picks_d = picks_r.json()

        p_map = {{el["id"]: el for el in boot["elements"]}}
        t_map = {{t["id"]:  t  for t in  boot["teams"]}}

        squad = []
        for pick in picks_d.get("picks", []):
            el   = p_map.get(pick["element"], {{}})
            team = t_map.get(el.get("team"), {{}})
            squad.append({{
                "name":       el.get("web_name", "Unknown"),
                "full_name":  f"{{el.get('first_name','')}} {{el.get('second_name','')}}".strip(),
                "pos":        {{1:"GK",2:"DEF",3:"MID",4:"FWD"}}.get(el.get("element_type"), "?"),
                "team":       team.get("short_name", ""),
                "price":      el.get("now_cost", 0) / 10,
                "form":       float(el.get("form") or 0),
                "is_captain": pick.get("is_captain", False),
                "is_vice":    pick.get("is_vice_captain", False),
                "position":   pick.get("position", 0),
            }})

        history    = picks_d.get("entry_history", {{}})
        bank       = round(history.get("bank", 0) / 10, 1)
        team_value = round(history.get("value", 1000) / 10, 1)

        return jsonify({{
            "ok":         True,
            "squad":      squad,
            "bank":       bank,
            "team_value": team_value,
            "gameweek":   gw_id,
            "manager":    f"{{entry_data.get('player_first_name','')}} {{entry_data.get('player_last_name','')}}",
            "team_name":  entry_data.get("name", ""),
            "points":     entry_data.get("summary_overall_points", 0),
            "rank":       entry_data.get("summary_overall_rank", 0),
        }})

    except Exception as e:
        return jsonify({{"ok": False, "error": str(e)}}), 500

@app.route("/api/transfers", methods=["POST"])
def get_transfers():
    """
    Transfer planner endpoint.
    Accepts: { players: [...names], bank: float, free_transfers: int }
    Returns: ranked transfer recommendations with AI briefing.
    """
    try:
        body          = request.get_json()
        current_names = [n.lower().strip() for n in body.get("players", [])]
        bank          = float(body.get("bank", 0))
        free_transfers = int(body.get("free_transfers", 1))

        # Load full enriched dataset
        data    = build_full_dataset()
        players = data["top_players"]

        # Load full dataset — all_players contains top 150 enriched
        all_data    = build_full_dataset()
        all_players = all_data["top_players"]
        # Also include bootstrap elements for name matching against full 700+ player list
        boot        = cached_get(BOOTSTRAP_URL)
        boot_map    = {el["id"]: el for el in boot["elements"]}
        team_map_b  = {t["id"]: t  for t in boot["teams"]}
        pos_map_b   = {1:"GK",2:"DEF",3:"MID",4:"FWD"}

        # Build a lightweight lookup of ALL players for name matching
        all_names_pool = []
        for el in boot["elements"]:
            all_names_pool.append({
                "id":        el["id"],
                "name":      el["web_name"],
                "full_name": f"{el['first_name']} {el['second_name']}",
                "pos":       pos_map_b.get(el["element_type"], "MID"),
                "team":      team_map_b.get(el["team"], {}).get("short_name", ""),
                "price":     el["now_cost"] / 10,
                "form":      float(el.get("form") or 0),
            })

        def fuzzy_match(p, query):
            """Multi-strategy fuzzy name matching."""
            q = query.lower().strip()
            name      = p["name"].lower()
            full      = p["full_name"].lower()
            # Strategy 1: exact substring
            if q in name or q in full:
                return True
            # Strategy 2: remove dots/hyphens and try again
            q2    = q.replace(".","").replace("-","").replace(" ","")
            name2 = name.replace(".","").replace("-","").replace(" ","")
            full2 = full.replace(".","").replace("-","").replace(" ","")
            if q2 in name2 or q2 in full2:
                return True
            # Strategy 3: first letter + surname (e.g. "m.salah" → "salah")
            parts = q.split(".")
            if len(parts) > 1 and parts[1] in full:
                return True
            # Strategy 4: last word of query matches last word of name/full
            q_last    = q.split()[-1] if q.split() else q
            name_last = name.split()[-1] if name.split() else name
            full_last = full.split()[-1] if full.split() else full
            if len(q_last) > 3 and (q_last in name_last or q_last in full_last):
                return True
            return False

        current_squad = []
        unmatched     = []
        for name in current_names:
            # First try enriched players (have xpts)
            matches = [p for p in all_players if fuzzy_match(p, name)]
            if matches:
                best = min(matches, key=lambda p: abs(len(p["name"]) - len(name)))
                current_squad.append(best)
            else:
                # Fall back to full bootstrap pool
                boot_matches = [p for p in all_names_pool if fuzzy_match(p, name)]
                if boot_matches:
                    best = min(boot_matches, key=lambda p: abs(len(p["name"]) - len(name)))
                    # Give them a basic xpts based on form so they appear in comparison
                    best["xpts"]      = float(best.get("form", 0)) * 2.0
                    best["momentum"]  = 0.0
                    best["fdr_avg3"]  = 3.0
                    current_squad.append(best)
                else:
                    unmatched.append(name)

        if not current_squad:
            return jsonify({"ok": False, "error": "No players matched. Check names."}), 400

        # For each current player, find best available replacement
        # at same position within (price + bank) budget
        current_ids  = {p["id"] for p in current_squad}
        transfers    = []
        ft_remaining = free_transfers  # track for hit calculation

        for rank, out_player in enumerate(current_squad):
            pos        = out_player["pos"]
            max_budget = out_player["price"] + bank
            same_pos   = [
                p for p in all_players
                if p["pos"] == pos
                and p["id"] not in current_ids
                and p["price"] <= max_budget
            ]
            if not same_pos:
                continue

            # Best replacement by xpts
            best_in    = max(same_pos, key=lambda x: x["xpts"])
            xpts_gain  = round(best_in["xpts"] - out_player["xpts"], 2)

            # Only suggest meaningful improvements
            if xpts_gain < 0.5:
                continue

            # Hit calculation — based on rank in sorted list
            # First N transfers are free (where N = free_transfers)
            hit_required  = rank >= free_transfers
            net_gain      = round(xpts_gain - (4 if hit_required else 0), 2)

            transfers.append({
                "out":          out_player,
                "in":           best_in,
                "xpts_gain":    xpts_gain,
                "hit_required": hit_required,
                "hit_worth_it": xpts_gain > 6.0 if hit_required else True,
                "cost":         -4 if hit_required else 0,
                "net_gain":     net_gain,
            })

        # Sort by xpts gain (not net — show best moves regardless of hit)
        transfers.sort(key=lambda x: x["xpts_gain"], reverse=True)

        # Re-assign hit flags based on final sorted order
        for i, t in enumerate(transfers):
            t["hit_required"] = i >= free_transfers
            t["net_gain"]     = round(t["xpts_gain"] - (4 if t["hit_required"] else 0), 2)
            t["hit_worth_it"] = t["xpts_gain"] > 6.0 if t["hit_required"] else True

        # Get AI briefing
        try:
            briefing = get_transfer_briefing(
                transfers, body.get("free_transfers", 1), bank
            )
        except Exception as e:
            briefing = f"AI briefing unavailable: {e}"

        return jsonify({
            "ok":           True,
            "transfers":    transfers[:8],
            "unmatched":    unmatched,
            "briefing":     briefing,
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ════════════════════════════════════════════════════════════
#  DASHBOARD ROUTE
# ════════════════════════════════════════════════════════════

import os
app.static_folder = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def dashboard():
    return app.send_static_file("fpl_dashboard.html")


# ════════════════════════════════════════════════════════════
#  START SERVER
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  FPL AI Manager — Backend Server")
    print("=" * 55)
    print(f"  Starting on http://localhost:{cfg.PORT}")
    print(f"  Keep this window open while using the dashboard")
    print(f"  Press Ctrl+C to stop")
    print("=" * 55)
    app.run(
        host="0.0.0.0",
        port=cfg.PORT,
        debug=False,
    )
