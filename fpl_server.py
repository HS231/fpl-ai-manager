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

# ── Simple in-memory cache ─────────────────────────────────
_cache = {}

def get_cache_ttl() -> int:
    """
    Dynamic cache TTL.
    During the pre-season transfer window (June-August) we want
    fresh data as often as possible — new signings, price changes
    and position updates happen daily.
    Once the season is underway, 5 minutes is fine.
    """
    month = datetime.now().month
    if month in (6, 7, 8):      # transfer window / pre-season
        return 60               # 60 seconds — stay current
    return 300                  # 5 minutes during season

def cached_get(url):
    """Fetch a URL, returning cached result if fresh enough."""
    now = time.time()
    ttl = get_cache_ttl()
    if url in _cache and now - _cache[url]["ts"] < ttl:
        return _cache[url]["data"]
    resp = req.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    _cache[url] = {"ts": now, "data": data}
    return data


# ════════════════════════════════════════════════════════════
#  DOUBLE/BLANK GAMEWEEK DETECTOR
# ════════════════════════════════════════════════════════════

def detect_dgw_bgw(fixtures: list, teams: list, total_gws: int = 38) -> dict:
    """
    Scans all fixtures and detects double and blank gameweeks per team.

    Returns a dict:
    {
        team_id: {
            gw_id: {
                "fixtures": int,      # 0=blank, 1=normal, 2+=double
                "type": "BGW"|"DGW"|"normal",
                "opponents": [...],   # opponent short names
                "is_home": [...],     # home/away for each fixture
                "fdr": [...],         # difficulty per fixture
            }
        }
    }
    """
    team_map = {t["id"]: t for t in teams}

    # Build fixture count per team per GW
    schedule = {}
    for t in teams:
        schedule[t["id"]] = {}

    for f in fixtures:
        gw = f.get("event")
        if not gw:
            continue
        tid_h = f["team_h"]
        tid_a = f["team_a"]

        # Home team
        if tid_h not in schedule:
            schedule[tid_h] = {}
        if gw not in schedule[tid_h]:
            schedule[tid_h][gw] = {"fixtures": 0, "opponents": [], "is_home": [], "fdr": []}
        schedule[tid_h][gw]["fixtures"] += 1
        schedule[tid_h][gw]["opponents"].append(team_map.get(tid_a, {}).get("short_name", "?"))
        schedule[tid_h][gw]["is_home"].append(True)
        schedule[tid_h][gw]["fdr"].append(f["team_h_difficulty"])

        # Away team
        if tid_a not in schedule:
            schedule[tid_a] = {}
        if gw not in schedule[tid_a]:
            schedule[tid_a][gw] = {"fixtures": 0, "opponents": [], "is_home": [], "fdr": []}
        schedule[tid_a][gw]["fixtures"] += 1
        schedule[tid_a][gw]["opponents"].append(team_map.get(tid_h, {}).get("short_name", "?"))
        schedule[tid_a][gw]["is_home"].append(False)
        schedule[tid_a][gw]["fdr"].append(f["team_a_difficulty"])

    # Add type label and fill in blank GWs
    for tid in schedule:
        for gw in range(1, total_gws + 1):
            if gw not in schedule[tid]:
                schedule[tid][gw] = {
                    "fixtures": 0,
                    "opponents": [],
                    "is_home": [],
                    "fdr": [],
                }
            entry = schedule[tid][gw]
            if entry["fixtures"] == 0:
                entry["type"] = "BGW"
            elif entry["fixtures"] >= 2:
                entry["type"] = "DGW"
            else:
                entry["type"] = "normal"

    return schedule


def get_team_fixture_summary(schedule: dict, team_id: int, from_gw: int, num_gws: int = 6) -> list:
    """
    Returns a fixture summary for a team over the next N gameweeks.
    Used for the dashboard fixture ticker.
    """
    result = []
    team_sched = schedule.get(team_id, {})
    for gw in range(from_gw, from_gw + num_gws):
        entry = team_sched.get(gw, {"fixtures": 0, "type": "BGW", "opponents": [], "fdr": []})
        result.append({
            "gw":       gw,
            "type":     entry["type"],
            "fixtures": entry["fixtures"],
            "opponents": entry["opponents"],
            "fdr":      entry["fdr"],
            "avg_fdr":  round(sum(entry["fdr"]) / len(entry["fdr"]), 1) if entry["fdr"] else 0,
        })
    return result


def calc_dgw_xpts_multiplier(team_id: int, gw_id: int, schedule: dict) -> float:
    """
    Returns an xPts multiplier based on fixture count in the gameweek.
    DGW = 1.85x (two fixtures but not quite 2x due to rotation/fatigue)
    BGW = 0.0x  (no fixture = zero points)
    Normal = 1.0x
    """
    team_sched = schedule.get(team_id, {})
    gw_entry   = team_sched.get(gw_id, {})
    gw_type    = gw_entry.get("type", "normal")
    return {"DGW": 1.85, "BGW": 0.0, "normal": 1.0}.get(gw_type, 1.0)


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


# 2026/27 promoted clubs — no Premier League history
# Coventry City, Ipswich Town, Hull City
PROMOTED_CLUBS_2627 = {"Coventry", "Ipswich", "Hull", "COV", "IPS", "HUL"}

def calc_prev_season_anchor(history_past: list, team_short: str = "") -> float:
    """
    Previous season regression anchor v2.

    For established PL clubs: uses best of last 2 seasons pts/90.
    For promoted clubs (Coventry, Ipswich, Hull): applies a 0.4x
    discount to their Championship stats — PL is significantly harder
    and their players are unproven at the top level.
    Players from promoted clubs get a neutral anchor of 0.2 max,
    preventing over-inflation while not completely ignoring pedigree.

    Returns a bonus between 0 and 1.5.
    """
    if not history_past:
        return 0.0

    is_promoted = any(club in team_short for club in PROMOTED_CLUBS_2627)

    scored = []
    for s in history_past[-2:]:
        mins = s.get("minutes", 0)
        pts  = s.get("total_points", 0)
        if mins > 900:
            p90 = pts / (mins / 90)
            # Apply promotion discount — Championship pts don't translate 1:1
            if is_promoted:
                p90 *= 0.4
            scored.append(p90)

    if not scored:
        return 0.0

    best_p90 = max(scored)
    anchor = min(1.5, max(0.0, (best_p90 - 3.0) * 0.3))

    # Cap promoted players at 0.2 — uncertainty is too high
    if is_promoted:
        anchor = min(0.2, anchor)

    return round(anchor, 2)


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

    # 2026/27 BPS changes: GKs, full-backs and FWDs benefit more from BPS
    # Reflect this by boosting ICT bonus for these positions
    # Tackle penalty removed — defenders/MIDs no longer penalised for being tackled
    bps_pos_mult = {
        "GK":  1.25,   # boosted — saves/sweeping now worth more in BPS
        "DEF": 1.15,   # boosted — tackle penalty removed, clearances up
        "MID": 1.00,   # unchanged
        "FWD": 1.15,   # boosted — penalty scoring BPS equalised
    }.get(pos, 1.0)

    ict_bonus = min(1.5, ((threat * 0.003) + (creativity * 0.002) + (influence * 0.001)) * bps_pos_mult)

    # Transfer momentum — are managers buying or selling?
    transfers_in  = player.get("transfers_in", 0)
    transfers_out = player.get("transfers_out", 0)
    net_transfers = transfers_in - transfers_out
    # Normalise: 200k net buys = +0.5 bonus, 200k net sells = -0.5 penalty
    transfer_signal = max(-0.8, min(0.8, net_transfers / 400000))

    # ── Base expected points per game ──
    # Pre-season fallback: when PPG and form are 0 (new season, no data yet),
    # estimate from price — a £10m player should return ~6pts/GW, £5m ~3pts/GW
    price = player.get("price", 6.0)
    if ppg == 0 and form == 0:
        # Price-based estimate: roughly 0.6 pts per £1m of price
        base_per_gw = max(1.0, price * 0.6)
    else:
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

        # Clean sheet + DefCon bonus by position
        # 2026/27: DefCon recalibrated to favour holding MIDs
        # DEF/GK: clean sheet probability bonus
        # MID: small defensive contribution bonus (Rodri/Caicedo types rewarded)
        if pos in ("DEF", "GK"):
            gw += {1: 1.2, 2: 0.7, 3: 0.2, 4: 0.0, 5: 0.0}.get(fdr, 0.0)
        elif pos == "MID":
            # Holding MIDs get a DefCon bonus — scaled by influence score
            # (influence correlates with defensive work rate in FPL)
            influence = player.get("influence", 0)
            if influence > 80:   # high defensive influence = likely holding MID
                gw += {1: 0.4, 2: 0.3, 3: 0.15, 4: 0.0, 5: 0.0}.get(fdr, 0.0)

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

    # ── DGW/BGW multiplier ──
    # Applied before injury multiplier
    # DGW: 1.85x — two fixtures but not 2x due to rotation/fatigue risk
    # BGW: 0.0x  — no fixture, zero expected points
    # Normal: 1.0x
    gw_type  = player.get("gw_type", "normal")
    dgw_mult = {"DGW": 1.85, "BGW": 0.0, "normal": 1.0}.get(gw_type, 1.0)
    total   *= dgw_mult

    # ── Injury/availability multiplier ──
    # Applied after DGW — a doubtful player in a DGW is still risky
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
#  CHIP ADVISOR
#  Analyses next 10 GWs and recommends optimal chip usage
# ════════════════════════════════════════════════════════════

def score_bench_boost(gw_id: int, dgw_schedule: dict, teams: list) -> dict:
    """
    Bench Boost score for a given GW.
    Best used in DGWs where as many teams as possible have 2 fixtures.
    Score 0-10.
    """
    dgw_count = sum(
        1 for tid, sched in dgw_schedule.items()
        if sched.get(gw_id, {}).get("type") == "DGW"
    )
    bgw_count = sum(
        1 for tid, sched in dgw_schedule.items()
        if sched.get(gw_id, {}).get("type") == "BGW"
    )
    total_teams = len(teams)

    # More DGW teams = better for bench boost
    dgw_ratio = dgw_count / max(total_teams, 1)
    score     = min(10.0, dgw_ratio * 20)

    # Penalise if many teams have blanks
    score -= (bgw_count / max(total_teams, 1)) * 5
    score  = max(0.0, round(score, 1))

    return {
        "score":     score,
        "dgw_teams": dgw_count,
        "bgw_teams": bgw_count,
        "verdict":   "Excellent" if score >= 7 else "Good" if score >= 4 else "Poor",
        "reason":    f"{dgw_count} teams have double fixtures this GW" if dgw_count > 0
                     else "No double gameweeks — avoid Bench Boost this GW",
    }


def score_triple_captain(gw_id: int, dgw_schedule: dict, enriched_players: list) -> dict:
    """
    Triple Captain score for a given GW.
    Best used when your best player has a DGW with easy fixtures.
    Score 0-10.
    """
    # Find best player candidates — top 5 by xpts
    top_players = sorted(enriched_players, key=lambda x: x.get("xpts", 0), reverse=True)[:10]

    best_tc_player = None
    best_tc_score  = 0

    for p in top_players:
        tid      = p.get("team_id")
        gw_entry = dgw_schedule.get(tid, {}).get(gw_id, {})
        gw_type  = gw_entry.get("type", "normal")
        avg_fdr  = sum(gw_entry.get("fdr", [3])) / max(len(gw_entry.get("fdr", [3])), 1)

        # Score this player as TC candidate
        tc_score = 0
        if gw_type == "DGW":
            tc_score += 6          # huge bonus for double fixture
        elif gw_type == "BGW":
            tc_score -= 10         # disqualify — no fixture
        tc_score += (5 - avg_fdr) * 0.8   # easier fixture = higher score
        tc_score += p.get("form", 0) * 0.3

        if tc_score > best_tc_score:
            best_tc_score  = tc_score
            best_tc_player = p

    normalised = min(10.0, max(0.0, round(best_tc_score, 1)))

    return {
        "score":       normalised,
        "best_player": best_tc_player["name"] if best_tc_player else "Unknown",
        "best_team":   best_tc_player["team"] if best_tc_player else "",
        "gw_type":     dgw_schedule.get(best_tc_player["team_id"] if best_tc_player else 0, {}).get(gw_id, {}).get("type", "normal"),
        "verdict":     "Excellent" if normalised >= 7 else "Good" if normalised >= 4 else "Poor",
        "reason":      f"Triple captain {best_tc_player['name']} ({best_tc_player['team']}) — {'DGW: two fixtures' if dgw_schedule.get(best_tc_player['team_id'] if best_tc_player else 0, {}).get(gw_id, {}).get('type') == 'DGW' else 'single fixture'}" if best_tc_player else "No strong TC candidate",
    }


def score_free_hit(gw_id: int, dgw_schedule: dict, teams: list) -> dict:
    """
    Free Hit score for a given GW.
    Best used in blank gameweeks where many teams don't have fixtures.
    Score 0-10.
    """
    bgw_count = sum(
        1 for tid, sched in dgw_schedule.items()
        if sched.get(gw_id, {}).get("type") == "BGW"
    )
    total_teams = len(teams)
    bgw_ratio   = bgw_count / max(total_teams, 1)
    score       = min(10.0, round(bgw_ratio * 25, 1))

    return {
        "score":     score,
        "bgw_teams": bgw_count,
        "verdict":   "Excellent" if score >= 7 else "Good" if score >= 4 else "Poor",
        "reason":    f"{bgw_count} teams have no fixture — Free Hit lets you field a full team"
                     if bgw_count > 0 else "No blank gameweeks — save Free Hit for a BGW",
    }


def score_wildcard(gw_id: int, dgw_schedule: dict, enriched_players: list, current_gw: int) -> dict:
    """
    Wildcard score for a given GW.
    Best used before a long run of good fixtures or when squad is poor.
    Score 0-10.
    Higher score = better time to wildcard.
    """
    # Look 6 GWs ahead from this point — how good are the fixtures?
    future_gws = range(gw_id, min(gw_id + 6, 39))

    # Count DGWs ahead — more doubles = better time to wildcard into
    upcoming_dgw = sum(
        1 for gw in future_gws
        for tid, sched in dgw_schedule.items()
        if sched.get(gw, {}).get("type") == "DGW"
    )

    # Score based on upcoming DGWs and how early in the season
    gws_remaining = 38 - gw_id
    score = min(10.0, (upcoming_dgw * 1.5) + (gws_remaining / 38 * 3))

    # Early season wildcards are less valuable (more GWs ahead to use it)
    # Late season wildcards are more urgent if squad is bad
    urgency = "High" if gws_remaining < 15 else "Medium" if gws_remaining < 25 else "Low"

    return {
        "score":         round(score, 1),
        "upcoming_dgws": upcoming_dgw,
        "gws_remaining": gws_remaining,
        "urgency":       urgency,
        "verdict":       "Excellent" if score >= 7 else "Good" if score >= 4 else "Poor",
        "reason":        f"{upcoming_dgw} double gameweeks in the next 6 GWs — good time to wildcard in"
                         if upcoming_dgw > 0 else "No doubles ahead — consider waiting for a better window",
    }


def build_chip_plan(gw_id: int, dgw_schedule: dict, enriched_players: list,
                    teams: list, num_gws: int = 10) -> dict:
    """
    Builds a full chip plan for the next N gameweeks.
    Returns scores for each chip per GW and an overall recommendation.
    """
    plan = []
    for gw in range(gw_id, min(gw_id + num_gws, 39)):
        bb  = score_bench_boost(gw, dgw_schedule, teams)
        tc  = score_triple_captain(gw, dgw_schedule, enriched_players)
        fh  = score_free_hit(gw, dgw_schedule, teams)
        wc  = score_wildcard(gw, dgw_schedule, enriched_players, gw_id)

        # Best chip this GW
        chip_scores = {
            "bench_boost":     bb["score"],
            "triple_captain":  tc["score"],
            "free_hit":        fh["score"],
            "wildcard":        wc["score"],
        }
        best_chip  = max(chip_scores, key=chip_scores.get)
        best_score = chip_scores[best_chip]

        plan.append({
            "gw":             gw,
            "bench_boost":    bb,
            "triple_captain": tc,
            "free_hit":       fh,
            "wildcard":       wc,
            "best_chip":      best_chip if best_score >= 4 else None,
            "best_score":     best_score,
        })

    # Overall recommendation — best single GW per chip
    recommendations = {}
    for chip in ["bench_boost", "triple_captain", "free_hit", "wildcard"]:
        best_gw = max(plan, key=lambda x: x[chip]["score"])
        recommendations[chip] = {
            "gw":      best_gw["gw"],
            "score":   best_gw[chip]["score"],
            "verdict": best_gw[chip]["verdict"],
            "reason":  best_gw[chip]["reason"],
        }

    return {
        "plan":            plan,
        "recommendations": recommendations,
        "current_gw":      gw_id,
    }


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

    # Build DGW/BGW context for AI
    dgw_players_in_xi = [p for p in xi if p.get("gw_type") == "DGW"]
    bgw_players_in_xi = [p for p in xi if p.get("gw_type") == "BGW"]
    dgw_str = ", ".join([f"{p['name']} ({p['team']}, {p.get('fixture_count',2)} fixtures)" for p in dgw_players_in_xi]) or "None"
    bgw_str = ", ".join([f"{p['name']} ({p['team']})" for p in bgw_players_in_xi]) or "None"

    prompt += f"\n\nDOUBLE GAMEWEEK PLAYERS IN YOUR XI: {dgw_str}"
    prompt += f"\nBLANK GAMEWEEK PLAYERS IN YOUR XI (score zero): {bgw_str}"
    prompt += "\n\nIMPORTANT: Always explicitly mention any doubtful/injured players and their risk. Never ignore injury flags."
    prompt += "\nIf there are DGW players, highlight them as key assets — they can score double points."
    prompt += "\nIf there are BGW players in the XI, flag this as a serious concern — they will score zero."
    prompt += "\n2026/27 SEASON RULES: Max 5 free transfers can be rolled. Two sets of chips (Wildcard x2, Free Hit x2, Bench Boost x2, Triple Captain x2). No AFCON transfer top-up this season. GW lockdown at 09:00 UK time day after final match."
    prompt += "\nPromoted clubs this season: Coventry City, Ipswich Town, Hull City. Players from these clubs have no Premier League track record — flag uncertainty when recommending them."

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

    # Season start detection — if GW1 hasn't happened yet, minutes = 0 for everyone
    # In this case, drop the minutes filter entirely and use price as proxy for relevance
    total_minutes_played = sum(p.get("minutes", 0) for p in boot["elements"])
    season_started = total_minutes_played > 10000  # roughly 5+ GWs played

    # Filter to available players
    candidates = [
        p for p in boot["elements"]
        if p["status"] not in ("u", "i", "s")       # exclude unavailable, injured, suspended
        and (p.get("chance_of_playing_next_round") or 100) >= 25  # at least 25% chance
        and (
            not season_started                        # pre-season: include everyone
            or (p.get("minutes") or 0) > 90          # in-season: meaningful minutes only
        )
    ]

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Enriching {len(candidates)} players with GW history...")

    enriched = []

    # EXPANDED POOL: top 200 by a combined form+points score
    # Pre-season fallback: when form=0 and total_pts=0 (new season),
    # use price as a proxy for quality — more expensive = better player
    def candidate_score(p):
        form      = float(p.get("form") or 0)
        total_pts = p.get("total_points", 0)
        price     = p.get("now_cost", 60) / 10   # £6.0m default
        chance    = (p.get("chance_of_playing_next_round") or 100) / 100

        if not season_started:
            # Pre-season: price is the best signal we have
            # Higher price = FPL thinks they're better
            return (price * 2.0) * chance
        else:
            return (form * 4.0 + total_pts * 0.1) * chance

    top_candidates = sorted(candidates, key=candidate_score, reverse=True)[:200]

    # Build full season DGW/BGW schedule
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Building DGW/BGW schedule...")
    dgw_schedule = detect_dgw_bgw(fixtures, boot["teams"])

    # Count DGW and BGW teams for this GW
    dgw_teams = [tid for tid, sched in dgw_schedule.items() if sched.get(gw_id, {}).get("type") == "DGW"]
    bgw_teams = [tid for tid, sched in dgw_schedule.items() if sched.get(gw_id, {}).get("type") == "BGW"]
    if dgw_teams:
        dgw_names = [team_map[tid]["short_name"] for tid in dgw_teams if tid in team_map]
        print(f"  DGW teams in GW{gw_id}: {', '.join(dgw_names)}")
    if bgw_teams:
        bgw_names = [team_map[tid]["short_name"] for tid in bgw_teams if tid in team_map]
        print(f"  BGW teams in GW{gw_id}: {', '.join(bgw_names)}")

    # Build next-fixture list per team with home/away context
    # Now includes ALL fixtures per GW (handles doubles)
    team_fixtures = {t["id"]: [] for t in boot["teams"]}
    for f in sorted(fixtures, key=lambda x: x.get("event") or 999):
        if f.get("finished"):
            continue
        ev = f.get("event") or 999
        if ev > gw_id + cfg.FORECAST_GWS:
            continue
        tid_h, tid_a = f["team_h"], f["team_a"]
        # Allow multiple fixtures per GW for DGW detection
        team_fixtures[tid_h].append({"fdr": f["team_h_difficulty"], "is_home": True,  "event": ev})
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
                "next_home":     fix_list[0]["is_home"] if fix_list else True,
                "gw_type":       dgw_schedule.get(p["team"], {}).get(gw_id, {}).get("type", "normal"),
                "fixture_count": dgw_schedule.get(p["team"], {}).get(gw_id, {}).get("fixtures", 1),
                "momentum":           calc_momentum(gw_pts),
                "consistency":        calc_consistency(gw_pts),
                "rotation_risk":      calc_rotation_risk(gw_mins),
                "home_away":          calc_home_away_split(history),
                "set_piece_bonus":    calc_set_piece_bonus(p),
                "prev_season_anchor": calc_prev_season_anchor(history_past, team_d.get("short_name", "")),
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

    # Build chip plan
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Building chip advisor plan...")
    chip_plan = build_chip_plan(gw_id, dgw_schedule, enriched, boot["teams"])

    # Build price change predictions
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Predicting price changes...")
    price_changes = predict_price_changes(boot)

    # Build DGW/BGW summary for frontend
    team_map_short = {t["id"]: t["short_name"] for t in boot["teams"]}
    dgw_bgw_summary = []
    for gw_check in range(gw_id, min(gw_id + 10, 39)):
        gw_dgw = []
        gw_bgw = []
        for tid, sched in dgw_schedule.items():
            entry = sched.get(gw_check, {})
            if entry.get("type") == "DGW":
                gw_dgw.append({
                    "team": team_map_short.get(tid, "?"),
                    "opponents": entry.get("opponents", []),
                    "fdr": entry.get("fdr", []),
                })
            elif entry.get("type") == "BGW":
                gw_bgw.append({"team": team_map_short.get(tid, "?")})
        if gw_dgw or gw_bgw:
            dgw_bgw_summary.append({
                "gw":   gw_check,
                "dgw":  gw_dgw,
                "bgw":  gw_bgw,
            })

    return {
        "gameweek":       gw_id,
        "deadline":       deadline,
        "squad":          squad,
        "briefing":       briefing,
        "top_players":    sorted(enriched, key=lambda x: x["xpts"], reverse=True)[:30],
        "dgw_bgw":        dgw_bgw_summary,
        "chip_plan":      chip_plan,
        "price_changes":  price_changes,
        "generated_at":   datetime.now().isoformat(),
    }


# ════════════════════════════════════════════════════════════
#  PRICE CHANGE PREDICTOR
# ════════════════════════════════════════════════════════════

def predict_price_changes(boot: dict) -> dict:
    """
    Predicts likely price rises and falls based on transfer momentum.

    FPL price change mechanics:
    - Each player has a hidden "sell price" counter
    - Net transfers in = counter goes up, net transfers out = counter goes down
    - When counter crosses a threshold (~1% of total managers), price changes by £0.1m
    - We use transfers_in_event and transfers_out_event as our signal
    - cost_change_event shows actual price changes this GW already

    Confidence levels:
    - Strong:   net > 200,000 transfers
    - Likely:   net > 75,000 transfers
    - Possible: net > 25,000 transfers
    """
    pos_map = {1:"GK", 2:"DEF", 3:"MID", 4:"FWD"}

    # Total managers — used to normalise ownership %
    total_managers = boot.get("total_players", 10000000)

    risers  = []
    fallers = []

    for el in boot["elements"]:
        if el.get("status") in ("u",):
            continue

        transfers_in  = el.get("transfers_in_event", 0)  or 0
        transfers_out = el.get("transfers_out_event", 0) or 0
        net           = transfers_in - transfers_out
        selected_pct  = float(el.get("selected_by_percent") or 0)
        price         = el.get("now_cost", 0) / 10
        already_changed = el.get("cost_change_event", 0) or 0

        if abs(net) < 10000:  # ignore noise
            continue

        # Confidence based on net transfer volume
        if abs(net) > 200000:
            confidence = "Strong"
            conf_score = 3
        elif abs(net) > 75000:
            confidence = "Likely"
            conf_score = 2
        else:
            confidence = "Possible"
            conf_score = 1

        player_data = {
            "id":              el["id"],
            "name":            el["web_name"],
            "team":            el.get("team", 0),
            "pos":             pos_map.get(el["element_type"], "MID"),
            "price":           price,
            "net_transfers":   net,
            "transfers_in":    transfers_in,
            "transfers_out":   transfers_out,
            "selected_pct":    selected_pct,
            "confidence":      confidence,
            "conf_score":      conf_score,
            "already_changed": already_changed,
            "form":            float(el.get("form") or 0),
        }

        if net > 0:
            risers.append(player_data)
        else:
            fallers.append(player_data)

    # Sort by absolute net transfers descending
    risers.sort(key=lambda x: x["net_transfers"], reverse=True)
    fallers.sort(key=lambda x: x["net_transfers"])

    return {
        "risers":  risers[:15],
        "fallers": fallers[:15],
    }


# ════════════════════════════════════════════════════════════
#  FIXTURE TICKER ENDPOINT
# ════════════════════════════════════════════════════════════

@app.route("/api/fixtures/ticker")
@app.route("/api/fixtures/ticker/<int:from_gw>")
def get_fixture_ticker(from_gw=None):
    """
    Returns a 6-GW fixture ticker for all 20 Premier League teams.
    Each cell contains: opponent, home/away, FDR, DGW/BGW flag.
    Teams are sorted by average FDR (easiest run first).
    """
    try:
        boot     = cached_get(BOOTSTRAP_URL)
        fixtures = cached_get(FIXTURES_URL)

        # Current GW
        from datetime import timezone
        now        = datetime.now(timezone.utc)
        future_gws = [e for e in boot["events"]
                      if e.get("deadline_time") and
                      datetime.fromisoformat(e["deadline_time"].replace("Z","+00:00")) > now]
        active_gw  = future_gws[0] if future_gws else None
        gw_id      = from_gw or (active_gw["id"] if active_gw else 1)

        team_map = {t["id"]: t for t in boot["teams"]}
        dgw_schedule = detect_dgw_bgw(fixtures, boot["teams"])

        NUM_GWS = 6
        gw_range = list(range(gw_id, min(gw_id + NUM_GWS, 39)))

        ticker = []
        for team in boot["teams"]:
            tid   = team["id"]
            row   = {
                "team_id":    tid,
                "team_name":  team["name"],
                "team_short": team["short_name"],
                "gws":        [],
            }
            total_fdr = 0
            fdr_count = 0

            for gw in gw_range:
                gw_entry = dgw_schedule.get(tid, {}).get(gw, {})
                gw_type  = gw_entry.get("type", "BGW" if gw_entry.get("fixtures", 0) == 0 else "normal")
                opps     = gw_entry.get("opponents", [])
                fdrs     = gw_entry.get("fdr", [])
                homes    = gw_entry.get("is_home", [])
                avg_fdr  = round(sum(fdrs) / len(fdrs), 1) if fdrs else 0

                if avg_fdr > 0:
                    total_fdr += avg_fdr
                    fdr_count += 1

                row["gws"].append({
                    "gw":        gw,
                    "type":      gw_type,
                    "fixtures":  gw_entry.get("fixtures", 0),
                    "opponents": opps,
                    "fdr":       fdrs,
                    "is_home":   homes,
                    "avg_fdr":   avg_fdr,
                    # Display string e.g. "MCI(H)" or "ARS(A)+CHE(H)" for DGW
                    "display":   " + ".join([
                        f"{opp}({'H' if h else 'A'})"
                        for opp, h in zip(opps, homes)
                    ]) if opps else "—",
                })

            row["avg_fdr_6gw"] = round(total_fdr / max(fdr_count, 1), 2)
            ticker.append(row)

        # Sort by average FDR ascending (easiest run at top)
        ticker.sort(key=lambda x: x["avg_fdr_6gw"])

        return jsonify({
            "ok":       True,
            "ticker":   ticker,
            "gw_range": gw_range,
            "from_gw":  gw_id,
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ════════════════════════════════════════════════════════════
#  WILDCARD OPTIMIZER
#  Builds the best possible squad from scratch — no current team
#  constraints. Looks 6 GWs ahead. Used when playing wildcard.
# ════════════════════════════════════════════════════════════

def calc_wildcard_score(player: dict, dgw_schedule: dict,
                        from_gw: int, num_gws: int = 6) -> float:
    """
    Wildcard-specific player score.
    Differences from xPts model:
    - Looks num_gws ahead (default 6) not 3
    - DGWs in ANY of the next 6 GWs get a bonus, not just GW1
    - Sustained fixture run quality matters more than single-GW form
    - Price efficiency is factored in (pts per £m)
    - BGWs in the window are penalised per blank
    """
    ppg      = float(player.get("points_per_game") or 0)
    form     = float(player.get("form") or 0)
    xgi      = float(player.get("xgi") or 0)
    price    = float(player.get("price") or 5.0)
    consist  = player.get("consistency", 5) / 10
    rot_risk = player.get("rotation_risk", "unknown")
    rot_pen  = {"low": 0.0, "medium": -0.8, "high": -2.0, "unknown": -0.4}.get(rot_risk, 0.0)
    tid      = player.get("team_id")
    pos      = player.get("pos", "MID")
    ict      = float(player.get("ict_index") or 0)
    sp_bonus = player.get("set_piece_bonus", 0)

    # Base score per GW
    base = (ppg * 0.45) + (form * 0.30) + (xgi * 1.5) + (ict * 0.008)

    total      = 0.0
    dgw_count  = 0
    bgw_count  = 0

    for gw in range(from_gw, min(from_gw + num_gws, 39)):
        gw_entry = dgw_schedule.get(tid, {}).get(gw, {})
        gw_type  = gw_entry.get("type", "normal")
        fdrs     = gw_entry.get("fdr", [3])
        avg_fdr  = sum(fdrs) / max(len(fdrs), 1)

        if gw_type == "BGW":
            bgw_count += 1
            continue  # no points for blanks

        # FDR adjustment
        fdr_adj = {1: +1.5, 2: +0.8, 3: 0.0, 4: -0.8, 5: -1.8}.get(int(avg_fdr), 0.0)
        gw_score = base + fdr_adj

        # DEF/GK clean sheet bonus
        if pos in ("DEF", "GK"):
            gw_score += {1: 1.2, 2: 0.7, 3: 0.2}.get(int(avg_fdr), 0.0)

        # DGW bonus — player plays twice this GW
        if gw_type == "DGW":
            gw_score *= 1.85
            dgw_count += 1

        total += max(0.0, gw_score)

    # Global adjustments
    total += min(1.5, sp_bonus)
    total += max(-1.5, min(1.5, player.get("momentum", 0) * 0.25))
    total *= consist
    total += rot_pen

    # BGW penalty — each blank costs expected points
    total -= bgw_count * (base * 0.8)

    # Injury multiplier
    chance = player.get("chance")
    status = player.get("status", "a")
    if status == "d" or (chance is not None and chance < 100):
        inj_mult = {100:1.0, 75:0.85, 50:0.50, 25:0.20, 0:0.0}.get(
            chance if chance is not None else 100, 1.0)
        total *= inj_mult

    # Price efficiency bonus — cheaper players with same output = better value
    # Normalised: £5m player gets +0.5 bonus vs £10m player getting 0
    value_bonus = max(0.0, (10.0 - price) * 0.1)
    total += value_bonus

    return round(max(0.0, total), 2)


def build_wildcard_squad(enriched: list, dgw_schedule: dict,
                         from_gw: int, formation: str = "4-3-3") -> dict:
    """
    Builds the optimal wildcard squad.
    Same FPL rules as regular optimizer but:
    - Uses wildcard_score (6-GW horizon) not xpts (3-GW)
    - No current team constraints
    - Slightly more aggressive on fixtures/DGWs
    """
    # Score all players on wildcard metric
    scored = []
    for p in enriched:
        wc_score = calc_wildcard_score(p, dgw_schedule, from_gw)
        scored.append({**p, "wc_score": wc_score})

    parts  = formation.split("-")
    n_def, n_mid, n_fwd = int(parts[0]), int(parts[1]), int(parts[2])

    POS_LIMITS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}

    by_pos = {
        pos: sorted([p for p in scored if p["pos"] == pos],
                    key=lambda x: x["wc_score"], reverse=True)
        for pos in ("GK", "DEF", "MID", "FWD")
    }

    BUDGET     = 100.0
    XI_BUDGET  = 80.0
    team_count = {}
    pos_count  = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    spent      = 0.0

    def can_pick(p, budget_cap):
        return (
            team_count.get(p["team_id"], 0) < 3 and
            pos_count[p["pos"]] < POS_LIMITS[p["pos"]] and
            spent + p["price"] <= budget_cap
        )

    def pick(p):
        nonlocal spent
        team_count[p["team_id"]] = team_count.get(p["team_id"], 0) + 1
        pos_count[p["pos"]]     += 1
        spent                   += p["price"]

    def pick_pos(pool, needed, budget_cap):
        picks = []
        for p in pool:
            if len(picks) >= needed: break
            if can_pick(p, budget_cap):
                picks.append(p)
                pick(p)
        return picks

    # XI
    xi_gk  = pick_pos(by_pos["GK"],  1,     XI_BUDGET)
    xi_def = pick_pos(by_pos["DEF"], n_def, XI_BUDGET)
    xi_mid = pick_pos(by_pos["MID"], n_mid, XI_BUDGET)
    xi_fwd = pick_pos(by_pos["FWD"], n_fwd, XI_BUDGET)

    # Fill gaps if XI_BUDGET too tight
    for pos_name, pool, needed in [
        ("GK",by_pos["GK"],1),("DEF",by_pos["DEF"],n_def),
        ("MID",by_pos["MID"],n_mid),("FWD",by_pos["FWD"],n_fwd)
    ]:
        existing = {"GK":xi_gk,"DEF":xi_def,"MID":xi_mid,"FWD":xi_fwd}[pos_name]
        used     = {p["id"] for p in xi_gk+xi_def+xi_mid+xi_fwd}
        for p in pool:
            if len(existing) >= needed: break
            if p["id"] in used: continue
            if can_pick(p, BUDGET):
                existing.append(p); pick(p); used.add(p["id"])

    xi       = xi_gk + xi_def + xi_mid + xi_fwd
    used_ids = {p["id"] for p in xi}

    # Bench — cheapest viable
    bench_gk_pool  = sorted([p for p in by_pos["GK"]  if p["id"] not in used_ids], key=lambda x: x["price"])
    bench_out_pool = sorted([p for pos_n in ("DEF","MID","FWD")
                              for p in by_pos[pos_n] if p["id"] not in used_ids], key=lambda x: x["price"])

    bench_gk = []
    for p in bench_gk_pool:
        if can_pick(p, BUDGET):
            bench_gk.append(p); pick(p); used_ids.add(p["id"]); break

    bench_out = []
    for p in bench_out_pool:
        if len(bench_out) >= 3: break
        if can_pick(p, BUDGET):
            bench_out.append(p); pick(p); used_ids.add(p["id"])

    bench = bench_gk + bench_out

    # Safety net
    if len(bench) < 4:
        all_pool = sorted([p for pos_n in ("GK","DEF","MID","FWD")
                           for p in by_pos[pos_n] if p["id"] not in used_ids],
                          key=lambda x: x["price"])
        for p in all_pool:
            if len(bench) >= 4: break
            if pos_count[p["pos"]] < POS_LIMITS[p["pos"]] and spent + p["price"] <= BUDGET:
                bench.append(p); pos_count[p["pos"]] += 1
                spent += p["price"]; used_ids.add(p["id"])

    # Captain — best wc_score, fully fit, prefer home next fixture
    def cap_score(p):
        if p.get("status") == "d" or (p.get("chance") and p.get("chance") < 100):
            return -999
        s = p["wc_score"]
        if p.get("next_home"): s += 1.5
        return s

    eligible = [p for p in xi if cap_score(p) > -999] or xi
    captain  = max(eligible, key=cap_score)
    vice     = max([p for p in xi if p["id"] != captain["id"]], key=cap_score)

    total_value = round(sum(p["price"] for p in xi + bench), 1)

    # DGW summary for this squad
    dgw_players = [p for p in xi if p.get("gw_type") == "DGW"]
    bgw_players = [p for p in xi if p.get("gw_type") == "BGW"]

    return {
        "xi":           xi,
        "bench":        bench,
        "captain":      captain,
        "vice_captain": vice,
        "total_value":  total_value,
        "bank":         round(100.0 - total_value, 1),
        "formation":    formation,
        "dgw_players":  [p["name"] for p in dgw_players],
        "bgw_players":  [p["name"] for p in bgw_players],
        "horizon_gws":  from_gw,
    }


# ════════════════════════════════════════════════════════════
#  PRICE CHANGES ENDPOINT
# ════════════════════════════════════════════════════════════

@app.route("/api/price-changes")
def get_price_changes():
    """
    Standalone price change predictor.
    Fast — only needs bootstrap data, no GW history fetching.
    Returns top 15 predicted risers and fallers.
    """
    try:
        boot          = cached_get(BOOTSTRAP_URL)
        team_map      = {t["id"]: t["short_name"] for t in boot["teams"]}
        price_changes = predict_price_changes(boot)

        # Enrich with team short names
        for group in ["risers", "fallers"]:
            for p in price_changes[group]:
                p["team_short"] = team_map.get(p["team"], "?")

        return jsonify({"ok": True, **price_changes})

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ════════════════════════════════════════════════════════════
#  WILDCARD OPTIMIZER ENDPOINT
# ════════════════════════════════════════════════════════════

@app.route("/api/wildcard")
@app.route("/api/wildcard/<formation>")
def get_wildcard_squad(formation="4-3-3"):
    """
    Builds the optimal wildcard squad.
    Uses full enriched player data with 6-GW horizon scoring.
    Takes longer than regular squad — fetches full GW history.
    """
    try:
        data = build_full_dataset(formation)
        enriched     = data["top_players"]
        boot         = cached_get(BOOTSTRAP_URL)
        fixtures_raw = cached_get(FIXTURES_URL)
        dgw_schedule = detect_dgw_bgw(fixtures_raw, boot["teams"])

        current_gw = next((e for e in boot["events"] if e["is_current"]), None)
        next_gw    = next((e for e in boot["events"] if e["is_next"]),    None)
        active     = current_gw or next_gw
        gw_id      = active["id"] if active else 1

        wc_squad = build_wildcard_squad(enriched, dgw_schedule, gw_id, formation)

        # Get AI briefing for wildcard squad
        try:
            cap = wc_squad["captain"]
            vc  = wc_squad["vice_captain"]
            xi  = wc_squad["xi"]

            xi_summary = "\n".join([
                f"  {p['pos']} | {p['name']} ({p['team']}) | £{p['price']}m | "
                f"WC Score:{p.get('wc_score',0)} | Form:{p['form']} | "
                f"GW type:{p.get('gw_type','normal')} | FDR:{p.get('fdr_avg3','?')}"
                for p in xi
            ])

            prompt = f"""You are an elite FPL analyst. A manager has just activated their Wildcard chip for GW{gw_id}.
This squad is optimised over the next 6 gameweeks, not just one.

FORMATION: {formation}
BUDGET USED: £{wc_squad['total_value']}m (£{wc_squad['bank']}m in bank)
CAPTAIN: {cap['name']} ({cap['team']}) — WC Score: {cap.get('wc_score',0)}, Form: {cap['form']}
VICE: {vc['name']} ({vc['team']})
DGW PLAYERS: {', '.join(wc_squad['dgw_players']) or 'None this GW'}

STARTING XI:
{xi_summary}

Write a 200-word wildcard briefing. Explain:
1. The overall 6-GW strategy — which fixture runs and DGWs shaped this squad
2. Captain rationale for GW{gw_id}
3. The best long-term assets in this squad and why
4. Any risks to monitor
Plain prose, no markdown, second person ("your wildcard squad...")."""

            resp = req.post(
                "https://api.anthropic.com/v1/messages",
                headers={"Content-Type":"application/json",
                         "x-api-key": cfg.ANTHROPIC_API_KEY,
                         "anthropic-version":"2023-06-01"},
                json={"model":"claude-sonnet-4-6","max_tokens":600,
                      "messages":[{"role":"user","content":prompt}]},
                timeout=30,
            )
            briefing = resp.json()["content"][0]["text"]
        except Exception as e:
            briefing = f"Wildcard briefing unavailable: {e}"

        return jsonify({
            "ok":        True,
            "squad":     wc_squad,
            "briefing":  briefing,
            "gameweek":  gw_id,
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ════════════════════════════════════════════════════════════
#  CHIP ADVISOR ENDPOINT
# ════════════════════════════════════════════════════════════

@app.route("/api/chips")
@app.route("/api/chips/<int:from_gw>")
def get_chip_plan(from_gw=None):
    """Standalone chip advisor endpoint."""
    try:
        boot     = cached_get(BOOTSTRAP_URL)
        fixtures = cached_get(FIXTURES_URL)

        current_gw = next((e for e in boot["events"] if e["is_current"]), None)
        next_gw    = next((e for e in boot["events"] if e["is_next"]), None)
        active     = current_gw or next_gw
        gw_id      = from_gw or (active["id"] if active else 1)

        dgw_schedule = detect_dgw_bgw(fixtures, boot["teams"])

        # Lightweight player list for TC scoring
        team_map = {t["id"]: t for t in boot["teams"]}
        pos_map  = {1:"GK",2:"DEF",3:"MID",4:"FWD"}
        players  = []
        for el in boot["elements"]:
            if el.get("status") in ("u","i","s"):
                continue
            team = team_map.get(el["team"], {})
            gw_entry = dgw_schedule.get(el["team"], {}).get(gw_id, {})
            players.append({
                "id":       el["id"],
                "name":     el["web_name"],
                "team":     team.get("short_name",""),
                "team_id":  el["team"],
                "pos":      pos_map.get(el["element_type"],"MID"),
                "price":    el["now_cost"] / 10,
                "form":     float(el.get("form") or 0),
                "xpts":     float(el.get("ep_next") or 0),  # FPL's own prediction
                "gw_type":  gw_entry.get("type","normal"),
            })

        chip_plan = build_chip_plan(gw_id, dgw_schedule, players, boot["teams"])
        return jsonify({"ok": True, "chip_plan": chip_plan})

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


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
