# scripts/case_frames.py
"""
Case Frame Registry for predicate-argument structure probing.

Deep Case Role Taxonomy:
  Agent     (WHO)   -- participant/initiator of the action
  Theme     (WHAT)  -- entity acted upon / sought
  Location  (WHERE) -- unified place role
  Source    (WHERE) -- path start; only used when split_path_cases=True
  Goal      (WHERE) -- path end;   only used when split_path_cases=True
  Time      (WHEN)  -- temporal anchor (date, time, duration …)
  Manner    (HOW)   -- description, quantity, price, instrument …

Each CaseFrame defines:
  predicate        : str   -- canonical verb ("find", "book", "transfer" …)
  theme_domain     : str   -- coarse domain label ("Hotel", "Flight", …)
  obligatory_cases : list  -- roles that *must* be filled for the frame to be satisfied
  optional_cases   : list  -- roles that enrich, but are not required
  split_path_cases : bool  -- if True, WHERE is split into Source + Goal
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Canonical Case Role Names
# ---------------------------------------------------------------------------
ROLE_AGENT    = "Agent"
ROLE_THEME    = "Theme"
ROLE_LOCATION = "Location"
ROLE_SOURCE   = "Source"      # only when split_path_cases=True
ROLE_GOAL     = "Goal"        # only when split_path_cases=True
ROLE_TIME     = "Time"
ROLE_MANNER   = "Manner"

# All base roles (used for label indexing etc.)
ALL_ROLES = [ROLE_AGENT, ROLE_THEME, ROLE_LOCATION, ROLE_SOURCE,
             ROLE_GOAL, ROLE_TIME, ROLE_MANNER]


# ---------------------------------------------------------------------------
# Slot-to-CaseRole mapping helpers
# ---------------------------------------------------------------------------
# Maps individual SGD / MultiWOZ slot names -> canonical case role.
# This is a best-effort heuristic; domain-specific overrides take precedence.
SLOT_TO_ROLE: dict[str, str] = {
    # --- Agent / Who ---
    "traveler_first_name": ROLE_AGENT,
    "traveler_last_name":  ROLE_AGENT,
    "passengers":          ROLE_AGENT,
    "group_size":          ROLE_AGENT,
    "number_of_adults":    ROLE_AGENT,
    "number_of_children":  ROLE_AGENT,
    "account_name":        ROLE_AGENT,
    "recipient_account_name": ROLE_AGENT,

    # --- Theme / What ---
    "hotel_name":          ROLE_THEME,
    "restaurant_name":     ROLE_THEME,
    "event_name":          ROLE_THEME,
    "movie_name":          ROLE_THEME,
    "car_type":            ROLE_THEME,
    "food":                ROLE_THEME,
    "category":            ROLE_THEME,

    # --- Source (path-start) ---
    "from_location":       ROLE_SOURCE,
    "origin":              ROLE_SOURCE,
    "departure_city":      ROLE_SOURCE,
    "departure":           ROLE_SOURCE,   # MultiWOZ

    # --- Goal (path-end) ---
    "to_location":         ROLE_GOAL,
    "destination":         ROLE_GOAL,
    "to_city":             ROLE_GOAL,

    # --- Location (non-directional) ---
    "city":                ROLE_LOCATION,
    "area":                ROLE_LOCATION,
    "location":            ROLE_LOCATION,
    "address":             ROLE_LOCATION,
    "airport":             ROLE_LOCATION,
    "pickup_location":     ROLE_LOCATION,

    # --- Time ---
    "date":                ROLE_TIME,
    "time":                ROLE_TIME,
    "leaving_date":        ROLE_TIME,
    "leaving_time":        ROLE_TIME,
    "departure_date":      ROLE_TIME,
    "departure_time":      ROLE_TIME,
    "arrival_date":        ROLE_TIME,
    "arrival_time":        ROLE_TIME,
    "return_date":         ROLE_TIME,
    "return_time":         ROLE_TIME,
    "booking_date":        ROLE_TIME,
    "checkin_date":        ROLE_TIME,
    "checkout_date":       ROLE_TIME,
    "start_date":          ROLE_TIME,
    "end_date":            ROLE_TIME,
    "event_date":          ROLE_TIME,
    "leaveat":             ROLE_TIME,   # MultiWOZ
    "arriveby":            ROLE_TIME,   # MultiWOZ
    "day":                 ROLE_TIME,

    # --- Manner / How ---
    "number_of_seats":     ROLE_MANNER,
    "seats":               ROLE_MANNER,
    "fare_type":           ROLE_MANNER,
    "trip_protection":     ROLE_MANNER,
    "pricerange":          ROLE_MANNER,
    "price":               ROLE_MANNER,
    "stars":               ROLE_MANNER,
    "stay":                ROLE_MANNER,
    "internet":            ROLE_MANNER,
    "parking":             ROLE_MANNER,
    "count":               ROLE_MANNER,
    "total_price":         ROLE_MANNER,
    "account_type":        ROLE_MANNER,
    "amount":              ROLE_MANNER,
}


def slot_to_role(slot_name: str, service: str = "") -> str:
    """Map a slot name to a deep case role.

    1. Exact match in SLOT_TO_ROLE.
    2. Keyword heuristics on the slot name words.
    3. Falls back to ROLE_MANNER (most generic).
    """
    sl = slot_name.lower()
    if sl in SLOT_TO_ROLE:
        return SLOT_TO_ROLE[sl]

    # Keyword heuristics
    if any(k in sl for k in ("origin", "from_", "departure", "source")):
        return ROLE_SOURCE
    if any(k in sl for k in ("destination", "to_", "_to", "goal", "arrival")):
        return ROLE_GOAL
    if any(k in sl for k in ("date", "time", "day", "when", "leave", "arrive", "checkin", "checkout")):
        return ROLE_TIME
    if any(k in sl for k in ("city", "area", "location", "address", "place", "venue")):
        return ROLE_LOCATION
    if any(k in sl for k in ("name", "first_name", "last_name")):
        # Could be Agent or Theme — default to Theme for entities
        if any(k in sl for k in ("traveler", "passenger", "person", "user", "rider")):
            return ROLE_AGENT
        return ROLE_THEME
    if any(k in sl for k in ("passenger", "group", "adult", "child", "traveler", "rider", "guest", "person")):
        return ROLE_AGENT

    return ROLE_MANNER  # fallback


# ---------------------------------------------------------------------------
# CaseFrame dataclass
# ---------------------------------------------------------------------------
@dataclass
class CaseFrame:
    predicate:        str
    theme_domain:     str              # "Hotel", "Flight", "Restaurant", …
    obligatory_cases: List[str]       # roles required for saturation
    optional_cases:   List[str] = field(default_factory=list)
    split_path_cases: bool = False    # True -> Source + Goal; False -> Location

    def effective_obligatory(self) -> List[str]:
        """Return obligatory cases, replacing the FIRST Location with Source+Goal if split.
        Each ROLE_LOCATION occurrence in obligatory_cases represents ONE path-axis slot.
        Use a single ROLE_LOCATION in the definition for transport frames; it expands to
        [Source, Goal] automatically.
        """
        if not self.split_path_cases:
            return self.obligatory_cases
        out = []
        path_expanded = False
        for r in self.obligatory_cases:
            if r == ROLE_LOCATION and not path_expanded:
                out.extend([ROLE_SOURCE, ROLE_GOAL])
                path_expanded = True
            else:
                out.append(r)
        return out

    def saturation(self, filled_roles: set[str]) -> tuple[float, bool]:
        """Compute (saturation_score, is_saturated) given currently-filled roles."""
        oblig = self.effective_obligatory()
        if not oblig:
            return 1.0, True
        n_filled = sum(1 for r in oblig if r in filled_roles)
        score = n_filled / len(oblig)
        return score, (n_filled == len(oblig))


# ---------------------------------------------------------------------------
# Frame Registry
# ---------------------------------------------------------------------------
# Key: (predicate, theme_domain) -- lowercase both
FRAME_REGISTRY: dict[tuple[str, str], CaseFrame] = {}


def _reg(frame: CaseFrame) -> None:
    key = (frame.predicate.lower(), frame.theme_domain.lower())
    FRAME_REGISTRY[key] = frame


# ---- Buses / Trains -------------------------------------------------------
# ROLE_LOCATION (single) expands to [Source, Goal] via effective_obligatory()
_reg(CaseFrame("find",    "Bus",     [ROLE_LOCATION, ROLE_TIME],
               split_path_cases=True,
               optional_cases=[ROLE_AGENT, ROLE_MANNER]))
_reg(CaseFrame("buy",     "Bus",     [ROLE_LOCATION, ROLE_TIME, ROLE_AGENT],
               split_path_cases=True))
_reg(CaseFrame("find",    "Train",   [ROLE_LOCATION, ROLE_TIME],
               split_path_cases=True,
               optional_cases=[ROLE_AGENT, ROLE_MANNER]))
_reg(CaseFrame("book",    "Train",   [ROLE_LOCATION, ROLE_TIME, ROLE_AGENT],
               split_path_cases=True))

# ---- Flights ---------------------------------------------------------------
_reg(CaseFrame("search",  "Flight",  [ROLE_LOCATION, ROLE_TIME],
               split_path_cases=True,
               optional_cases=[ROLE_AGENT, ROLE_MANNER]))
_reg(CaseFrame("reserve", "Flight",  [ROLE_LOCATION, ROLE_TIME, ROLE_AGENT, ROLE_MANNER],
               split_path_cases=True))

# ---- Hotels / Houses -------------------------------------------------------
_reg(CaseFrame("search",  "Hotel",   [ROLE_LOCATION, ROLE_TIME],
               optional_cases=[ROLE_THEME, ROLE_MANNER]))
_reg(CaseFrame("reserve", "Hotel",   [ROLE_THEME, ROLE_TIME, ROLE_AGENT],
               optional_cases=[ROLE_LOCATION, ROLE_MANNER]))
_reg(CaseFrame("find",    "Hotel",   [ROLE_LOCATION],
               optional_cases=[ROLE_MANNER]))
_reg(CaseFrame("book",    "Hotel",   [ROLE_THEME, ROLE_TIME, ROLE_AGENT]))

# ---- Restaurants -----------------------------------------------------------
_reg(CaseFrame("find",    "Restaurant", [ROLE_LOCATION],
               optional_cases=[ROLE_THEME, ROLE_MANNER]))
_reg(CaseFrame("reserve", "Restaurant", [ROLE_THEME, ROLE_TIME, ROLE_AGENT]))

# ---- Rides / Cars ----------------------------------------------------------
_reg(CaseFrame("find",    "RideSharing", [ROLE_LOCATION, ROLE_GOAL],
               optional_cases=[ROLE_AGENT, ROLE_MANNER]))
_reg(CaseFrame("reserve", "RideSharing", [ROLE_LOCATION, ROLE_GOAL, ROLE_AGENT]))
_reg(CaseFrame("find",    "Car",      [ROLE_LOCATION, ROLE_TIME],
               optional_cases=[ROLE_THEME, ROLE_MANNER]))
_reg(CaseFrame("reserve", "Car",      [ROLE_LOCATION, ROLE_TIME, ROLE_AGENT]))

# ---- Events / Movies -------------------------------------------------------
_reg(CaseFrame("find",    "Event",    [ROLE_LOCATION, ROLE_TIME],
               optional_cases=[ROLE_THEME, ROLE_MANNER]))
_reg(CaseFrame("buy",     "Event",    [ROLE_THEME, ROLE_TIME, ROLE_AGENT, ROLE_LOCATION]))
_reg(CaseFrame("find",    "Movie",    [ROLE_LOCATION],
               optional_cases=[ROLE_THEME, ROLE_TIME, ROLE_MANNER]))
_reg(CaseFrame("play",    "Movie",    [ROLE_THEME]))
_reg(CaseFrame("rent",    "Movie",    [ROLE_THEME, ROLE_TIME]))

# ---- Banking ---------------------------------------------------------------
_reg(CaseFrame("check",   "Bank",     [ROLE_MANNER]))   # account_type
_reg(CaseFrame("transfer","Bank",     [ROLE_MANNER, ROLE_MANNER, ROLE_AGENT]))  # amount, acct_type, recipient

# ---- Homes / Apartments ----------------------------------------------------
_reg(CaseFrame("find",    "Home",     [ROLE_LOCATION],
               optional_cases=[ROLE_MANNER]))
_reg(CaseFrame("schedule","Home",     [ROLE_THEME, ROLE_TIME]))

# ---- Travel / Generic MultiWOZ ----------------------------------------------
_reg(CaseFrame("find",    "Attraction", [ROLE_LOCATION],
               optional_cases=[ROLE_THEME]))
_reg(CaseFrame("book",    "Attraction", [ROLE_THEME, ROLE_TIME, ROLE_AGENT]))


# ---------------------------------------------------------------------------
# Intent -> (predicate, theme_domain) mapping for SGD
# ---------------------------------------------------------------------------
INTENT_TO_PREDICATE_DOMAIN: dict[str, tuple[str, str]] = {
    # Buses
    "FindBus":              ("find",    "Bus"),
    "BuyBusTicket":         ("buy",     "Bus"),
    # Flights
    "SearchOnewayFlight":   ("search",  "Flight"),
    "SearchRoundtripFlights":("search", "Flight"),
    "ReserveOnewayFlight":  ("reserve", "Flight"),
    "ReserveRoundtripFlights":("reserve","Flight"),
    # Hotels / Houses
    "SearchHotel":          ("search",  "Hotel"),
    "ReserveHotel":         ("reserve", "Hotel"),
    "SearchHouse":          ("search",  "Hotel"),
    "BookHouse":            ("book",    "Hotel"),
    # Restaurants
    "FindRestaurants":      ("find",    "Restaurant"),
    "ReserveRestaurant":    ("reserve", "Restaurant"),
    # Rides
    "GetRide":              ("find",    "RideSharing"),
    "ReserveRide":          ("reserve", "RideSharing"),
    # Cars
    "GetCarsAvailable":     ("find",    "Car"),
    "ReserveCar":           ("reserve", "Car"),
    # Events / Movies
    "FindEvents":           ("find",    "Event"),
    "BuyEventTickets":      ("buy",     "Event"),
    "GetEventDates":        ("find",    "Event"),
    "FindMovies":           ("find",    "Movie"),
    "PlayMovie":            ("play",    "Movie"),
    "RentMovie":            ("rent",    "Movie"),
    # Banking
    "CheckBalance":         ("check",   "Bank"),
    "TransferMoney":        ("transfer","Bank"),
    # Homes
    "FindApartment":        ("find",    "Home"),
    "FindHomeByArea":       ("find",    "Home"),
    "ScheduleVisit":        ("schedule","Home"),
    # Alarms / Calendar – generic
    "AddAlarm":             ("add",     "Alarm"),
    "GetAlarms":            ("find",    "Alarm"),
    "AddEvent":             ("add",     "Calendar"),
    "GetEvents":            ("find",    "Calendar"),
    "GetAvailableTime":     ("find",    "Calendar"),
}

# MultiWOZ domain -> (predicate, theme_domain)
MULTIWOZ_DOMAIN_TO_PREDICATE_DOMAIN: dict[str, tuple[str, str]] = {
    "hotel":      ("find",    "Hotel"),
    "restaurant": ("find",    "Restaurant"),
    "train":      ("find",    "Train"),
    "taxi":       ("find",    "RideSharing"),
    "attraction": ("find",    "Attraction"),
    "hospital":   ("find",    "Home"),  # rough fallback
    "police":     ("find",    "Home"),
}


def get_frame(predicate: str, theme_domain: str) -> CaseFrame | None:
    """Look up a CaseFrame by predicate and theme_domain (case-insensitive)."""
    return FRAME_REGISTRY.get((predicate.lower(), theme_domain.lower()))


def get_frame_for_intent(intent: str) -> CaseFrame | None:
    """Look up a CaseFrame given an SGD intent string."""
    pd = INTENT_TO_PREDICATE_DOMAIN.get(intent)
    if pd is None:
        return None
    return get_frame(*pd)


def get_frame_for_multiwoz_domain(domain: str) -> CaseFrame | None:
    """Look up a CaseFrame given a MultiWOZ domain string."""
    pd = MULTIWOZ_DOMAIN_TO_PREDICATE_DOMAIN.get(domain.lower())
    if pd is None:
        return None
    return get_frame(*pd)
