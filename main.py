import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import pandas_gbq
import requests
from dotenv import load_dotenv
from meetup.client import Client
from meetup.token_manager import TokenCacheGCS, TokenManager

np.random.seed(42)

logging.basicConfig(
    format="%(message)s",
    stream=sys.stdout,
    level=logging.INFO,
)
_LOGGER = logging.getLogger("main")

load_dotenv()

ADD_TO_GUESTLIST = bool(os.environ.get("MEETUP_ADD_TO_GUESTLIST", False))
CLIENT_ID = os.environ["MEETUP_CLIENT_ID"]
CLIENT_SECRET = os.environ["MEETUP_CLIENT_SECRET"]
EVENT_ID = os.environ["MEETUP_EVENT_ID"]
GROUP_ID = os.environ["MEETUP_GROUP_ID"]
PROJECT_ID = os.environ["MEETUP_PROJECT_ID"]
TICKETS_MAX = int(os.environ["MEETUP_TICKETS_MAX"])
TOKEN_BLOB = os.environ["MEETUP_TOKEN_BLOB"]
TOKEN_BUCKET = os.environ["MEETUP_TOKEN_BUCKET"]

TOKEN_MANAGER = TokenManager(
    CLIENT_ID, CLIENT_SECRET, TokenCacheGCS(TOKEN_BUCKET, TOKEN_BLOB)
)
CLIENT = Client(lambda: TOKEN_MANAGER.token().access_token)


def _rsvps():
    _LOGGER.info("Requesting RSVPs from Meetup API.")
    rsvps = pd.concat(CLIENT.scan(url=f"/{GROUP_ID}/events/{EVENT_ID}/rsvps"))
    rsvps["member_id"] = rsvps["member"].apply(lambda row: row["id"])
    rsvps["member_name"] = rsvps["member"].apply(lambda row: row["name"])
    rsvps["attendees"] = rsvps["guests"] + 1
    rsvps = rsvps[["member_id", "member_name", "response", "attendees"]]
    return rsvps


def _attendances():
    _LOGGER.info("Querying attendances from Google BigQuery.")
    query = """
        WITH attendances_latest AS (
          SELECT
            * EXCEPT(row_number)
          FROM (
            SELECT
              *,
              ROW_NUMBER() OVER (PARTITION BY group_id, event_id, member_id ORDER BY requested_at DESC) row_number
            FROM `meetup_raw.attendances`
          )
          WHERE row_number = 1
        )
        SELECT
          member_id,
          COUNT(*) attendances
        FROM attendances_latest
        WHERE status="attended"
        GROUP BY member_id
    """
    return pandas_gbq.read_gbq(
        query,
        project_id=PROJECT_ID,
    )


def _raffle(rsvps, attendances):
    """
    Merges RSVPs and attendances on member id.
    Adds 1 to all attendances to give members without prior attendance a chance to win a ticket.
    """
    _LOGGER.info("Building raffle table.")
    raffle = rsvps.merge(attendances, how="left", on="member_id")
    raffle["attendances"] = raffle["attendances"].fillna(0).astype(int) + 1
    raffle = raffle[["member_id", "member_name", "response", "attendances"]]
    return raffle


def _winners(raffle, tickets_available):
    """
    Selects winners from all members on the waitlist.
    Gives higher chances to people with more past attendances.
    """
    _LOGGER.info("Selecting winners.")
    winners = raffle[raffle.response == "waitlist"].sample(
        tickets_available, weights="attendances"
    )["member_id"]
    _LOGGER.info(f"Winners are: {winners.tolist()}.")
    return winners


def _add_to_guestlist(member_id):
    """
    Adds a group member to the event guestlist.
    WARNING: This cannot be undone automatically.
    """

    def _headers():
        return {"Authorization": f"Bearer {TOKEN_MANAGER.token().access_token}"}

    def _params(member_id):
        return {
            "member_id": member_id,
            "event_id": EVENT_ID,
            "rsvp": "yes",
        }

    _LOGGER.debug(f"Requesting to move #{member_id} to guestlist.")
    res = requests.post(
        "https://api.meetup.com/2/rsvp", headers=_headers(), params=_params(member_id)
    )
    res.raise_for_status()
    res = res.json()
    _LOGGER.info(
        f"""Moving '{res["member"]["name"]}' #{res["member"]["member_id"]} to guestlist."""
    )


def main():
    """
    1. Fetches RSVPs from Meetup API.
    2. Fetches historical attendances from Google BigQuery.
    3. Determines winners based on RSVPs and attendance.
    4. Adds winners to guestlist.
    """
    rsvps = _rsvps()
    tickets_taken = rsvps[rsvps.response == "yes"]["attendees"].sum()
    _LOGGER.info(f"{tickets_taken} tickets taken.")
    tickets_available = max(0, TICKETS_MAX - tickets_taken)
    _LOGGER.info(f"{tickets_available} tickets available.")
    waitlist = rsvps[rsvps.response == "waitlist"]["attendees"].sum()
    _LOGGER.info(f"{waitlist} membes on the waitlist.")
    attendances = _attendances()
    raffle = _raffle(rsvps, attendances)
    winners = _winners(raffle, min(tickets_available, waitlist))
    if ADD_TO_GUESTLIST:
        for winner_id in winners:
            _add_to_guestlist(winner_id)
            time.sleep(0.5)


if __name__ == "__main__":
    main()
