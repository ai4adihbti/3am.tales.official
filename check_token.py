#!/usr/bin/env python3
"""Check your Instagram token and (optionally) refresh it.

Long-lived Instagram tokens last 60 days. Refresh once a month:

    IG_ACCESS_TOKEN=... python scripts/check_token.py --refresh

then paste the printed token into the IG_ACCESS_TOKEN repository secret.
"""
import argparse
import os
import sys

import requests

ap = argparse.ArgumentParser()
ap.add_argument("--refresh", action="store_true")
args = ap.parse_args()

token = os.getenv("IG_ACCESS_TOKEN")
if not token:
    sys.exit("Set IG_ACCESS_TOKEN first.")

me = requests.get("https://graph.instagram.com/me",
                  params={"fields": "user_id,username,account_type", "access_token": token}, timeout=30).json()
if "error" in me:
    sys.exit(f"Token problem: {me['error'].get('message')}")
print(f"OK · @{me.get('username')} · account type {me.get('account_type')} · IG_USER_ID = {me.get('user_id')}")

if args.refresh:
    r = requests.get("https://graph.instagram.com/refresh_access_token",
                     params={"grant_type": "ig_refresh_token", "access_token": token}, timeout=30).json()
    if "access_token" not in r:
        sys.exit(f"Refresh failed: {r}")
    days = int(r.get("expires_in", 0)) // 86400
    print(f"\nNew token (valid ~{days} days) — paste into the IG_ACCESS_TOKEN secret:\n{r['access_token']}")
