"""Post-deploy health check of the live API: /health, then /ready.

    python scripts/check_deployment.py [--url https://<prod>] [--deploy-log vercel.log] \
        [--deploy-url https://<deployment>] [--attempts 12] [--interval 10]

Which URL: ``--url`` (the LIVE_API_URL override) if given, else the production
alias parsed from the ``vercel deploy --prod`` log ("Aliased: https://..."),
else ``--deploy-url``. Per-deployment URLs can sit behind Vercel Deployment
Protection, so:

* 200            ok
* 401 / 403      protected: cannot verify -> GitHub ``::warning::``, exit 0
* anything else  retried (3xx included: redirects are not followed); if it never turns 200 -> ``::error::``, exit 1 (a 404
                 means the routes are broken, which is a failed deploy)

Standard library only.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Vercel CLI 58+ prints "▲ Aliased   https://x.vercel.app" (no colon); older ones "Aliased: https://...".
ALIAS = re.compile(r"Aliased:?\s+(https://[A-Za-z0-9.-]+)")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Report a 3xx as-is. Deployment Protection answers 302 to its SSO page, and
    following it ends on a 200 login page: a false pass."""

    def redirect_request(self, *args, **kwargs):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def production_url(log_text: str) -> str | None:
    """The first production alias in a ``vercel deploy --prod`` log, if any."""
    match = ALIAS.search(log_text)
    return match.group(1) if match else None


def pick_url(override: str | None, log_text: str, deploy_url: str | None) -> str:
    for candidate in (override, production_url(log_text), deploy_url):
        if candidate and candidate.strip():
            url = candidate.strip().rstrip("/")
            if not url.startswith("https://"):
                raise ValueError(f"refusing to check a non-https URL: {url}")
            return url
    raise ValueError("no URL to check (set LIVE_API_URL or check the deploy log)")


def status_of(url: str, timeout: float) -> int:
    """HTTP status, or 0 for a timeout / connection error."""
    try:
        with _OPENER.open(url, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as err:
        return err.code
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0


def check(base: str, attempts: int = 12, interval: float = 10, timeout: float = 60,
          probe=status_of, sleep=time.sleep, out=print) -> int:
    for path in ("/health", "/ready"):
        code = 0
        for attempt in range(1, attempts + 1):
            code = probe(base + path, timeout)
            out(f"{path} attempt {attempt}: {code or 'timeout'}")
            if code == 200 or code in (401, 403):
                break
            if attempt < attempts:
                sleep(interval)
        if code == 200:
            continue
        if code in (401, 403):
            out(f"::warning title=health check skipped::{base} answered {code} (Deployment Protection?); "
                "cannot verify. Set the LIVE_API_URL variable to the public production URL.")
            return 0
        out(f"::error title=health check failed::{base}{path} answered {code or 'timeout'} "
            f"after {attempts} attempts")
        return 1
    out(f"{base}: /health and /ready are 200")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default=None, help="override (LIVE_API_URL)")
    ap.add_argument("--deploy-log", type=Path, help="combined output of `vercel deploy --prod`")
    ap.add_argument("--deploy-url", default=None)
    ap.add_argument("--attempts", type=int, default=12)
    ap.add_argument("--interval", type=float, default=10)
    ap.add_argument("--timeout", type=float, default=60)
    args = ap.parse_args(argv)
    log_text = args.deploy_log.read_text(encoding="utf-8", errors="replace") if args.deploy_log else ""
    try:
        base = pick_url(args.url, log_text, args.deploy_url)
    except ValueError as exc:
        print(f"::error::{exc}")
        return 1
    print(f"checking {base}")
    return check(base, args.attempts, args.interval, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
