# meeting_brief.py
"""CLI tool that produces Markdown meeting briefs for the next 24 h.

Key design goals
–––––––––––––––––
• **Local-only** execution – no data stored off the machine.
• Works with either Google Calendar + Gmail *or* Apple Calendar (via `caldav`).
• External participant detection based on your corporate domain.
• Optional LinkedIn/company enrichment (uses Playwright browser automation).
• Summarisation can use OpenAI (if you export OPENAI_API_KEY) *or* fallback to a purely local heuristic.

Run `python meeting_brief.py --help` for usage.

Dependencies (add to `requirements.txt`)
–––––––––––––––––––––––––––––––––––––––
google-api-python-client
google-auth-httplib2
google-auth-oauthlib
python-dotenv
rich
openai>=1.0.0  # optional – comment out if not using
caldav  # only needed for Apple/iCloud users
beautifulsoup4, lxml  # for lightweight LinkedIn scraping (optional)
playwright>=1.40.0

First-time setup
–––––––––––––––
1.  Create a Google Cloud Project → OAuth consent screen (internal) → 'Desktop App' credentials.
    Save the JSON as `client_secret.json` in the same folder.
2.  Run `python meeting_brief.py --google-login` once; a browser window lets you grant
    Calendar + Gmail readonly scopes.  A token file (`token.json`) is stored locally.
3.  Export environment vars (add these to ~/.zshrc or ~/.bashrc):
    ```bash
    export CORP_DOMAIN="yourcorp.com"   # used to detect externals
    export OPENAI_API_KEY="sk-..."       # optional – otherwise rule-based summary
    ```

Examples
–––––––
• Brief all meetings for next 24 h, dump to `brief.md`:
  ```bash
  python meeting_brief.py --output brief.md
  ```
• Same but skip LinkedIn enrichment (faster, fully local):
  ```bash
  python meeting_brief.py --no-linkedin
  ```
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import sqlite3
import sys
import asyncio
from collections import defaultdict
from textwrap import shorten
from typing import List, Dict, Any, Optional
import base64
import re
import string

from rich.console import Console
from rich.progress import track

# Optional imports guarded at runtime
try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    from googleapiclient.discovery import build  # type: ignore
    from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore
    from google.oauth2.credentials import Credentials  # type: ignore
    from google.auth.transport.requests import Request  # type: ignore
except ImportError:
    build = InstalledAppFlow = Credentials = Request = None  # type: ignore

from linkedin_playwright import async_linkedin_lookup

# ---------- CONFIG ----------
SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/gmail.readonly",
]
CACHE_DIR = pathlib.Path.home() / ".cache" / "meeting_brief"
CACHE_DB = CACHE_DIR / "cache.db"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

console = Console()


# ---------- UTILS ----------

def check_environment() -> None:
    """Check and display environment variable status."""
    console.print("[bold blue]Environment Check:")
    
    # Check required variables
    corp_domain = os.environ.get("CORP_DOMAIN")
    if corp_domain:
        console.print(f"[green]✓ CORP_DOMAIN: {corp_domain}")
    else:
        console.print("[red]✗ CORP_DOMAIN: Not set (will use 'example.com' as default)")
    
    # Check optional variables
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        masked_key = openai_key[:8] + "..." + openai_key[-4:] if len(openai_key) > 12 else "***"
        console.print(f"[green]✓ OPENAI_API_KEY: {masked_key}")
    else:
        console.print("[yellow]⚠ OPENAI_API_KEY: Not set (OpenAI features disabled)")
    
    console.print()

def local_db() -> sqlite3.Connection:
    conn = sqlite3.connect(CACHE_DB)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS enrichment (
          key TEXT PRIMARY KEY,
          payload TEXT,
          ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    return conn


def cache_get(key: str) -> Optional[dict]:
    row = local_db().execute("SELECT payload FROM enrichment WHERE key=?", (key,)).fetchone()
    if row:
        return json.loads(row[0])
    return None


def cache_set(key: str, value: dict) -> None:
    local_db().execute(
        "INSERT OR REPLACE INTO enrichment(key, payload, ts) VALUES(?, ?, CURRENT_TIMESTAMP)",
        (key, json.dumps(value)),
    )
    local_db().commit()


def openai_summarise(text: str, max_tokens: int = 256, debug: bool = False, extra_instructions: str = "") -> str:
    if not OpenAI:
        raise RuntimeError("openai package not installed – pip install openai OR use --no-openai")
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Set OPENAI_API_KEY env var OR use --no-openai to disable")
    
    # Create a more focused prompt for better summaries
    prompt = f"""Analyze this email conversation and provide a sharp, business-focused summary in 2-3 sentences. Focus on:
- Key decisions or agreements made
- Action items or next steps
- Important context or background shared
- Ignore any conversation around scheduling and finding time on the calendar
{extra_instructions}

Email conversation:
{text}

Summary:"""
    
    if debug:
        console.print(f"[magenta]OpenAI prompt ({len(prompt)} chars):")
        console.print(f"[dim]{prompt}")
    
    console.print(f"[magenta]Summarizing with OpenAI...")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.3,  # Lower temperature for more focused output
    )
    summary = response.choices[0].message.content.strip()
    console.print(f"[green]Summary completed ({len(summary)} chars)")
    if debug:
        console.print(f"[dim]Summary: {summary}")
    return summary


def is_meaningful_text(text: str) -> bool:
    """Return True if text is likely meaningful (not empty, not just base64, not just gibberish)."""
    if not text or not text.strip():
        return False
    # If it's just one long word (no spaces, >100 chars), likely not meaningful
    if len(text.strip().split()) <= 1 and len(text.strip()) > 100:
        return False
    # If it's mostly non-printable or non-ASCII
    printable = set(string.printable)
    ratio = sum(1 for c in text if c in printable) / max(1, len(text))
    if ratio < 0.7:
        return False
    # If it's just base64 (urlsafe)
    base64ish = re.fullmatch(r'[A-Za-z0-9_\-\n=]+', text.strip())
    if base64ish and len(text.strip()) > 100:
        return False
    # If it's just a repeated character
    if len(set(text.strip())) == 1:
        return False
    return True


# ---------- GOOGLE AUTH ----------

def google_creds(interactive: bool = False) -> Credentials:  # type: ignore[name-defined]
    token_file = CACHE_DIR / "token.json"
    creds: Optional[Credentials] = None  # noqa: F821
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)  # type: ignore[arg-type]
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())  # type: ignore
        elif interactive:
            flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", SCOPES)  # type: ignore[arg-type]
            creds = flow.run_local_server(port=0)
            token_file.write_text(creds.to_json())
        else:
            console.print("[red]Need credentials – run with --google-login once first.")
            sys.exit(1)
    return creds  # type: ignore[return-value]


# ---------- GOOGLE API WRAPPERS ----------

def get_events(creds, hours: int = 24) -> List[dict]:  # type: ignore[override]
    console.print(f"[blue]Fetching calendar events for the next {hours} hours...")
    service = build("calendar", "v3", credentials=creds)
    now = dt.datetime.utcnow().isoformat() + "Z"
    max_time = (dt.datetime.utcnow() + dt.timedelta(hours=hours)).isoformat() + "Z"
    events_result = (
        service.events()
        .list(calendarId="primary", timeMin=now, timeMax=max_time, singleEvents=True, orderBy="startTime")
        .execute()
    )
    events = events_result.get("items", [])
    console.print(f"[green]Found {len(events)} total events")
    return events


def extract_email_bodies(payload, debug=False):
    """Recursively extract and decode all text/plain (or fallback text/html) bodies from a Gmail message payload."""
    bodies = []
    if not payload:
        return bodies
    # Prefer text/plain
    if payload.get('mimeType', '').startswith('text/plain'):
        data = payload.get('body', {}).get('data', '')
        if data:
            if re.fullmatch(r'[A-Za-z0-9_-]+', data):
                try:
                    decoded = base64.urlsafe_b64decode(data + '===').decode('utf-8', errors='replace')
                    bodies.append(decoded)
                except Exception as e:
                    if debug:
                        console.print(f"[red]Failed to decode email body: {e}")
                    bodies.append(data)
            else:
                bodies.append(data)
    # Fallback to text/html if no text/plain
    elif payload.get('mimeType', '').startswith('text/html'):
        data = payload.get('body', {}).get('data', '')
        if data:
            if re.fullmatch(r'[A-Za-z0-9_-]+', data):
                try:
                    decoded = base64.urlsafe_b64decode(data + '===').decode('utf-8', errors='replace')
                    bodies.append(decoded)
                except Exception as e:
                    if debug:
                        console.print(f"[red]Failed to decode email body: {e}")
                    bodies.append(data)
            else:
                bodies.append(data)
    # Recursively check parts
    for part in payload.get('parts', []) or []:
        bodies.extend(extract_email_bodies(part, debug=debug))
    return bodies


def gmail_search(creds, query: str, max_results: int = 10, debug: bool = False) -> List[str]:  # type: ignore[override]
    if debug:
        console.print(f"[yellow]Searching Gmail for: {query}")
    service = build("gmail", "v1", credentials=creds)
    response = service.users().messages().list(userId="me", q=query, maxResults=max_results).execute()
    ids = [m["id"] for m in response.get("messages", [])]
    if debug:
        console.print(f"[yellow]Found {len(ids)} email messages")
    bodies: List[str] = []
    for i, mid in enumerate(ids, 1):
        if debug:
            console.print(f"[yellow]Processing email {i}/{len(ids)}")
        msg = service.users().messages().get(userId="me", id=mid, format="full").execute()
        payload = msg.get('payload', {})
        extracted = extract_email_bodies(payload, debug=debug)
        bodies.extend(extracted)
    if debug:
        console.print(f"[green]Extracted {len(bodies)} email bodies")
    return bodies


# ---------- LINKEDIN SCRAPER (OPTIONAL) ----------

import pathlib
CONTEXT_PATH = pathlib.Path.home() / ".linkedin_context.json"

def linkedin_lookup(email: str, display_name: Optional[str] = None, debug: bool = False) -> Optional[dict]:
    import traceback
    try:
        if debug:
            console.print(f"[magenta]Starting async_linkedin_lookup for {email} ({display_name})")
        result = asyncio.run(async_linkedin_lookup(email, display_name=display_name, debug=debug))
        if debug:
            console.print(f"[magenta]async_linkedin_lookup finished for {email}: {result}")
        return result
    except Exception as e:
        console.print(f"[red]LinkedIn Playwright error for {email}: {e}")
        if debug:
            console.print(traceback.format_exc())
        return None



# ---------- CORE LOGIC ----------

def openai_extract_name_company(email: str, display_name: str, debug: bool = False) -> dict:
    """Use OpenAI's structured outputs API to extract first name, last name, company, and optional notes from email and display name."""
    if not OpenAI or "OPENAI_API_KEY" not in os.environ:
        return {}
    function_schema = [
        {
            "name": "extract_person_info",
            "description": "Extract a person's first name, last name, company, and optional notes from email and display name. If unsure, perform a web search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "first_name": {"type": "string", "description": "First name of the person, if available."},
                    "last_name": {"type": "string", "description": "Last name of the person, if available."},
                    "company": {"type": "string", "description": "Current company of the person, if available."},
                    "notes": {"type": "string", "description": "Any extra comments, uncertainty, or context."}
                },
                "required": []
            },
        }
    ]
    prompt = (
        f'Given the following information about a person:\n'
        f'- Email: {email}\n'
        f'- Display Name: {display_name}\n\n'
        'Extract the person\'s first name, last name, and current company (if available). '
        'If you are unsure, perform a web search to find the most likely answer.'
    )
    if debug:
        console.print(f"[magenta]OpenAI extraction prompt (function call):\n{prompt}")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        functions=function_schema,
        function_call={"name": "extract_person_info"},
        max_tokens=150,
        temperature=0.1,
    )
    result = {}
    try:
        func_args = response.choices[0].message.function_call.arguments
        import json as _json
        result = _json.loads(func_args)
    except Exception as e:
        if debug:
            console.print(f"[red]OpenAI function_call error: {e}")
            console.print(f"[red]Raw function_call arguments: {getattr(response.choices[0].message, 'function_call', None)}")
        result = {}
    if debug:
        console.print(f"[magenta]OpenAI extracted: {result}")
        if isinstance(result, dict) and result.get("notes"):
            console.print(f"[yellow]OpenAI notes: {result['notes']}")
    return result

def build_brief(events: List[dict], creds, args) -> str:  # type: ignore[override]
    md_lines: List[str] = []
    corp_domain = os.environ.get("CORP_DOMAIN", "example.com").lower()
    
    if args.debug:
        console.print(f"[blue]Processing {len(events)} events for external attendees...")
    external_meetings = 0

    for i, ev in enumerate(track(events, description="Processing events"), 1):
        attendees = ev.get("attendees", [])
        externals = [
            a for a in attendees
            if not a.get("email", "").lower().endswith(corp_domain)
            and not a.get("email", "").lower().endswith("@resource.calendar.google.com")
        ]
        
        if not externals:
            if args.debug:
                console.print(f"[dim]Event {i}/{len(events)}: {ev.get('summary', '(No Title)')} - internal only, skipping")
            continue  # skip internal-only meetings
        
        external_meetings += 1
        if args.debug:
            console.print(f"[blue]Event {i}/{len(events)}: {ev.get('summary', '(No Title)')} - {len(externals)} external attendees")
        
        start = ev["start"].get("dateTime", ev["start"].get("date"))
        md_lines.append(f"## {ev.get('summary','(No Title)')}  –  {start}")
        md_lines.append("")

        # Collect all external emails, names, and companies for this event
        external_emails = set()
        external_names = set()
        external_companies = set()
        profiles = {}
        for a in externals:
            email = a.get("email", "")
            display = a.get("displayName", email)
            external_emails.add(email)
            if display and display != email:
                external_names.add(display)
            # Lookup LinkedIn once per attendee
            company = None
            if not args.no_linkedin:
                # Try to extract company from email domain (with and without TLD)
                domain = email.split('@')[-1] if '@' in email else ''
                company_candidates = set()
                if domain:
                    # Add full domain (e.g., acme.com)
                    company_candidates.add(domain)
                    # Add domain without TLD (e.g., acme)
                    if '.' in domain:
                        company_candidates.add(domain.split('.')[0])
                # Use OpenAI to extract name/company (let OpenAI do its own web search if needed)
                extracted = openai_extract_name_company(email, display, debug=args.debug)
                first_name = extracted.get("first_name", "")
                last_name = extracted.get("last_name", "")
                company = extracted.get("company", "")
                # Add OpenAI-extracted company and candidates from email
                if company:
                    company_candidates.add(company)
                # Use extracted name/company for LinkedIn search
                linkedin_query = f"{first_name} {last_name} {' '.join(company_candidates)}".strip()
                if args.debug:
                    console.print(f"[magenta]LinkedIn search query: {linkedin_query}")
                    console.print(f"[magenta]Company candidates from email: {company_candidates}")
                profile = linkedin_lookup("", display_name=linkedin_query, debug=args.debug)
                if args.debug:
                    console.print(f"[magenta]linkedin_lookup returned for {linkedin_query}: {profile}")
                profiles[email] = profile
                if profile:
                    company = profile.get("data", {}).get("companyName", None)
                    if company:
                        external_companies.add(company)
                # Add all company candidates to external_companies for Gmail search
                for cc in company_candidates:
                    external_companies.add(cc)

        # Build a single Gmail search query for all externals
        search_terms = [f"to:{email} OR from:{email}" for email in external_emails]
        search_terms += [name for name in external_names]
        search_terms += [company for company in external_companies]
        gmail_query = " OR ".join(search_terms)
        if args.debug:
            console.print(f"[yellow]Gmail search query for event: {gmail_query}")
        all_bodies = gmail_search(creds, gmail_query, debug=args.debug)
        all_bodies = list({body for body in all_bodies})  # deduplicate
        # Sort emails in reverse chronological order (newest first) and limit context length
        # Note: Gmail API returns messages in reverse chronological order by default
        joined = "\n---\n".join(all_bodies)
        # Limit to 100,000 characters to stay within reasonable context length
        if len(joined) > 100000:
            if args.debug:
                console.print(f"[yellow]Email context too long ({len(joined)} chars), truncating to 100,000 chars")
            joined = joined[:100000]
            # Try to truncate at a reasonable boundary (end of an email)
            last_separator = joined.rfind("\n---\n")
            if last_separator > 50000:  # Only truncate at separator if it's not too early
                joined = joined[:last_separator]
            if args.debug:
                console.print(f"[yellow]Truncated email context to {len(joined)} chars")

        # Summarize emails for the event
        if not args.no_email:
            if is_meaningful_text(joined):
                if OpenAI and not args.no_openai:
                    summary = openai_summarise(
                        joined,
                        debug=args.debug,
                        extra_instructions=(
                            "Only consider emails relevant to the whole group (not just internal-only threads). "
                            "Do not include any information about scheduling or calendar coordination in the summary."
                        )
                    )
                else:
                    if args.debug:
                        console.print(f"[yellow]Using rule-based summary (no OpenAI)")
                    summary = shorten(joined, 400)
            else:
                summary = "(No meaningful past email conversation found)"
            md_lines.append("**Past conversations:** " + summary)

        # For each external, add LinkedIn enrichment and attendee section
        for j, a in enumerate(externals, 1):
            email = a.get("email", "")
            display = a.get("displayName", email)
            md_lines.append(f"### {display}  (<{email}>)")
            profile = profiles.get(email)
            company = profile.get("data", {}).get("companyName", None) if profile else None
            # LinkedIn enrichment
            if not args.no_linkedin:
                if profile:
                    headline = profile.get("data", {}).get("headline", "")
                    if args.debug:
                        console.print(f"[cyan]    LinkedIn headline: {headline}")
                        console.print(f"[cyan]    Current company: {company}")
                    # Get work experience summary from async_linkedin_lookup
                    experience_summary = profile.get("data", {}).get("summary", "No work experience summary available")
                    if args.debug:
                        console.print(f"[cyan]    Professional background summary: {experience_summary}")
                    md_lines.append(f"**LinkedIn:** {headline} @ {company if company else ''}")
                    md_lines.append(f"**Work Experience:** {experience_summary}")
                    # Add LinkedIn URLs if available
                    profile_url = profile.get("data", {}).get("profile_url")
                    search_results_url = profile.get("data", {}).get("search_results_url")
                    if profile_url:
                        md_lines.append(f"**LinkedIn Profile:** [View Profile]({profile_url})")
                    if search_results_url:
                        md_lines.append(f"**Search Results:** [View Search]({search_results_url})")
                else:
                    if args.debug:
                        console.print(f"[dim]    No LinkedIn data available for {email}")
            else:
                if args.debug:
                    console.print(f"[dim]    LinkedIn lookup disabled for {email}")
            md_lines.append("")
        md_lines.append("\n---\n")

    if args.debug:
        console.print(f"[green]Completed processing {external_meetings} external meetings")
    return "\n".join(md_lines) or "No external meetings found in the next period."


# ---------- MAIN ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Markdown briefs for upcoming meetings (24 h by default).")
    parser.add_argument("--hours", type=int, default=24, help="Look-ahead window in hours")
    parser.add_argument("--output", "-o", type=pathlib.Path, default=None, help="Markdown output file (defaults to stdout)")
    parser.add_argument("--google-login", action="store_true", help="Interactive Google OAuth flow to create token.json")
    parser.add_argument("--no-linkedin", action="store_true", help="Skip LinkedIn enrichment")
    parser.add_argument("--no-email", action="store_true", help="Skip Gmail conversation summary")
    parser.add_argument("--no-openai", action="store_true", help="Never call OpenAI even if key is present")
    parser.add_argument("--debug", action="store_true", help="Enable detailed debug logging")
    args = parser.parse_args()

    console.print("[bold blue]Meeting Brief Generator Starting...")
    check_environment()
    console.print(f"[dim]Configuration: hours={args.hours}, linkedin={not args.no_linkedin}, email={not args.no_email}, openai={not args.no_openai}, debug={args.debug}")
    
    creds = google_creds(interactive=args.google_login)
    if args.google_login:
        console.print("[green]Google credentials stored.  Re-run without --google-login to generate briefs.")
        return

    events = get_events(creds, hours=args.hours)
    
    md = build_brief(events, creds, args)

    if args.output:
        args.output.write_text(md)
        console.print(f"[green]✓ Wrote {len(md.splitlines())} lines to {args.output}")
    else:
        console.print(md)
    
    console.print("[bold green]Meeting brief generation completed!")


if __name__ == "__main__":
    main()
