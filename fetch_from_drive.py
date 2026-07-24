"""
fetch_from_drive.py

Downloads all CSV files from a shared Google Drive folder (recursively)
into a local staging directory, using a read-only service account -- no
human's machine or Drive-for-Desktop sync required. Meant to run headlessly,
e.g. inside GitHub Actions on a schedule.

ONE-TIME SETUP (all free):
  1. Google Cloud Console -> create a project (if you don't have one) ->
     enable the "Google Drive API".
  2. IAM & Admin -> Service Accounts -> create one -> create a JSON key for
     it and download it.
  3. Open your shared Drive submissions folder -> Share -> paste the
     service account's email (looks like
     xxx@yyy.iam.gserviceaccount.com) with "Viewer" access. This is the
     step that actually grants access -- the key alone does nothing until
     the folder is shared with that email.
  4. In your GitHub repo: Settings -> Secrets and variables -> Actions ->
     add a secret GDRIVE_SA_KEY containing the full contents of the JSON
     key file, and a secret (or plain variable) GDRIVE_FOLDER_ID with the
     folder's ID (the long string in its Drive URL).

USAGE (locally, for testing):
    export GOOGLE_APPLICATION_CREDENTIALS=./service_account.json
    python fetch_from_drive.py --folder-id <FOLDER_ID> --out ./staging

USAGE (in GitHub Actions): see .github/workflows/dashboard.yml, which writes
the GDRIVE_SA_KEY secret out to a temp file and points
GOOGLE_APPLICATION_CREDENTIALS at it before calling this script.
"""

import argparse
import os
import random
import time

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Transient/server-side errors worth retrying (Google's own guidance: back off
# and retry on these). Anything else (403 permissions, 404 not found, etc.) is
# a real problem and should fail immediately rather than retry.
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


def with_retries(fn, max_attempts=5, base_delay=2.0):
    """Calls fn() with exponential backoff + jitter on transient Google API errors.
    Re-raises immediately on non-retryable errors, or after the last attempt."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except HttpError as e:
            status = getattr(e.resp, "status", None)
            if status not in RETRYABLE_STATUSES or attempt == max_attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
            print(f"  transient error (HTTP {status}), retrying in {delay:.1f}s "
                  f"(attempt {attempt}/{max_attempts})...", flush=True)
            time.sleep(delay)


def get_service():
    cred_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    creds = service_account.Credentials.from_service_account_file(cred_path, scopes=SCOPES)
    return build("drive", "v3", credentials=creds)


def list_csvs_recursive(service, folder_id, path_prefix=""):
    """Yields (file_id, relative_local_path) for every .csv under folder_id."""
    query = f"'{folder_id}' in parents and trashed = false"
    page_token = None
    while True:
        resp = with_retries(lambda: service.files().list(
            q=query,
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token,
        ).execute())

        for f in resp.get("files", []):
            name = f["name"]
            if f["mimeType"] == "application/vnd.google-apps.folder":
                yield from list_csvs_recursive(service, f["id"], os.path.join(path_prefix, name))
            elif name.lower().endswith(".csv") and not name.startswith("_"):
                yield f["id"], os.path.join(path_prefix, name)

        page_token = resp.get("nextPageToken")
        if not page_token:
            break


def download_file(service, file_id, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    request = service.files().get_media(fileId=file_id)
    with open(dest_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = with_retries(downloader.next_chunk)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder-id", required=True, help="Google Drive folder ID of the shared submissions folder")
    parser.add_argument("--out", default="./staging", help="Local folder to download CSVs into")
    args = parser.parse_args()

    service = get_service()
    count = 0
    for file_id, rel_path in list_csvs_recursive(service, args.folder_id):
        dest = os.path.join(args.out, rel_path)
        download_file(service, file_id, dest)
        count += 1
        print(f"downloaded {rel_path}", flush=True)

    print(f"\nDownloaded {count} CSV file(s) to {args.out}", flush=True)


if __name__ == "__main__":
    main()
