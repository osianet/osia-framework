"""
YouTube video uploader for OSIA weekly briefings.

Uses the YouTube Data API v3 with OAuth 2.0 to upload briefing videos.
Requires a one-time OAuth consent flow to generate a refresh token,
after which uploads run headlessly.

Setup:
  1. Create OAuth 2.0 credentials in Google Cloud Console (Desktop app type)
  2. Download the client secret JSON and save as config/youtube_client_secret.json
  3. Run: uv run python -m src.intelligence.youtube_uploader --auth
     This opens a browser for consent and saves the token to config/youtube_token.json
  4. Uploads will work headlessly from then on.
"""

import argparse
import json
import logging
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("osia.youtube_uploader")

_REPO_ROOT = Path(__file__).parent.parent.parent
_CLIENT_SECRET_PATH = _REPO_ROOT / "config" / "youtube_client_secret.json"
_TOKEN_PATH = _REPO_ROOT / "config" / "youtube_token.json"

YOUTUBE_UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
YOUTUBE_TOKEN_URL = "https://oauth2.googleapis.com/token"  # noqa: S105
YOUTUBE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
SCOPES = "https://www.googleapis.com/auth/youtube.upload"

# Default category: 25 = News & Politics
DEFAULT_CATEGORY_ID = "25"


def _load_client_credentials() -> dict:
    """Load OAuth client ID and secret from the client secret JSON."""
    if not _CLIENT_SECRET_PATH.exists():
        raise FileNotFoundError(
            f"YouTube client secret not found at {_CLIENT_SECRET_PATH}. "
            "Download it from Google Cloud Console → APIs & Services → Credentials."
        )
    data = json.loads(_CLIENT_SECRET_PATH.read_text())
    # Google exports as {"installed": {...}} or {"web": {...}}
    creds = data.get("installed") or data.get("web") or data
    return {
        "client_id": creds["client_id"],
        "client_secret": creds["client_secret"],
    }


def _load_token() -> dict | None:
    """Load saved OAuth token (access_token + refresh_token)."""
    if _TOKEN_PATH.exists():
        return json.loads(_TOKEN_PATH.read_text())
    return None


def _save_token(token_data: dict) -> None:
    """Persist OAuth token to disk."""
    _TOKEN_PATH.write_text(json.dumps(token_data, indent=2))
    logger.info("Token saved to %s", _TOKEN_PATH)


async def _refresh_access_token(client_id: str, client_secret: str, refresh_token: str) -> str:
    """Exchange a refresh token for a fresh access token."""
    async with httpx.AsyncClient(timeout=15.0) as http:
        resp = await http.post(
            YOUTUBE_TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["access_token"]


class YouTubeUploader:
    """Uploads videos to YouTube via the Data API v3."""

    def __init__(self) -> None:
        self._creds = _load_client_credentials()
        self._token_data = _load_token()
        if not self._token_data or "refresh_token" not in self._token_data:
            raise RuntimeError(
                "YouTube not authenticated. Run: uv run python -m src.intelligence.youtube_uploader --auth"
            )

    async def _get_access_token(self) -> str:
        """Get a valid access token, refreshing if needed."""
        return await _refresh_access_token(
            self._creds["client_id"],
            self._creds["client_secret"],
            self._token_data["refresh_token"],
        )

    async def upload(
        self,
        video_path: Path,
        title: str,
        description: str,
        tags: list[str] | None = None,
        category_id: str = DEFAULT_CATEGORY_ID,
        privacy_status: str = "unlisted",
    ) -> dict:
        """Upload a video to YouTube.

        Args:
            video_path: Path to the MP4 file.
            title: Video title (max 100 chars).
            description: Video description.
            tags: Optional list of tags.
            category_id: YouTube category ID (default: 25 = News & Politics).
            privacy_status: 'public', 'unlisted', or 'private'.

        Returns:
            Dict with video id, url, and status.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        access_token = await self._get_access_token()

        metadata = {
            "snippet": {
                "title": title[:100],
                "description": description[:5000],
                "tags": tags or [],
                "categoryId": category_id,
            },
            "status": {
                "privacyStatus": privacy_status,
                "selfDeclaredMadeForKids": False,
            },
        }

        file_size = video_path.stat().st_size
        logger.info("Uploading %s (%.1f MB) as '%s'", video_path.name, file_size / 1024 / 1024, title)

        # Initiate resumable upload
        async with httpx.AsyncClient(timeout=300.0) as http:
            init_resp = await http.post(
                f"{YOUTUBE_UPLOAD_URL}?uploadType=resumable&part=snippet,status",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json; charset=UTF-8",
                    "X-Upload-Content-Type": "video/mp4",
                    "X-Upload-Content-Length": str(file_size),
                },
                json=metadata,
            )
            init_resp.raise_for_status()
            upload_url = init_resp.headers["Location"]

            # Upload the video bytes
            video_bytes = video_path.read_bytes()
            upload_resp = await http.put(
                upload_url,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "video/mp4",
                    "Content-Length": str(file_size),
                },
                content=video_bytes,
            )
            upload_resp.raise_for_status()

        result = upload_resp.json()
        video_id = result.get("id", "unknown")
        url = f"https://www.youtube.com/watch?v={video_id}"
        logger.info("Upload complete: %s", url)

        return {
            "video_id": video_id,
            "url": url,
            "status": result.get("status", {}).get("uploadStatus", "unknown"),
        }


def _run_auth_flow() -> None:
    """Interactive OAuth 2.0 consent flow. Run once to generate a refresh token."""
    creds = _load_client_credentials()
    client_id = creds["client_id"]
    client_secret = creds["client_secret"]

    # Use out-of-band redirect for headless/CLI environments
    redirect_uri = "urn:ietf:wg:oauth:2.0:oob"

    auth_url = (
        f"{YOUTUBE_AUTH_URL}?"
        f"client_id={client_id}&"
        f"redirect_uri={redirect_uri}&"
        f"response_type=code&"
        f"scope={SCOPES}&"
        f"access_type=offline&"
        f"prompt=consent"
    )

    print("\n=== YouTube OAuth Setup ===")  # noqa: T201
    print("\n1. Open this URL in your browser:\n")  # noqa: T201
    # The auth URL intentionally contains the client_id — this is a public
    # identifier (not a secret) required by the OAuth consent flow. The user
    # must open this URL in their browser to authorize the app.
    print(auth_url)  # noqa: T201
    print("\n2. Sign in with the Google account that owns @the-osia channel")  # noqa: T201
    print("3. Authorize the app and copy the authorization code\n")  # noqa: T201

    code = input("Paste the authorization code here: ").strip()

    # Exchange code for tokens
    import httpx as _httpx

    resp = _httpx.post(
        YOUTUBE_TOKEN_URL,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
        },
    )
    resp.raise_for_status()
    token_data = resp.json()

    if "refresh_token" not in token_data:
        print("ERROR: No refresh_token in response. Try revoking app access and re-running.")
        return

    _save_token(token_data)
    print(f"\n✓ Authentication successful. Token saved to {_TOKEN_PATH}")
    print("  You can now run briefings with YouTube upload enabled.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YouTube uploader for OSIA briefings")
    parser.add_argument("--auth", action="store_true", help="Run the OAuth consent flow")
    args = parser.parse_args()

    if args.auth:
        _run_auth_flow()
    else:
        print("Usage: uv run python -m src.intelligence.youtube_uploader --auth")
