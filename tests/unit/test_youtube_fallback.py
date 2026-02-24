"""Tests for YouTube Downloader 5-Level Fallback Chain."""

import importlib
import os
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Skill directory uses a hyphen -- import via importlib
_tools = importlib.import_module("skills.youtube-downloader.tools")
_innertube = _tools._extract_transcript_via_innertube
_ytdlp = _tools._extract_transcript_via_ytdlp
_third_party = _tools._extract_transcript_via_third_party
_parse_vtt = _tools._parse_vtt_file

FAKE_ID = "dQw4w9WgXcQ"
FAKE_URL = f"https://www.youtube.com/watch?v={FAKE_ID}"
INNERTUBE_HTML = (
    'var ytInitialPlayerResponse = {"captions":{"playerCaptionsTracklistRenderer":'
    '{"captionTracks":[{"baseUrl":"https://captions.example.com/en",'
    '"languageCode":"en"}]}}};'
)
CAPTION_XML = (
    '<transcript><text start="0.5">Hello</text>'
    '<text start="2.1">World &amp; friends</text></transcript>'
)
SAMPLE_VTT = (
    "WEBVTT\n\n00:00:01.000 --> 00:00:04.000\nHello world\n\n"
    "00:00:05.500 --> 00:00:08.200\nSecond segment here\n\n"
    "00:00:05.500 --> 00:00:08.200\nHello world\n"
)
SUPADATA_RESP = {
    "lang": "en",
    "content": [
        {"text": "Hello", "offset": 1500, "duration": 3000, "lang": "en"},
        {"text": "World", "offset": 5000, "duration": 2000, "lang": "en"},
    ],
}
SEGS = [{"text": "OK", "start": 0.0}]


def _mock_resp(status=200, json_data=None, text=""):
    r = MagicMock(status_code=status, text=text)
    r.json.return_value = json_data or {}
    r.raise_for_status = MagicMock()
    if status >= 400:
        r.raise_for_status.side_effect = Exception(f"HTTP {status}")
    return r


# ── Level 2: Innertube ──────────────────────────────────────────────────────


@pytest.mark.unit
class TestInnertube:
    @patch("requests.get")
    def test_success(self, mock_get):
        mock_get.side_effect = [_mock_resp(text=INNERTUBE_HTML), _mock_resp(text=CAPTION_XML)]
        result = _innertube(FAKE_ID)
        assert len(result) == 2
        assert result[0] == {"text": "Hello", "start": 0.5}
        assert result[1]["text"] == "World & friends"  # HTML entity unescaped

    @patch("requests.get")
    def test_no_player_response(self, mock_get):
        mock_get.return_value = _mock_resp(text="<html>nothing</html>")
        assert _innertube(FAKE_ID) is None

    @patch("requests.get")
    def test_empty_captions(self, mock_get):
        mock_get.return_value = _mock_resp(text='var ytInitialPlayerResponse = {"captions":{}};')
        assert _innertube(FAKE_ID) is None

    @patch("requests.get", side_effect=ConnectionError("fail"))
    def test_network_error(self, _):
        assert _innertube(FAKE_ID) is None

    @patch("requests.get")
    def test_http_error(self, mock_get):
        mock_get.return_value = _mock_resp(status=403, text="Forbidden")
        assert _innertube(FAKE_ID) is None


# ── Level 3: yt-dlp ─────────────────────────────────────────────────────────


@pytest.mark.unit
class TestYtdlp:
    @patch("shutil.which", return_value=None)
    def test_not_installed(self, _):
        assert _ytdlp(FAKE_ID) is None

    def test_success_with_vtt(self):
        with (
            patch("shutil.which", return_value="/usr/bin/yt-dlp"),
            patch("subprocess.run", return_value=MagicMock(stderr="", returncode=0)),
            patch.object(_tools, "_parse_vtt_file", return_value=SEGS),
            patch("pathlib.Path.glob", return_value=["/tmp/fake.vtt"]),
        ):
            result = _ytdlp(FAKE_ID)
        assert result == SEGS

    def test_timeout(self):
        with (
            patch("shutil.which", return_value="/usr/bin/yt-dlp"),
            patch("subprocess.run", side_effect=subprocess.TimeoutExpired("yt-dlp", 60)),
            patch("pathlib.Path.glob", return_value=[]),
        ):
            assert _ytdlp(FAKE_ID) is None

    def test_cookie_file_env(self):
        mock_run = MagicMock(return_value=MagicMock(stderr="", returncode=0))
        with (
            patch("shutil.which", return_value="/usr/bin/yt-dlp"),
            patch("subprocess.run", mock_run),
            patch("pathlib.Path.glob", return_value=[]),
            patch.dict(os.environ, {"YOUTUBE_COOKIES_FILE": "/tmp/cookies.txt"}),
            patch("os.path.isfile", return_value=True),
        ):
            _ytdlp(FAKE_ID)
        assert any("--cookies" in c[0][0] for c in mock_run.call_args_list)


# ── Level 4: Third-party API ────────────────────────────────────────────────


@pytest.mark.unit
class TestThirdPartyAPI:
    @patch("requests.get")
    def test_supadata_ms_conversion(self, mock_get):
        mock_get.return_value = _mock_resp(json_data=SUPADATA_RESP)
        with patch.dict(os.environ, {"SUPADATA_API_KEY": "key"}, clear=False):
            result = _third_party(FAKE_ID)
        assert result[0]["start"] == 1.5  # 1500ms
        assert result[1]["start"] == 5.0  # 5000ms

    @patch("requests.get")
    def test_supadata_plain_text(self, mock_get):
        mock_get.return_value = _mock_resp(json_data={"content": "Full transcript here."})
        with patch.dict(os.environ, {"SUPADATA_API_KEY": "key"}, clear=False):
            result = _third_party(FAKE_ID)
        assert len(result) == 1
        assert result[0]["start"] == 0.0

    @patch("requests.get")
    def test_supadata_401(self, mock_get):
        mock_get.return_value = _mock_resp(status=401)
        with patch.dict(os.environ, {"SUPADATA_API_KEY": "bad"}, clear=False):
            assert _third_party(FAKE_ID) is None

    @patch("requests.get")
    def test_supadata_429(self, mock_get):
        mock_get.return_value = _mock_resp(status=429)
        with patch.dict(os.environ, {"SUPADATA_API_KEY": "key"}, clear=False):
            assert _third_party(FAKE_ID) is None

    def test_no_api_keys(self):
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("SUPADATA_API_KEY", "YOUTUBE_TRANSCRIPT_API_URL")
        }
        with patch.dict(os.environ, env, clear=True):
            assert _third_party(FAKE_ID) is None

    @patch("requests.get", side_effect=TimeoutError("timeout"))
    def test_timeout(self, _):
        with patch.dict(os.environ, {"SUPADATA_API_KEY": "key"}, clear=False):
            assert _third_party(FAKE_ID) is None


# ── VTT Parsing ─────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestVTTParsing:
    def test_valid_vtt(self, tmp_path):
        (tmp_path / "t.vtt").write_text(SAMPLE_VTT, encoding="utf-8")
        result = _parse_vtt(str(tmp_path / "t.vtt"))
        assert len(result) == 2  # duplicate deduped
        assert result[0] == {"text": "Hello world", "start": 1.0}
        assert result[1] == {"text": "Second segment here", "start": 5.5}

    def test_empty_vtt(self, tmp_path):
        (tmp_path / "e.vtt").write_text("WEBVTT\n\n", encoding="utf-8")
        assert _parse_vtt(str(tmp_path / "e.vtt")) is None

    def test_missing_file(self):
        assert _parse_vtt("/nonexistent/fake.vtt") is None

    def test_strips_html_tags(self, tmp_path):
        (tmp_path / "t.vtt").write_text(
            "WEBVTT\n\n00:00:01.000 --> 00:00:04.000\n<c.color>Tagged</c> text\n\n",
            encoding="utf-8",
        )
        assert _parse_vtt(str(tmp_path / "t.vtt"))[0]["text"] == "Tagged text"

    def test_hour_timestamp(self, tmp_path):
        (tmp_path / "t.vtt").write_text(
            "WEBVTT\n\n01:30:05.000 --> 01:30:10.000\nLong video\n\n", encoding="utf-8"
        )
        assert _parse_vtt(str(tmp_path / "t.vtt"))[0]["start"] == 5405.0


# ── Fallback Orchestrator ────────────────────────────────────────────────────


def _patch_l1_fail():
    return patch.dict("sys.modules", {"youtube_transcript_api": None})


def _patch_levels(**overrides):
    """Patch L2-L5 with given return values (default None)."""
    defaults = {
        "_extract_transcript_via_innertube": None,
        "_extract_transcript_via_ytdlp": None,
        "_extract_transcript_via_third_party": None,
        "_extract_transcript_via_browser": MagicMock(return_value=None),
    }
    defaults.update({k: v for k, v in overrides.items()})
    from contextlib import ExitStack

    stack = ExitStack()
    mocks = {}
    for name, val in defaults.items():
        if isinstance(val, MagicMock):
            m = stack.enter_context(patch.object(_tools, name, val))
        else:
            m = stack.enter_context(patch.object(_tools, name, return_value=val))
        mocks[name] = m
    return stack, mocks


@pytest.mark.unit
class TestFallbackOrchestrator:
    def _run(self):
        return _tools._extract_transcript(FAKE_ID, FAKE_URL)

    def test_l1_success_skips_rest(self):
        yt_mod = MagicMock()
        tr = MagicMock()
        tr.snippets = [SimpleNamespace(text="L1", start=0.0)]
        yt_mod.YouTubeTranscriptApi.return_value.fetch.return_value = tr
        l2 = MagicMock(return_value=None)
        with (
            patch.dict("sys.modules", {"youtube_transcript_api": yt_mod}),
            patch.object(_tools, "_extract_transcript_via_innertube", l2),
        ):
            result = self._run()
        assert result[0]["text"] == "L1"
        l2.assert_not_called()

    def test_l3_succeeds_after_l1_l2_fail(self):
        stack, mocks = _patch_levels(_extract_transcript_via_ytdlp=SEGS)
        with _patch_l1_fail(), stack:
            result = self._run()
        assert result == SEGS
        mocks["_extract_transcript_via_innertube"].assert_called_once()
        mocks["_extract_transcript_via_third_party"].assert_not_called()

    def test_l4_success_skips_l5(self):
        stack, mocks = _patch_levels(_extract_transcript_via_third_party=SEGS)
        with _patch_l1_fail(), stack:
            result = self._run()
        assert result == SEGS
        mocks["_extract_transcript_via_browser"].assert_not_called()

    def test_all_fail(self):
        stack, _ = _patch_levels()
        with _patch_l1_fail(), stack, patch("asyncio.get_event_loop") as ml:
            ml.return_value.is_running.return_value = False
            with patch("asyncio.run", return_value=None):
                assert self._run() is None

    def test_l5_browser_fallback(self):
        browser_result = [{"text": "Browser", "start": 0.0}]
        stack, _ = _patch_levels()
        with _patch_l1_fail(), stack, patch("asyncio.get_event_loop") as ml:
            ml.return_value.is_running.return_value = False
            with patch("asyncio.run", return_value=browser_result):
                result = self._run()
        assert result[0]["text"] == "Browser"
