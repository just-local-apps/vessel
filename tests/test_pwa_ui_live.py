"""Playwright smoke tests against the deployed Fly app.

Reads VESSEL_AUTH_TOKEN from .env (or env) and the target URL from
VESSEL_LIVE_URL (default https://vessel-ravi.fly.dev). Skips automatically
if no token is available.

Run with:
    uv run --extra dev --extra ui-test pytest tests/test_pwa_ui_live.py -v

These tests do NOT mutate prod state — they only navigate, click toggles,
and assert that the UI is wired up. Counts/contents are checked against
whatever the live state actually contains.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")
from playwright.sync_api import Page, expect, sync_playwright  # noqa: E402


def _load_env_token() -> str | None:
    env_var = os.environ.get("VESSEL_AUTH_TOKEN")
    if env_var:
        return env_var.strip()
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if not env_file.exists():
        return None
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line.startswith("VESSEL_AUTH_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


LIVE_URL = os.environ.get("VESSEL_LIVE_URL", "https://vessel-ravi.fly.dev")
TOKEN = _load_env_token()
SKIP_REASON = "set VESSEL_AUTH_TOKEN (or .env) to run live tests"


pytestmark = pytest.mark.skipif(not TOKEN, reason=SKIP_REASON)


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        yield b
        b.close()


@pytest.fixture
def page(browser):
    ctx = browser.new_context()
    p = ctx.new_page()
    # Pre-seed the token so the auth dialog never fires.
    p.add_init_script(
        f"localStorage.setItem('vessel.token', {json.dumps(TOKEN)});"
    )
    # Hard-disable the service worker so we always test the freshest shell.
    p.add_init_script(
        "if ('serviceWorker' in navigator) { "
        "navigator.serviceWorker.getRegistrations()"
        ".then(rs => rs.forEach(r => r.unregister())); }"
    )
    yield p
    ctx.close()


def _wait_for_render(page: Page) -> None:
    expect(page.locator('[data-testid="app-root"]'))\
        .to_have_attribute("data-loading", "false", timeout=10000)


def _open(page: Page) -> None:
    page.goto(f"{LIVE_URL}/pwa/?cb={os.urandom(4).hex()}")
    _wait_for_render(page)


# ---------------------------------------------------------------------------
# Smoke tests against the live deployment
# ---------------------------------------------------------------------------


def test_live_shell_serves_new_testids(page):
    """The deployed HTML must include the testid attributes for the
    single-view shell — proves the latest build is what's being served.
    The footer carries refresh + sign-out; the show-all / show-calendar
    toggles are gone because the calendar is the only view."""
    _open(page)
    for tid in [
        "app-root",
        "day-nav",
        "day-label",
        "refresh-footer",
        "sign-out",
        "calendar-section",
        "calendar-events",
        "swipe-area",
        "pull-indicator",
        "chat-area",
        "chat-input",
        "move-dialog",
    ]:
        expect(page.locator(f'[data-testid="{tid}"]')).to_have_count(1)


def test_live_default_mode_is_calendar_and_renders_without_error(page):
    """The app is a single calendar view. Opening it lands directly on
    'what's on my schedule', with no toggling required. The render must
    complete without setting `data-error`."""
    _open(page)
    root = page.locator('[data-testid="app-root"]')
    expect(root).to_have_attribute("data-mode", "calendar")
    expect(page.locator('[data-testid="day-label"]')).to_have_text("calendar")
    assert root.get_attribute("data-error") in (None, "")


def test_live_calendar_interleaves_open_tasks_with_events(page):
    """The calendar view renders BOTH event cards and open task cards —
    they share the day groups so the worklist sits alongside scheduled
    blocks. Past-due open tasks roll forward and surface under today's
    group; future tasks stay on their own due date."""
    _open(page)
    payload = page.evaluate(
        """
        async () => {
          const t = localStorage.getItem('vessel.token');
          const r = await fetch('/api/tasks/all', {
              headers: { Authorization: 'Bearer ' + t } });
          return await r.json();
        }
        """
    )
    today_iso = payload["now"][:10]
    open_tasks = [
        t for t in payload.get("tasks", [])
        if t.get("completed_at") is None and t.get("skipped_at") is None
    ]
    open_events = [
        e for e in payload.get("events", [])
        if e.get("completed_at") is None and e.get("skipped_at") is None
    ]
    for task in open_tasks:
        effective = today_iso if task["due_date"] < today_iso else task["due_date"]
        card = page.locator(f'[data-testid="task-card-{task["id"]}"]')
        expect(card).to_be_visible()
        day_group = page.locator(f'[data-testid="calendar-day-{effective}"]')
        expect(day_group.locator(f'[data-testid="task-card-{task["id"]}"]')).to_have_count(1)
    for ev in open_events:
        expect(
            page.locator(f'[data-testid="event-card-{ev["id"]}"]')
        ).to_be_visible()


def test_live_health_reports_tracing_enabled(page):
    """The deployed /health endpoint must show the tracing pipeline as fully
    wired. If any stage flips false, traces stop reaching Phoenix."""
    _open(page)
    raw = page.evaluate(
        """
        async () => {
          const r = await fetch('/health');
          return await r.json();
        }
        """
    )
    assert raw.get("ok") is True
    t = raw.get("tracing", {})
    assert t.get("initialized") is True, "observability.init() never ran"
    assert t.get("configured") is True, (
        "PHOENIX_API_KEY/PHOENIX_COLLECTOR_ENDPOINT not set on the deployed app"
    )
    assert t.get("tracer_provider_set") is True, (
        "phoenix.otel.register() failed: " + str(t.get("error"))
    )
    assert t.get("anthropic_instrumented") is True, (
        "Anthropic SDK not instrumented: " + str(t.get("error"))
    )


def test_live_skip_dialog_is_present_and_usable(page):
    """Sanity check that runs even when live state has zero tasks: the
    skip-dialog and its controls are present in the deployed shell, the
    dialog opens via showModal(), and the cancel button closes it. This
    validates the UI surface without needing a task to exist."""
    _open(page)
    # All the new elements must exist exactly once on the deployed shell.
    for tid in ["skip-dialog", "skip-reason", "skip-submit", "skip-cancel"]:
        expect(page.locator(f'[data-testid="{tid}"]')).to_have_count(1)

    # Drive showModal() programmatically — same call our JS makes when
    # the user swipes left or clicks the per-card "skip" button.
    page.evaluate(
        "document.getElementById('skip-dialog').showModal()"
    )
    expect(page.locator('[data-testid="skip-dialog"]')).to_be_visible()

    # The cancel-button click handler is wired up by promptSkipReason() at
    # call time. Here we just exercise the dialog primitive so we don't
    # depend on a specific task existing — close() is what cancel ultimately
    # invokes.
    page.evaluate("document.getElementById('skip-dialog').close()")
    expect(page.locator('[data-testid="skip-dialog"]')).to_be_hidden()


def test_live_swipe_left_opens_skip_dialog_and_records_reason(page):
    """REMOVED — this test grabbed the first pending task in live state
    and actually skipped it (with reason "swipe-probe reason"), which
    silently destroyed the user's real worklist on every post-deploy
    run. The user spotted it when 'wash dishes' kept disappearing.

    The swipe → POST /api/tasks/{id}/skip path is now covered by the
    hermetic Playwright suite (tests/test_pwa_ui.py
    ::test_swipe_left_in_all_view_skips_task_via_api), which uses
    expect_response to assert the POST body — without touching the
    deployed Fly state. The dialog-rendering smoke is still here as
    test_live_skip_dialog_is_present_and_usable.

    Leaving the function name as a tombstone so the next person who
    finds an old test report doesn't think the live skip flow is
    untested. Marked as skipped so it shows up explicitly in the run."""
    pytest.skip(
        "removed — this test mutated real state on every run; "
        "covered hermetically by test_swipe_left_in_all_view_skips_task_via_api"
    )


def test_live_cancel_button_prefills_chat_input(page):
    """Tapping a task card's `cancel` button no longer opens the
    dedicated skip dialog — that flow was unified into the chat
    assistant. The button now primes the chat input with
    `cancel/change "<title>" [id:<id>]: ` and focuses it; the user
    types the reason and submits through the same path as any other
    CRUD intent. The id is included so the chat assistant resolves
    the entity unambiguously. This test does NOT submit (would
    mutate live state); it only verifies the prefill + focus happens
    and that no legacy /skip POST fires."""
    _open(page)
    state = page.evaluate(
        """
        async () => {
          const t = localStorage.getItem('vessel.token');
          const r = await fetch('/api/state', {
              headers: { Authorization: 'Bearer ' + t } });
          if (!r.ok) return { error: r.status };
          return await r.json();
        }
        """
    )
    today_tasks = [
        t for t in ((state or {}).get("state") or {}).get("tasks") or []
        if not t.get("completed_at") and not t.get("skipped_at")
    ]
    if not today_tasks:
        pytest.skip("no pending tasks to exercise the cancel button")
    task = today_tasks[0]
    task_id = task["id"]
    task_title = task.get("title", "")

    card = page.locator(f'[data-testid="task-card-{task_id}"]')
    if card.count() == 0:
        pytest.skip("task not present in current view")

    # Track skip POSTs — must be zero, the legacy endpoint is no
    # longer driven by the UI.
    posts: list[str] = []
    page.on(
        "request",
        lambda req: posts.append(req.url) if "/skip" in req.url and req.method == "POST" else None,
    )
    page.locator(f'[data-testid="task-cancel-{task_id}"]').click()

    # No skip dialog appears.
    expect(page.locator('[data-testid="skip-dialog"]')).not_to_be_visible()
    # Chat input is prefilled and focused.
    expected_prefix = f'cancel/change "{task_title}" [id:{task_id}]: '
    chat_input = page.locator('[data-testid="chat-input"]')
    expect(chat_input).to_have_value(expected_prefix)
    assert page.evaluate(
        "document.activeElement === document.getElementById('chat-input')"
    )
    # Clear the input so we don't accidentally submit on a stray
    # Enter; this avoids mutating live state.
    chat_input.fill("")
    page.wait_for_timeout(150)
    assert posts == [], f"unexpected /skip POSTs: {posts}"


def test_live_disclaimer_link_in_footer(page):
    """The disclaimer link must exist in the footer and navigate to the
    disclaimer page, which shows the no-warranty copy."""
    _open(page)
    link = page.locator('[data-testid="disclaimer-link"]')
    expect(link).to_have_count(1)
    expect(link).to_be_visible()

    with page.expect_navigation():
        link.click()

    assert "/disclaimer" in page.url
    # Core section headings must be present.
    for heading in ["no warranty", "your data", "your key", "no support"]:
        expect(page.get_by_role("heading", name=heading)).to_be_visible()


def test_live_disclaimer_back_link_returns_to_app(page):
    """The back link on the disclaimer page must navigate back to /pwa/."""
    page.goto(f"{LIVE_URL}/pwa/disclaimer.html")
    back = page.get_by_text("← back to vessel", exact=False)
    expect(back).to_be_visible()
    with page.expect_navigation():
        back.click()
    assert "/pwa/" in page.url


def test_live_sign_out_clears_token_and_shows_auth_dialog(page):
    """Clicking sign out must clear the stored token and re-show the
    access-key dialog so a new key can be entered."""
    _open(page)
    token_before = page.evaluate("localStorage.getItem('vessel.token')")
    assert token_before, "token should be set before sign-out"

    page.locator('[data-testid="sign-out"]').click()

    token_after = page.evaluate("localStorage.getItem('vessel.token')")
    assert not token_after, "token must be cleared after sign-out"

    expect(page.locator('[data-testid="token-dialog"]')).to_be_visible()
    expect(page.locator('[data-testid="token-input"]')).to_be_visible()


def test_live_state_endpoint_diagnostic(page):
    """Diagnostic: dump /api/state so we can see exactly what the deployed
    server has for this token. Helps explain a 0-count 'all' view."""
    _open(page)
    raw = page.evaluate(
        """
        async () => {
          const t = localStorage.getItem('vessel.token');
          const r = await fetch('/api/state', { headers: { Authorization: 'Bearer ' + t } });
          if (!r.ok) return { error: r.status };
          const d = await r.json();
          const s = d.state || {};
          return {
            projects: (s.projects || []).length,
            tasks: (s.tasks || []).length,
            tasks_open: (s.tasks || []).filter(t => !t.completed_at).length,
            tasks_completed: (s.tasks || []).filter(t => !!t.completed_at).length,
            calendar: (s.calendar || []).length,
            now: d.now,
            current_window: d.current_window,
          };
        }
        """
    )
    print("\n[live state]", json.dumps(raw, indent=2))
    assert "error" not in raw, f"/api/state returned: {raw}"
