"""Playwright UI tests for the Vessel PWA.

Each test boots a fresh in-memory FastAPI app on a random port, navigates
Chromium to /pwa/, and exercises one feature via stable data-testid hooks
defined in index.html / app.js.

Run only this file with:
    uv run --extra dev --extra ui-test pytest tests/test_pwa_ui.py -v

The PWA is a single-view app: the calendar is the home screen. Events
are rendered per day. There is no task view, no all-tasks mode — only
the calendar view, an always-visible chat row, and a sign-out button in
the footer.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pytest

playwright = pytest.importorskip("playwright.sync_api")
from playwright.sync_api import Page, expect, sync_playwright  # noqa: E402

from tests._pwa_app import (  # noqa: E402
    build_app,
    make_state_full,
    make_state_with_calendar_only,
    make_state_with_one_open_event,
    start_server,
    test_now_local,
)
from vessel.models import CalendarEvent, StateData


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        yield b
        b.close()


@pytest.fixture
def page_factory(browser):
    """Returns a function that boots a server with the given state and
    yields (page, base_url, state_box). Cleans up server + page after
    the test.

    Pass `chat_client=` to inject a fake OpenAI-shaped client for the
    chat endpoint (chat tests use this to deliver deterministic tool
    calls without an LLM).
    """
    started: list = []

    def _make(state: StateData, *, chat_client=None):
        app, box = build_app(state, chat_client=chat_client)
        thread, base_url = start_server(app)
        ctx = browser.new_context()
        page = ctx.new_page()
        # Seed only the auth token so the sign-in dialog never fires.
        page.add_init_script(
            "localStorage.setItem('vessel.token', 'a-test-token-of-sufficient-length');"
        )
        started.append((thread, ctx, page))
        return page, base_url, box

    yield _make

    # Reset chat client slot so the next test doesn't inherit our fake.
    import vessel.pwa.routes as _r
    _r._set_chat_client_for_test(None)

    for thread, ctx, page in started:
        try:
            page.close()
            ctx.close()
        finally:
            thread.stop()


def _wait_for_render(page: Page) -> None:
    """Block until the SPA has finished a refresh cycle."""
    expect(page.locator('[data-testid="app-root"]'))\
        .to_have_attribute("data-loading", "false", timeout=5000)


def _open(page: Page, base_url: str) -> None:
    """Open the app at the only view it has: the calendar."""
    page.goto(f"{base_url}/pwa/")
    _wait_for_render(page)


# ---------------------------------------------------------------------------
# Calendar landing
# ---------------------------------------------------------------------------


def test_calendar_is_the_only_view(page_factory):
    """The home screen renders the calendar with `data-mode="calendar"`
    and a `calendar` day label — there is no toggle to flip away from.
    The footer carries refresh and sign-out only."""
    today = datetime.now().date()
    page, url, _ = page_factory(make_state_full(today))
    _open(page, url)

    root = page.locator('[data-testid="app-root"]')
    expect(root).to_have_attribute("data-mode", "calendar")
    expect(page.locator('[data-testid="day-label"]')).to_have_text("calendar")
    expect(page.locator('[data-testid="calendar-section"]')).to_be_visible()
    # The show-all / show-calendar toggles are gone — calendar is the
    # only view — but refresh and sign-out remain in the footer.
    expect(page.locator('[data-testid="show-all"]')).to_have_count(0)
    expect(page.locator('[data-testid="show-calendar"]')).to_have_count(0)
    expect(page.locator('[data-testid="refresh-footer"]')).to_be_visible()
    expect(page.locator('[data-testid="sign-out"]')).to_be_visible()


def test_footer_refresh_re_renders_calendar(page_factory):
    """Clicking the footer refresh button re-fetches and re-renders the
    calendar so a server-side state change becomes visible without
    reloading the page."""
    today = datetime.now().date()
    page, url, box = page_factory(make_state_with_one_open_event(today))
    _open(page, url)

    box["state"] = make_state_full(today)
    page.locator('[data-testid="refresh-footer"]').click()
    _wait_for_render(page)

    expect(
        page.locator('[data-testid="event-card-ev-today"]')
    ).to_be_visible()


def test_completed_and_skipped_events_are_filtered(page_factory):
    """`make_state_full` includes a completed event — it must NOT appear
    in the calendar list."""
    today = datetime.now().date()
    page, url, _ = page_factory(make_state_full(today))
    _open(page, url)
    expect(page.locator('[data-testid="event-card-ev-today"]')).to_be_visible()
    expect(page.locator('[data-testid="event-card-ev-tomorrow"]')).to_be_visible()
    expect(page.locator('[data-testid="event-card-ev-completed"]')).to_have_count(0)


def test_window_label_shows_total_open_items(page_factory):
    """The header window-label reads the count of open events rendered
    in the calendar view."""
    today = datetime.now().date()
    page, url, _ = page_factory(make_state_full(today))
    _open(page, url)
    # make_state_full has 2 open events (ev-today, ev-tomorrow) and 1 completed
    expect(page.locator('[data-testid="window-label"]')).to_have_text("2")


def test_calendar_empty_state_when_no_events(page_factory):
    """If state has no events, the calendar empty state surfaces a plain
    message instead of a blank screen."""
    state = StateData()
    page, url, _ = page_factory(state)
    _open(page, url)
    expect(page.locator('[data-testid="calendar-empty"]')).to_be_visible()
    expect(page.locator('[data-testid="calendar-empty"]')).to_contain_text(
        "nothing scheduled"
    )


# ---------------------------------------------------------------------------
# Calendar event cards
# ---------------------------------------------------------------------------


def test_calendar_event_card_has_only_cancel_change_button(page_factory):
    """Event cards expose a single quiet `cancel/change` button. It
    primes the chat input with `cancel/change "<title>" [id:<id>]: `
    instead of opening a dedicated skip-reason dialog — every
    change-intent goes through the chat assistant now."""
    today = datetime.now().date()
    page, url, _ = page_factory(make_state_with_calendar_only(today))
    _open(page, url)

    cancel_btn = page.locator('[data-testid="event-cancel-ev-gym"]')
    expect(cancel_btn).to_be_visible()
    expect(cancel_btn).to_have_text("cancel/change")
    expect(page.locator('[data-testid="event-done-ev-gym"]')).to_have_count(0)
    expect(page.locator('[data-testid="event-change-ev-gym"]')).to_have_count(0)

    posts: list[str] = []
    page.on(
        "request",
        lambda r: posts.append(r.url)
        if "/events/" in r.url and "/skip" in r.url and r.method == "POST"
        else None,
    )

    cancel_btn.click()
    # No skip dialog opens.
    expect(page.locator('[data-testid="skip-dialog"]')).not_to_be_visible()
    chat_input = page.locator('[data-testid="chat-input"]')
    expect(chat_input).to_have_value(
        'cancel/change "Go to the gym" [id:ev-gym]: '
    )
    assert page.evaluate(
        "document.activeElement === document.getElementById('chat-input')"
    )
    page.wait_for_timeout(150)
    assert posts == [], f"unexpected /skip POSTs: {posts}"


def test_completed_and_skipped_calendar_events_are_filtered(page_factory):
    """Calendar mode hides events the user marked done or explicitly
    skipped — closed items leave the list."""
    today = datetime.now().date()
    base = datetime(2026, 4, 25, tzinfo=timezone.utc)
    future = lambda hours: datetime.now(timezone.utc) + timedelta(hours=hours)
    state = StateData(
        calendar=[
            CalendarEvent(id="ev-open", title="Open meeting",
                          description="", start=future(2), end=future(3)),
            CalendarEvent(id="ev-done", title="Already done",
                          description="", start=future(4), end=future(5),
                          completed_at=base),
            CalendarEvent(id="ev-skipped",
                          title="Already skipped",
                          description="", start=future(6), end=future(7),
                          skipped_at=base, skip_reason="not relevant"),
        ],
    )
    page, url, _ = page_factory(state)
    _open(page, url)

    expect(page.locator('[data-testid="event-card-ev-open"]')).to_be_visible()
    expect(page.locator('[data-testid="event-card-ev-done"]')).to_have_count(0)
    expect(
        page.locator('[data-testid="event-card-ev-skipped"]')
    ).to_have_count(0)
    expect(page.locator('[data-testid="window-label"]')).to_have_text("1")


def test_tap_event_card_opens_detail_dialog_with_editable_fields(page_factory):
    """Tap on an event card opens the detail/edit modal pre-populated
    with the event's fields. Saving PATCHes /api/calendar/{id}."""
    today = datetime.now().date()
    page, url, box = page_factory(make_state_with_calendar_only(today))
    _open(page, url)

    page.locator('[data-testid="event-card-ev-gym"]').click()

    expect(page.locator('[data-testid="detail-dialog"]')).to_be_visible()
    expect(page.locator('[data-testid="detail-title"]')).to_have_value(
        "Go to the gym"
    )
    expect(page.locator('[data-testid="detail-notes"]')).to_have_value(
        "cardio + lift"
    )

    page.locator('[data-testid="detail-title"]').fill("Lift heavy")
    with page.expect_response(
        lambda r: "/api/calendar/ev-gym" in r.url
        and r.request.method == "PATCH"
        and r.status == 200
    ):
        page.locator('[data-testid="detail-save"]').click()
    expect(page.locator('[data-testid="detail-dialog"]')).to_be_hidden()
    ev = next(e for e in box["state"].calendar if e.id == "ev-gym")
    assert ev.title == "Lift heavy"


def test_delete_event_from_detail_dialog_removes_it_from_calendar(page_factory):
    """Clicking the delete button in the detail dialog sends DELETE
    /api/calendar/{id} and removes the event from the calendar list.

    Regression: window.confirm() is suppressed by Chrome when called
    from within a <dialog> element — the confirm returned false silently
    and the DELETE never fired. The fix removes confirm() entirely."""
    today = datetime.now().date()
    page, url, box = page_factory(make_state_with_calendar_only(today))
    _open(page, url)

    # Event should be visible before deletion.
    expect(page.locator('[data-testid="event-card-ev-gym"]')).to_be_visible()

    # Open the detail dialog.
    page.locator('[data-testid="event-card-ev-gym"]').click()
    expect(page.locator('[data-testid="detail-dialog"]')).to_be_visible()

    # Two-step delete: first click arms the button, second fires the DELETE.
    delete_btn = page.locator('[data-testid="detail-delete"]')
    delete_btn.click()
    expect(delete_btn).to_have_text("confirm delete?")

    with page.expect_response(
        lambda r: "/api/calendar/ev-gym" in r.url
        and r.request.method == "DELETE"
        and r.status == 200
    ):
        delete_btn.click()

    # Dialog closes and event is gone from the list.
    expect(page.locator('[data-testid="detail-dialog"]')).to_be_hidden()
    expect(page.locator('[data-testid="event-card-ev-gym"]')).to_have_count(0)
    assert all(e.id != "ev-gym" for e in box["state"].calendar)


def test_calendar_event_detail_opens_as_big_card(browser):
    """Tap on an event in the list opens the detail dialog sized as a
    big card — full viewport width on a phone-sized viewport so it
    reads as 'open this event' rather than 'small popup'."""
    today = datetime.now().date()
    app, _ = build_app(make_state_with_calendar_only(today))
    thread, base_url = start_server(app)
    ctx = browser.new_context(viewport={"width": 430, "height": 932})
    page = ctx.new_page()
    page.add_init_script(
        "localStorage.setItem('vessel.token', 'a-test-token-of-sufficient-length');"
    )
    try:
        page.goto(f"{base_url}/pwa/")
        _wait_for_render(page)
        expect(page.locator('[data-testid="day-label"]')).to_have_text("calendar")
        page.locator('[data-testid="event-card-ev-gym"]').click()
        expect(page.locator('[data-testid="detail-dialog"]')).to_be_visible()
        sizes = page.evaluate(
            """() => {
                const d = document.getElementById('detail-dialog');
                const r = d.getBoundingClientRect();
                return {
                  width: r.width,
                  height: r.height,
                  viewportW: window.innerWidth,
                  viewportH: window.innerHeight,
                };
            }"""
        )
        assert sizes["width"] >= 0.95 * sizes["viewportW"], sizes
        assert sizes["height"] >= 0.6 * sizes["viewportH"], sizes
    finally:
        page.close()
        ctx.close()
        thread.stop()


def test_overlapping_calendar_events_render_side_by_side_with_red(page_factory):
    """Two events whose time ranges intersect render inside a flex
    `.overlap-cluster` row and pick up the red conflict styling."""
    today = datetime.now().date()
    base = datetime.combine(
        today + timedelta(days=1), datetime.min.time()
    ).replace(hour=10, tzinfo=timezone.utc)
    a = CalendarEvent(
        id="ev-a", title="Event A", description="",
        start=base, end=base + timedelta(hours=1),
    )
    b = CalendarEvent(
        id="ev-b", title="Event B", description="",
        start=base + timedelta(minutes=30), end=base + timedelta(hours=2),
    )
    state = StateData(calendar=[a, b])
    page, url, _ = page_factory(state)
    _open(page, url)

    cluster = page.locator('[data-testid="overlap-cluster-ev-a"]')
    expect(cluster).to_be_visible()
    assert cluster.locator('[data-testid="event-card-ev-a"]').count() == 1
    assert cluster.locator('[data-testid="event-card-ev-b"]').count() == 1

    layout = page.evaluate(
        """() => {
            const c = document.querySelector(
              '[data-testid="overlap-cluster-ev-a"]'
            );
            const card = c.querySelector(
              '[data-testid="event-card-ev-a"]'
            );
            const cs = getComputedStyle(c);
            const cardCs = getComputedStyle(card);
            return {
              display: cs.display,
              borderTopColor: cardCs.borderTopColor,
            };
        }"""
    )
    assert layout["display"] == "flex"
    m = re.search(r"rgb\((\d+),\s*(\d+),\s*(\d+)", layout["borderTopColor"])
    assert m, layout
    r, g, b = (int(x) for x in m.groups())
    assert r > 100 and g < 60 and b < 60, (
        f"expected reddish border on overlap card, got {layout}"
    )


def test_arrive_by_renders_on_event_card_when_set(page_factory):
    """If `arrive_by` is set on an event, the meta line shows it
    alongside the start/end range."""
    today = datetime.now().date()
    base_dt = datetime.combine(
        today + timedelta(days=1), datetime.min.time()
    ).replace(hour=10, tzinfo=timezone.utc)
    ev = CalendarEvent(
        id="ev-doc", title="Doctor", description="",
        start=base_dt, end=base_dt + timedelta(minutes=30),
        arrive_by=base_dt - timedelta(minutes=15),
    )
    state = StateData(calendar=[ev])
    page, url, _ = page_factory(state)
    _open(page, url)

    meta = page.locator(
        '[data-testid="event-card-ev-doc"] .task-meta'
    )
    expect(meta).to_be_visible()
    expect(meta).to_contain_text("arrive by")


# ---------------------------------------------------------------------------
# Card color identity (light blue events, red conflicts)
# ---------------------------------------------------------------------------


def test_event_cards_render_with_light_blue_palette(page_factory):
    """`data-kind="event"` cards get the light-blue calendar palette —
    a striking blue background that visually separates events from
    tasks. Asserts on computed style so a class rename doesn't slip
    by without a corresponding CSS update."""
    today = datetime.now().date()
    page, url, _ = page_factory(make_state_with_calendar_only(today))
    _open(page, url)

    bg = page.evaluate(
        """() => getComputedStyle(
            document.querySelector('[data-testid="event-card-ev-gym"]')
        ).backgroundColor"""
    )
    m = re.search(r"rgb\((\d+),\s*(\d+),\s*(\d+)", bg)
    assert m, bg
    r, g, b = (int(x) for x in m.groups())
    # Light blue means: blue channel dominant, all channels are light
    # (well above 128). The current palette uses #dbeafe ≈ rgb(219,234,254).
    assert b > 200 and g > 200 and r > 200 and b >= r, (
        f"expected light-blue event card, got rgb({r},{g},{b})"
    )


# ---------------------------------------------------------------------------
# Text selection (long-press / drag to copy)
# ---------------------------------------------------------------------------


def _select_element_text(page: Page, selector: str) -> str:
    """Highlight every text node inside `selector` via the Selection API
    and return what `window.getSelection().toString()` reads back."""
    return page.evaluate(
        """(sel) => {
            const el = document.querySelector(sel);
            if (!el) return null;
            const range = document.createRange();
            range.selectNodeContents(el);
            const s = window.getSelection();
            s.removeAllRanges();
            s.addRange(range);
            return s.toString();
        }""",
        selector,
    )


def _user_select_value(page: Page, selector: str) -> str:
    return page.evaluate(
        """(sel) => {
            const el = document.querySelector(sel);
            if (!el) return null;
            return getComputedStyle(el).userSelect ||
                   getComputedStyle(el).webkitUserSelect;
        }""",
        selector,
    )


def test_calendar_event_text_is_selectable_for_copy(page_factory):
    """Event titles and the inline description block are both
    selectable so the user can long-press / drag to copy them out."""
    today = datetime.now().date()
    page, url, _ = page_factory(make_state_with_calendar_only(today))
    _open(page, url)

    title_sel = '[data-testid="event-card-ev-gym"] .task-title'
    notes_sel = '[data-testid="event-notes-ev-gym"]'
    expect(page.locator(title_sel)).to_be_visible()
    assert _user_select_value(page, title_sel) == "text"
    selected_title = _select_element_text(page, title_sel)
    assert selected_title and "Go to the gym" in selected_title

    expect(page.locator(notes_sel)).to_be_visible()
    selected_notes = _select_element_text(page, notes_sel)
    assert selected_notes and "cardio + lift" in selected_notes


# ---------------------------------------------------------------------------
# Chat box (in-app /api/chat path)
# ---------------------------------------------------------------------------


def test_chat_input_is_present_and_focusable(page_factory):
    """The chat row sits above the footer, ready to receive a tap."""
    today = datetime.now().date()
    page, url, _ = page_factory(make_state_full(today))
    _open(page, url)
    expect(page.locator('[data-testid="chat-area"]')).to_be_visible()
    expect(page.locator('[data-testid="chat-input"]')).to_be_visible()
    expect(page.locator('[data-testid="chat-send"]')).to_be_visible()
    expect(page.locator('[data-testid="chat-question"]')).to_be_hidden()

    page.locator('[data-testid="chat-input"]').click()
    is_focused = page.evaluate(
        "document.activeElement === document.getElementById('chat-input')"
    )
    assert is_focused


def test_chat_applies_in_one_shot_and_renders_popup_cards(page_factory):
    """The chat assistant infers from context — no clarifying questions.
    One submit → tool calls land → state updates → input clears → the
    items the assistant created/changed render as popup cards above
    the chat input. The text-only reply bubble stays empty when
    something was applied (the popups speak for themselves)."""
    today = datetime.now().date()

    import json
    from dataclasses import dataclass, field

    @dataclass
    class _FnCall:
        name: str
        arguments: str

    @dataclass
    class _FakeToolCall:
        id: str
        function: _FnCall
        type: str = "function"

    @dataclass
    class _FakeMsg:
        content: str = ""
        tool_calls: list = field(default_factory=list)

    @dataclass
    class _FakeChoice:
        message: _FakeMsg

    @dataclass
    class _FakeResp:
        choices: list[_FakeChoice]

    # Add a calendar event so the popup card shows up.
    new_start = datetime.combine(
        today + timedelta(days=1), datetime.min.time()
    ).replace(hour=9, tzinfo=timezone.utc)
    queue = [
        _FakeMsg(
            content="",
            tool_calls=[
                _FakeToolCall(
                    id="call_0",
                    function=_FnCall(
                        name="add_calendar_event",
                        arguments=json.dumps({"fields": {
                            "title": "file taxes",
                            "start": new_start.isoformat(),
                            "end": (new_start + timedelta(hours=1)).isoformat(),
                        }}),
                    ),
                )
            ],
        ),
        _FakeMsg(content="added file taxes"),
    ]

    class _Client:
        def __init__(self):
            self.chat = self
            self.completions = self

        async def create(self, **_kwargs):
            msg = queue.pop(0) if queue else _FakeMsg(content="(done)")
            return _FakeResp(choices=[_FakeChoice(message=msg)])

    page, url, box = page_factory(
        make_state_full(today), chat_client=_Client()
    )
    _open(page, url)

    chat_input = page.locator('[data-testid="chat-input"]')
    chat_input.fill("remind me to file taxes tomorrow morning")
    page.locator('[data-testid="chat-send"]').click()

    # Popup stack becomes visible with a card for the new event.
    results = page.locator('[data-testid="chat-results"]')
    expect(results).to_be_visible()
    # The card for the added event should exist
    added_cards = page.locator('[data-testid^="chat-result-event-added-"]')
    expect(added_cards.first).to_be_visible()
    # Text reply bubble stays empty when something applied.
    expect(page.locator('[data-testid="chat-question"]')).to_be_hidden()
    assert chat_input.input_value() == ""
    # The new event should be in the state
    assert any(e.title == "file taxes" for e in box["state"].calendar)


def test_chat_handles_empty_input_silently(page_factory):
    """Submitting an empty (or whitespace-only) chat input must not fire
    a request."""
    today = datetime.now().date()
    page, url, _ = page_factory(make_state_full(today))
    _open(page, url)

    requests: list[str] = []
    page.on(
        "request",
        lambda r: requests.append(r.url) if "/api/chat" in r.url else None,
    )

    page.locator('[data-testid="chat-input"]').fill("   ")
    page.locator('[data-testid="chat-send"]').click()
    page.wait_for_timeout(200)
    assert requests == [], f"unexpected /api/chat call(s): {requests}"


# ---------------------------------------------------------------------------
# Button press feedback (sign-out is the only footer button left)
# ---------------------------------------------------------------------------


def test_button_press_feedback_shows_pressed_class(page_factory):
    """Pointer down → button picks up `pressed`; pointer up → it clears.
    Locks in the visible press signal that :active alone can't deliver
    to Playwright."""
    today = datetime.now().date()
    page, url, _ = page_factory(make_state_full(today))
    _open(page, url)

    btn = page.locator('[data-testid="sign-out"]')
    box = btn.bounding_box()
    assert box is not None
    cx = box["x"] + box["width"] / 2
    cy = box["y"] + box["height"] / 2

    page.mouse.move(cx, cy)
    page.mouse.down()
    expect(btn).to_have_class(re.compile(r"\bpressed\b"))
    page.mouse.up()
    expect(btn).not_to_have_class(re.compile(r"\bpressed\b"))
