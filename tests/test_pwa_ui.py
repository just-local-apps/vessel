"""Playwright UI tests for the Vessel PWA.

Each test boots a fresh in-memory FastAPI app on a random port, navigates
Chromium to /pwa/, and exercises one feature via stable data-testid hooks
defined in index.html / app.js.

Run only this file with:
    uv run --extra dev --extra ui-test pytest tests/test_pwa_ui.py -v

The PWA is a single-view app: the calendar is the home screen. Events
and open tasks are interleaved per day; past-due open tasks roll forward
to today. There is no focus mode, no all-tasks mode, no in-app refresh
button — only the calendar view, an always-visible chat row, and a
sign-out button in the footer.
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
    make_state_with_one_open_task,
    start_server,
    test_now_local,
)
from vessel.models import StateData
from vessel.models.enums import Cadence, ProjectStatus  # noqa: E402
from vessel.models.state import CalendarEvent, Project  # noqa: E402


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
        # No mode seeding — the app always renders the calendar view.
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
    page, url, box = page_factory(make_state_with_one_open_task(today))
    _open(page, url)

    box["state"] = make_state_full(today)
    page.locator('[data-testid="refresh-footer"]').click()
    _wait_for_render(page)

    expect(
        page.locator('[data-testid="task-card-t-today-anytime"]')
    ).to_be_visible()


def test_calendar_interleaves_tasks_and_events_per_day(page_factory):
    """Tasks and events share day groups. An event for tomorrow plus a
    task due tomorrow both land under tomorrow's day group; an open
    task due today lands under today's group."""
    today = datetime.now().date()
    project = Project(
        id="p1", name="P", status=ProjectStatus.active, tracked=True,
        cadence=Cadence.daily,
        last_touched=datetime(2026, 4, 25, tzinfo=timezone.utc),
    )
    base = datetime(2026, 4, 25, tzinfo=timezone.utc)
    tomorrow = today + timedelta(days=1)
    from vessel.models.state import Task as _Task
    from vessel.models.enums import Tier as _T, TimeWindow as _W

    ev_tomorrow = CalendarEvent(
        id="ev-meeting", project_id="p1", title="Standup", description="",
        start=datetime.combine(tomorrow, datetime.min.time()).replace(
            hour=10, tzinfo=timezone.utc
        ),
        end=datetime.combine(tomorrow, datetime.min.time()).replace(
            hour=11, tzinfo=timezone.utc
        ),
    )
    task_today = _Task(
        id="t-today", project_id="p1", title="Today task",
        time_window=_W.anytime, tier=_T.must_today,
        due_date=today, estimated_minutes=15, created_at=base,
    )
    task_tomorrow = _Task(
        id="t-tomorrow", project_id="p1", title="Tomorrow task",
        time_window=_W.workday, tier=_T.flex,
        due_date=tomorrow, estimated_minutes=15, created_at=base,
    )
    state = StateData(
        projects=[project],
        tasks=[task_today, task_tomorrow],
        calendar=[ev_tomorrow],
    )
    page, url, _ = page_factory(state)
    _open(page, url)

    today_group = page.locator(f'[data-testid="calendar-day-{today.isoformat()}"]')
    tomorrow_group = page.locator(
        f'[data-testid="calendar-day-{tomorrow.isoformat()}"]'
    )
    expect(today_group.locator('[data-testid="task-card-t-today"]')).to_have_count(1)
    expect(tomorrow_group.locator('[data-testid="task-card-t-tomorrow"]')).to_have_count(1)
    expect(tomorrow_group.locator('[data-testid="event-card-ev-meeting"]')).to_have_count(1)


def test_past_due_open_tasks_roll_forward_to_today(page_factory):
    """An open task whose due_date is yesterday must surface under
    today's group — the fluid worklist behavior the user asked for.
    Server state is NOT mutated; this is a client-side projection."""
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    project = Project(
        id="p1", name="P", status=ProjectStatus.active, tracked=True,
        cadence=Cadence.daily,
        last_touched=datetime(2026, 4, 25, tzinfo=timezone.utc),
    )
    base = datetime(2026, 4, 25, tzinfo=timezone.utc)
    from vessel.models.state import Task as _Task
    from vessel.models.enums import Tier as _T, TimeWindow as _W

    stale = _Task(
        id="t-stale", project_id="p1", title="Yesterday's leftover",
        time_window=_W.anytime, tier=_T.must_today,
        due_date=yesterday, estimated_minutes=15, created_at=base,
    )
    state = StateData(projects=[project], tasks=[stale])
    page, url, box = page_factory(state)
    _open(page, url)

    today_group = page.locator(f'[data-testid="calendar-day-{today.isoformat()}"]')
    expect(today_group.locator('[data-testid="task-card-t-stale"]')).to_have_count(1)
    # Server state still says yesterday — no rewrite.
    server_task = next(t for t in box["state"].tasks if t.id == "t-stale")
    assert server_task.due_date == yesterday


def test_completed_and_skipped_tasks_are_filtered(page_factory):
    """`make_state_full` includes a completed task — it must NOT appear
    in the calendar list."""
    today = datetime.now().date()
    page, url, _ = page_factory(make_state_full(today))
    _open(page, url)
    expect(page.locator('[data-testid="task-card-t-today-anytime"]')).to_be_visible()
    expect(page.locator('[data-testid="task-card-t-tomorrow"]')).to_be_visible()
    expect(page.locator('[data-testid="task-card-t-completed"]')).to_have_count(0)


def test_window_label_shows_total_open_items(page_factory):
    """The header window-label reads the count of open events + open
    tasks rendered in the calendar view."""
    today = datetime.now().date()
    page, url, _ = page_factory(make_state_full(today))
    _open(page, url)
    # make_state_full has 2 open tasks and 0 events → "2".
    expect(page.locator('[data-testid="window-label"]')).to_have_text("2")


def test_calendar_empty_state_when_no_events_or_tasks(page_factory):
    """If state has neither events nor open tasks, the calendar empty
    state surfaces a plain message instead of a blank screen."""
    today = datetime.now().date()
    project = Project(
        id="p1", name="P", status=ProjectStatus.active, tracked=True,
        cadence=Cadence.daily,
        last_touched=datetime(2026, 4, 25, tzinfo=timezone.utc),
    )
    state = StateData(projects=[project])
    page, url, _ = page_factory(state)
    _open(page, url)
    expect(page.locator('[data-testid="calendar-empty"]')).to_be_visible()
    expect(page.locator('[data-testid="calendar-empty"]')).to_contain_text(
        "nothing scheduled"
    )


# ---------------------------------------------------------------------------
# Per-task action buttons (done / change / cancel)
# ---------------------------------------------------------------------------


def test_done_button_completes_task_via_api(page_factory):
    """Tapping `done` on a task card fires the complete endpoint and the
    card disappears from the calendar list."""
    today = datetime.now().date()
    page, url, _ = page_factory(make_state_full(today))
    _open(page, url)

    expect(
        page.locator('[data-testid="task-card-t-today-anytime"]')
    ).to_be_visible()
    expect(
        page.locator('[data-testid="task-done-t-today-anytime"]')
    ).to_be_visible()

    with page.expect_response(
        lambda r: r.url.endswith("/api/tasks/t-today-anytime/complete")
        and r.request.method == "POST"
        and r.status == 200
    ):
        page.locator('[data-testid="task-done-t-today-anytime"]').click()
    _wait_for_render(page)

    expect(
        page.locator('[data-testid="task-card-t-today-anytime"]')
    ).to_have_count(0)


def test_cancel_button_prefills_chat_input(page_factory):
    """The task `cancel` button no longer pops a dedicated skip dialog
    — it primes the chat input with
    `cancel/change "<title>" [id:<id>]: ` and focuses it so the user
    types why and submits through the same chat path that handles
    every other CRUD intent. The id is included so the chat assistant
    can resolve the entity unambiguously (titles aren't unique)."""
    today = datetime.now().date()
    page, url, _ = page_factory(make_state_full(today))
    _open(page, url)

    posts: list[str] = []
    page.on(
        "request",
        lambda r: posts.append(r.url) if "/skip" in r.url and r.method == "POST" else None,
    )

    page.locator('[data-testid="task-cancel-t-tomorrow"]').click()
    # No skip dialog appears any more.
    expect(page.locator('[data-testid="skip-dialog"]')).not_to_be_visible()
    # Chat input is pre-filled with the change-request prefix and
    # focused, ready for the user to type the reason.
    chat_input = page.locator('[data-testid="chat-input"]')
    expect(chat_input).to_have_value(
        'cancel/change "tomorrow task" [id:t-tomorrow]: '
    )
    assert page.evaluate(
        "document.activeElement === document.getElementById('chat-input')"
    )
    # No /skip POST should have fired — the legacy endpoint is no
    # longer used by the UI.
    page.wait_for_timeout(150)
    assert posts == [], f"unexpected skip POSTs: {posts}"


def test_change_button_opens_detail_dialog(page_factory):
    """The `change` button opens the detail card pre-filled with the
    task's fields so the user can edit and save."""
    today = datetime.now().date()
    page, url, _ = page_factory(make_state_full(today))
    _open(page, url)

    page.locator('[data-testid="task-change-t-tomorrow"]').click()
    expect(page.locator('[data-testid="detail-dialog"]')).to_be_visible()
    expect(page.locator('[data-testid="detail-title"]')).to_have_value(
        "tomorrow task"
    )


# ---------------------------------------------------------------------------
# Calendar event cards
# ---------------------------------------------------------------------------


def test_calendar_event_card_has_only_cancel_change_button(page_factory):
    """Event cards expose a single quiet `cancel/change` button. It
    primes the chat input with `cancel/change "<title>" [id:<id>]: `
    instead of opening a dedicated skip-reason dialog — every
    change-intent goes through the chat assistant now. Done + change
    buttons are not rendered on event cards."""
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


def test_completed_and_skipped_events_are_filtered(page_factory):
    """Calendar mode hides events the user marked done or explicitly
    skipped — closed items leave the list."""
    today = datetime.now().date()
    base = datetime(2026, 4, 25, tzinfo=timezone.utc)
    project = Project(
        id="p1", name="P", status=ProjectStatus.active, tracked=True,
        cadence=Cadence.event_driven, last_touched=base,
    )
    future = lambda hours: datetime.now(timezone.utc) + timedelta(hours=hours)
    state = StateData(
        projects=[project],
        calendar=[
            CalendarEvent(id="ev-open", project_id="p1", title="Open meeting",
                          description="", start=future(2), end=future(3)),
            CalendarEvent(id="ev-done", project_id="p1", title="Already done",
                          description="", start=future(4), end=future(5),
                          completed_at=base),
            CalendarEvent(id="ev-skipped", project_id="p1",
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
    project = Project(
        id="p1", name="Demo", status=ProjectStatus.active, tracked=True,
        cadence=Cadence.event_driven,
        last_touched=datetime(2026, 4, 25, tzinfo=timezone.utc),
    )
    base = datetime.combine(
        today + timedelta(days=1), datetime.min.time()
    ).replace(hour=10, tzinfo=timezone.utc)
    a = CalendarEvent(
        id="ev-a", project_id="p1", title="Event A", description="",
        start=base, end=base + timedelta(hours=1),
    )
    b = CalendarEvent(
        id="ev-b", project_id="p1", title="Event B", description="",
        start=base + timedelta(minutes=30), end=base + timedelta(hours=2),
    )
    state = StateData(projects=[project], calendar=[a, b])
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
    project = Project(
        id="p1", name="Demo", status=ProjectStatus.active, tracked=True,
        cadence=Cadence.event_driven,
        last_touched=datetime(2026, 4, 25, tzinfo=timezone.utc),
    )
    base_dt = datetime.combine(
        today + timedelta(days=1), datetime.min.time()
    ).replace(hour=10, tzinfo=timezone.utc)
    ev = CalendarEvent(
        id="ev-doc", project_id="p1", title="Doctor", description="",
        start=base_dt, end=base_dt + timedelta(minutes=30),
        arrive_by=base_dt - timedelta(minutes=15),
    )
    state = StateData(projects=[project], calendar=[ev])
    page, url, _ = page_factory(state)
    _open(page, url)

    meta = page.locator(
        '[data-testid="event-card-ev-doc"] .task-meta'
    )
    expect(meta).to_be_visible()
    expect(meta).to_contain_text("arrive by")


# ---------------------------------------------------------------------------
# Card color identity (light blue events, grey tasks, red conflicts)
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


def test_task_cards_render_with_grey_palette(page_factory):
    """`data-kind="task"` cards get the grey palette so the worklist
    items read as 'flexible / fit between events' next to the blue
    event blocks."""
    today = datetime.now().date()
    page, url, _ = page_factory(make_state_full(today))
    _open(page, url)

    bg = page.evaluate(
        """() => getComputedStyle(
            document.querySelector('[data-testid="task-card-t-today-anytime"]')
        ).backgroundColor"""
    )
    m = re.search(r"rgb\((\d+),\s*(\d+),\s*(\d+)", bg)
    assert m, bg
    r, g, b = (int(x) for x in m.groups())
    # Grey: all channels close to each other, in the mid-to-light range.
    spread = max(r, g, b) - min(r, g, b)
    assert spread < 20 and 150 < r < 230, (
        f"expected grey task card, got rgb({r},{g},{b})"
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


def test_task_card_text_is_selectable_for_copy(page_factory):
    """Task card titles opt into native text selection so a long-press
    or drag fills the clipboard selection with the chosen text."""
    today = datetime.now().date()
    page, url, _ = page_factory(make_state_full(today))
    _open(page, url)

    title_sel = '[data-testid="task-card-t-today-anytime"] .task-title'
    expect(page.locator(title_sel)).to_be_visible()
    assert _user_select_value(page, title_sel) == "text"
    selected = _select_element_text(page, title_sel)
    assert selected and "anytime today" in selected, selected


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
# Push API (still callable, even though no UI gesture drives it)
# ---------------------------------------------------------------------------


def test_push_api_still_moves_task_to_next_day(page_factory):
    """The /api/tasks/{id}/push endpoint still exists for any caller
    (CLI, scheduled jobs, future re-introduction) even though no UI
    surface drives it."""
    today = datetime.now().date()
    page, url, _ = page_factory(make_state_full(today))
    _open(page, url)

    payload = page.evaluate(
        """
        async () => {
          const t = localStorage.getItem('vessel.token');
          const r = await fetch('/api/tasks/t-today-anytime/push', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              Authorization: 'Bearer ' + t,
            },
            body: JSON.stringify({ days: 1 }),
          });
          return { status: r.status, body: await r.json() };
        }
        """
    )
    assert payload["status"] == 200, payload
    assert payload["body"]["ok"] is True


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

    # Add a task (project p1 already exists in make_state_full) so the
    # popup card shows up as a clickable task entry in the stack.
    queue = [
        _FakeMsg(
            content="",
            tool_calls=[
                _FakeToolCall(
                    id="call_0",
                    function=_FnCall(
                        name="add_task",
                        arguments=json.dumps({"fields": {
                            "id": "t-file-taxes",
                            "project_id": "p1",
                            "title": "file taxes",
                            "tier": "must_today",
                            "due_date": today.isoformat(),
                            "estimated_minutes": 30,
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
    chat_input.fill("remind me to file taxes today")
    page.locator('[data-testid="chat-send"]').click()

    # Popup stack becomes visible with a card for the new task.
    results = page.locator('[data-testid="chat-results"]')
    expect(results).to_be_visible()
    card = page.locator(
        '[data-testid="chat-result-task-added-t-file-taxes"]'
    )
    expect(card).to_be_visible()
    expect(card).to_contain_text("file taxes")
    # Text reply bubble stays empty when something applied.
    expect(page.locator('[data-testid="chat-question"]')).to_be_hidden()
    assert chat_input.input_value() == ""
    assert any(t.id == "t-file-taxes" for t in box["state"].tasks)


def test_chat_popup_click_opens_detail_dialog(page_factory):
    """Tapping a popup card opens the same detail/edit modal a card
    tap would. Closing the dialog leaves the popup in place so the
    user can revisit or dismiss it explicitly."""
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

    queue = [
        _FakeMsg(
            content="",
            tool_calls=[
                _FakeToolCall(
                    id="call_0",
                    function=_FnCall(
                        name="add_task",
                        arguments=json.dumps({"fields": {
                            "id": "t-popup-edit",
                            "project_id": "p1",
                            "title": "edit me",
                            "tier": "flex",
                            "due_date": today.isoformat(),
                            "estimated_minutes": 10,
                        }}),
                    ),
                )
            ],
        ),
        _FakeMsg(content="added edit me"),
    ]

    class _Client:
        def __init__(self):
            self.chat = self
            self.completions = self

        async def create(self, **_kwargs):
            msg = queue.pop(0) if queue else _FakeMsg(content="(done)")
            return _FakeResp(choices=[_FakeChoice(message=msg)])

    page, url, _ = page_factory(
        make_state_full(today), chat_client=_Client()
    )
    _open(page, url)

    page.locator('[data-testid="chat-input"]').fill("add edit me task")
    page.locator('[data-testid="chat-send"]').click()

    card = page.locator(
        '[data-testid="chat-result-task-added-t-popup-edit"]'
    )
    expect(card).to_be_visible()
    card.click()
    expect(page.locator('[data-testid="detail-dialog"]')).to_be_visible()
    expect(page.locator('[data-testid="detail-title"]')).to_have_value("edit me")

    # Close the modal — the popup must still be there so the user
    # can come back to it.
    page.locator('[data-testid="detail-cancel"]').click()
    expect(page.locator('[data-testid="detail-dialog"]')).to_be_hidden()
    expect(card).to_be_visible()


def test_chat_popup_dismiss_removes_only_that_card(page_factory):
    """Each popup carries an × button. Dismissing one card must NOT
    remove the others — the chat may produce several items and the
    user wants to walk through them one at a time."""
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

    queue = [
        _FakeMsg(
            content="",
            tool_calls=[
                _FakeToolCall(
                    id="c0",
                    function=_FnCall(
                        name="add_task",
                        arguments=json.dumps({"fields": {
                            "id": "t-a",
                            "project_id": "p1",
                            "title": "task A",
                            "tier": "flex",
                            "due_date": today.isoformat(),
                            "estimated_minutes": 10,
                        }}),
                    ),
                ),
                _FakeToolCall(
                    id="c1",
                    function=_FnCall(
                        name="add_task",
                        arguments=json.dumps({"fields": {
                            "id": "t-b",
                            "project_id": "p1",
                            "title": "task B",
                            "tier": "flex",
                            "due_date": today.isoformat(),
                            "estimated_minutes": 10,
                        }}),
                    ),
                ),
            ],
        ),
        _FakeMsg(content="added two"),
    ]

    class _Client:
        def __init__(self):
            self.chat = self
            self.completions = self

        async def create(self, **_kwargs):
            msg = queue.pop(0) if queue else _FakeMsg(content="(done)")
            return _FakeResp(choices=[_FakeChoice(message=msg)])

    page, url, _ = page_factory(
        make_state_full(today), chat_client=_Client()
    )
    _open(page, url)

    page.locator('[data-testid="chat-input"]').fill("add A and B")
    page.locator('[data-testid="chat-send"]').click()

    card_a = page.locator('[data-testid="chat-result-task-added-t-a"]')
    card_b = page.locator('[data-testid="chat-result-task-added-t-b"]')
    expect(card_a).to_be_visible()
    expect(card_b).to_be_visible()

    # Dismiss card A's × — B should still be visible.
    card_a.locator(".chat-result-close").click()
    expect(card_a).to_have_count(0)
    expect(card_b).to_be_visible()


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
