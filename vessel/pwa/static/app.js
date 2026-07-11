const TOKEN_KEY = "vessel.token";

// Single-view app: the calendar is the home screen. Events and open
// tasks are interleaved per day; tasks whose due_date is in the past
// roll forward to today (client-side display only — no state mutation).
const state = { mode: "calendar" };

function getToken() {
  return localStorage.getItem(TOKEN_KEY) || "";
}

function setToken(t) {
  if (t) localStorage.setItem(TOKEN_KEY, t);
  else localStorage.removeItem(TOKEN_KEY);
}

function _localISO() {
  // Local-time ISO with the user's UTC offset attached. We can't use
  // Date.prototype.toISOString() because that always emits UTC (Z),
  // which the server then turns into a UTC date — at late EDT hours
  // the UTC date is already tomorrow, so "today" resolves wrong.
  // Build "YYYY-MM-DDTHH:MM:SS+/-HH:MM" so the server sees the user's
  // wall clock AND offset.
  const d = new Date();
  const offMin = -d.getTimezoneOffset(); // east-of-UTC positive
  const sign = offMin >= 0 ? "+" : "-";
  const pad = (n) => String(n).padStart(2, "0");
  const offH = pad(Math.floor(Math.abs(offMin) / 60));
  const offM = pad(Math.abs(offMin) % 60);
  return (
    `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}` +
    `T${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}` +
    `${sign}${offH}:${offM}`
  );
}

async function api(path, opts = {}) {
  const token = getToken();
  // X-Vessel-Client-Now: every request carries the client's wall clock
  // (with timezone offset, NOT UTC) so the server and any LLM it
  // invokes reasons in the user's tz instead of Fly's. Without this an
  // EDT user sees "today" answered in LA time and the skip-assistant
  // sets due_date in the past.
  const headers = Object.assign(
    {
      "Content-Type": "application/json",
      "X-Vessel-Client-Now": _localISO(),
    },
    opts.headers || {},
    token ? { Authorization: `Bearer ${token}` } : {}
  );
  const resp = await fetch(path, { ...opts, headers });
  if (resp.status === 401) {
    promptForToken();
    throw new Error("unauthorized");
  }
  if (!resp.ok) throw new Error(`request failed: ${resp.status}`);
  return resp.json();
}

function promptForToken() {
  const dialog = document.getElementById("token-dialog");
  const input = document.getElementById("token-input");
  input.value = "";
  dialog.showModal();
}

document.getElementById("token-form").addEventListener("submit", () => {
  const value = document.getElementById("token-input").value.trim();
  if (value) {
    setToken(value);
    setTimeout(refresh, 50);
  }
});

function _formatHM(iso) {
  return new Date(iso).toLocaleTimeString(undefined, {
    hour: "numeric",
    minute: "2-digit",
  });
}

function _mapsHref(location) {
  // Google Maps deep link. The `?api=1&query=` form is the documented
  // universal URL — Maps app on iOS/Android intercepts it; desktop
  // opens maps.google.com. Whatever the user typed (street address,
  // place name, "home") goes through verbatim and Maps does the
  // geocoding. We send the FULL string here even though the card
  // displays a trimmed version: Maps geocodes city/state context,
  // and using the practice name + zip improves accuracy.
  return `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(location)}`;
}

function _shortLocation(location) {
  // Pick the street-address segment out of a verbose location string
  // for compact display on the card. The pattern Vessel sees from
  // Google Calendar invites is roughly:
  //   "<Practice/place name>, [<City>,] <number street>, <city state zip>"
  // Strip the practice prefix and zip suffix by finding the first
  // comma-separated chunk that begins with a digit (street number).
  // Falls back to the original string when no digit-led segment is
  // found — e.g., "home", "Slack huddle", "1234 Main" with no commas.
  if (!location) return location;
  const parts = location.split(",").map((s) => s.trim()).filter(Boolean);
  if (parts.length <= 1) return location;
  const street = parts.find((p) => /^\d/.test(p));
  return street || parts[0];
}

function _clusterByOverlap(events) {
  // Walk events sorted by start; an event joins the active cluster if
  // it starts before the cluster's running max-end. Otherwise it
  // starts a new cluster. Equal start==end (zero-length events) do
  // not count as overlapping with an event whose start matches.
  const out = [];
  for (const ev of events) {
    const start = new Date(ev.start).getTime();
    const last = out[out.length - 1];
    if (
      last &&
      start < last.reduce((m, e) => Math.max(m, new Date(e.end).getTime()), 0)
    ) {
      last.push(ev);
    } else {
      out.push([ev]);
    }
  }
  return out;
}

function renderEventCard(event) {
  // Calendar list cards: NO swipe gestures. The user wanted explicit
  // buttons here — swipes were too easy to fire by accident while
  // scrolling the day list. Tap the card to open the detail card;
  // the action buttons live in a row at the bottom.
  const card = document.createElement("div");
  card.className = "task-card";
  card.dataset.testid = `event-card-${event.id}`;
  card.dataset.eventId = event.id;
  card.dataset.kind = "event";
  const isCompleted = event.completed_at != null;
  const isSkipped = event.skipped_at != null;
  if (isCompleted || isSkipped) {
    card.classList.add("completed");
    card.dataset.completed = "true";
  }

  const title = document.createElement("div");
  title.className = "task-title";
  title.textContent = event.title;
  card.appendChild(title);

  const meta = document.createElement("div");
  meta.className = "task-meta";
  const closedTag = isCompleted ? "done" : isSkipped ? "skipped" : "";
  // Build the meta line as DOM nodes so the location can be a link
  // (opens Google Maps) instead of a non-clickable string. Everything
  // else stays text.
  const sep = () => {
    const s = document.createElement("span");
    s.className = "task-meta-sep";
    s.textContent = " · ";
    return s;
  };
  const timeSpan = document.createElement("span");
  timeSpan.textContent = `${_formatHM(event.start)} – ${_formatHM(event.end)}`;
  meta.appendChild(timeSpan);
  if (event.arrive_by) {
    meta.appendChild(sep());
    const ab = document.createElement("span");
    ab.textContent = `arrive by ${_formatHM(event.arrive_by)}`;
    meta.appendChild(ab);
  }
  if (event.location) {
    meta.appendChild(sep());
    const loc = document.createElement("a");
    loc.className = "task-meta-location";
    loc.dataset.testid = `event-location-${event.id}`;
    loc.href = _mapsHref(event.location);
    loc.target = "_blank";
    loc.rel = "noopener noreferrer";
    // Display the trimmed street-only string; the link still uses
    // the full address for accurate geocoding. The full address is
    // available in the detail dialog if the user needs to copy it.
    loc.textContent = _shortLocation(event.location);
    loc.title = event.location;
    loc.addEventListener("pointerdown", (e) => e.stopPropagation());
    loc.addEventListener("click", (e) => e.stopPropagation());
    meta.appendChild(loc);
  }
  if (closedTag) {
    meta.appendChild(sep());
    const tag = document.createElement("span");
    tag.textContent = closedTag;
    meta.appendChild(tag);
  }
  card.appendChild(meta);

  if (event.description) {
    const notes = document.createElement("div");
    notes.className = "task-notes";
    notes.dataset.testid = `event-notes-${event.id}`;
    notes.textContent = event.description;
    card.appendChild(notes);
  }

  if (event.url) {
    const link = document.createElement("a");
    link.className = "task-url";
    link.href = event.url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = event.url;
    link.addEventListener("pointerdown", (e) => e.stopPropagation());
    card.appendChild(link);
  }

  attachCardTap(card, () => openDetailDialog("event", event));

  if (isCompleted || isSkipped) return card;

  // Action row: a single, quiet `cancel/change` button. The user
  // asked for "no done either" — calendar events naturally pass once
  // their end time elapses, so an explicit "done" is noise. The
  // button no longer pops a dedicated skip-reason dialog; instead it
  // primes the chat input with `cancel/change "<title>" [id:<id>]: `
  // and focuses it. The user types what they want (move to next week,
  // skip, etc.) and submits — the chat assistant decides whether to
  // delete, reschedule, or update. One CRUD path for every "I want
  // to change this" intent. Results surface as the same popup stack
  // the chat box renders for ordinary submissions.
  const actions = document.createElement("div");
  actions.className = "card-actions";
  actions.appendChild(
    _actionButton({
      label: "cancel/change",
      kind: "ghost",
      testid: `event-cancel-${event.id}`,
      handler: () => focusChatWithChangeRequest("event", event.id, event.title),
    })
  );
  card.appendChild(actions);
  return card;
}

function _actionButton({ label, kind, testid, handler }) {
  // Small factory used by both event and task list rows. The card
  // already has a tap-to-open listener; we stop propagation so the
  // detail dialog doesn't open underneath the action the user just
  // chose. `pointerdown` cancellation also keeps the tap-vs-swipe
  // detector from misreading the press.
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className =
    kind === "ghost" ? "ghost pill" :
    kind === "danger" ? "danger pill" : "pill";
  btn.dataset.testid = testid;
  btn.textContent = label;
  btn.addEventListener("pointerdown", (e) => e.stopPropagation());
  btn.addEventListener("click", async (e) => {
    e.stopPropagation();
    await withBusy(btn, async () => {
      try {
        await handler();
      } catch (err) {
        console.warn(`${label} failed`, err);
      }
    });
  });
  return btn;
}

// ---------- Undo toast ----------
//
// After a right- or left-swipe, a fixed-position toast pinned to thumb
// level (above the chat row + footer at the bottom of the viewport)
// shows "<message> · undo". Anchored at the bottom because the previous
// "morph the card slot into an undo row" placement put the button at
// the top of the screen on focus mode — too far for a one-handed thumb
// to reach. 1-second window — long enough to register an "oh wait"
// reflex tap, short enough not to linger after the user has moved on.
// The user explicitly asked for ~1s; the Material Snackbar 4s default
// felt sticky in the swipe-heavy flow.
const UNDO_TIMEOUT_MS = 1000;

let _undoToastTimer = null;
let _undoToastDismissed = true;

function _hideUndoToast() {
  const toast = document.getElementById("undo-toast");
  if (!toast) return;
  toast.classList.remove("visible");
  toast.hidden = true;
}

function showUndoToast(message, undoFn) {
  const toast = document.getElementById("undo-toast");
  if (!toast) return;
  // Replace any in-flight undo first so the latest swipe owns the toast.
  if (_undoToastTimer) clearTimeout(_undoToastTimer);
  _undoToastDismissed = false;

  const msg = toast.querySelector('[data-testid="undo-message"]');
  const button = toast.querySelector('[data-testid="undo-button"]');
  msg.textContent = message;
  toast.hidden = false;
  // Force reflow so the .visible transition fires reliably (display:
  // none → block in the same frame otherwise skips the animation).
  // eslint-disable-next-line no-unused-expressions
  toast.offsetHeight;
  toast.classList.add("visible");

  const dismiss = () => {
    if (_undoToastDismissed) return;
    _undoToastDismissed = true;
    clearTimeout(_undoToastTimer);
    _undoToastTimer = null;
    _hideUndoToast();
    // No refresh here — the caller already refreshed when the swipe
    // committed, so the next card is already loaded behind the toast.
  };
  _undoToastTimer = setTimeout(dismiss, UNDO_TIMEOUT_MS);

  // Replace the click handler each call so the button fires the
  // current swipe's undo, not a stale closure.
  button.onclick = async () => {
    if (_undoToastDismissed) return;
    _undoToastDismissed = true;
    clearTimeout(_undoToastTimer);
    _undoToastTimer = null;
    _hideUndoToast();
    try {
      await undoFn();
    } catch (err) {
      console.warn("undo failed", err);
    }
    refresh();
  };
}

async function completeTask(taskId) {
  await api(`/api/tasks/${taskId}/complete`, { method: "POST" });
}

async function uncompleteTask(taskId) {
  return api(`/api/tasks/${taskId}/uncomplete`, { method: "POST" });
}

async function completeEvent(eventId) {
  await api(`/api/events/${eventId}/complete`, { method: "POST" });
}

async function uncompleteEvent(eventId) {
  return api(`/api/events/${eventId}/uncomplete`, { method: "POST" });
}

async function unskipTask(taskId) {
  return api(`/api/tasks/${taskId}/unskip`, { method: "POST" });
}

async function unskipEvent(eventId) {
  return api(`/api/events/${eventId}/unskip`, { method: "POST" });
}

async function skipEvent(eventId, eventTitle) {
  const reason = await promptSkipReason(eventId, eventTitle);
  if (!reason) return false;
  await api(`/api/events/${eventId}/skip`, {
    method: "POST",
    body: JSON.stringify({ reason }),
  });
  return true;
}

async function moveEvent(eventId, minutes) {
  await api(`/api/events/${eventId}/move`, {
    method: "POST",
    body: JSON.stringify({ minutes }),
  });
}

function promptMoveEvent(eventTitle) {
  // Resolves with one of: number (minutes to push), "skip", or null
  // (cancel). Replaces the modal's button handlers each call so the
  // result is always tied to the currently-shown card.
  return new Promise((resolve) => {
    const dialog = document.getElementById("move-dialog");
    const titleEl = document.getElementById("move-event-title");
    const cancel = document.getElementById("move-cancel");
    const skip = document.getElementById("move-skip");
    const buttons = Array.from(
      dialog.querySelectorAll("button[data-minutes]")
    );
    titleEl.textContent = `"${eventTitle}"`;

    function cleanup(result) {
      buttons.forEach((b) => b.removeEventListener("click", onTimeBtn));
      cancel.removeEventListener("click", onCancel);
      skip.removeEventListener("click", onSkip);
      dialog.close();
      resolve(result);
    }
    function onTimeBtn(e) {
      cleanup(parseInt(e.currentTarget.dataset.minutes, 10));
    }
    function onCancel() { cleanup(null); }
    function onSkip() { cleanup("skip"); }

    buttons.forEach((b) => b.addEventListener("click", onTimeBtn));
    cancel.addEventListener("click", onCancel);
    skip.addEventListener("click", onSkip);
    dialog.showModal();
  });
}

async function handleEventLeftSwipe(eventId, eventTitle) {
  // Returns: false (cancelled, restore card), "moved" (refresh, no
  // inline-undo), or true (skipped, attachCardSwipe will show the
  // standard inline-undo strip with unskipEvent as undo).
  const choice = await promptMoveEvent(eventTitle);
  if (choice === null) return false;
  if (choice === "skip") {
    return await skipEvent(eventId, eventTitle);
  }
  if (typeof choice === "number") {
    await moveEvent(eventId, choice);
    return "moved";
  }
  return false;
}

function promptSkipReason(taskId, taskTitle) {
  return new Promise((resolve) => {
    const dialog = document.getElementById("skip-dialog");
    const reason = document.getElementById("skip-reason");
    const titleEl = document.getElementById("skip-task-title");
    const cancel = document.getElementById("skip-cancel");
    const form = document.getElementById("skip-form");
    titleEl.textContent = `"${taskTitle}"`;
    reason.value = "";
    function cleanup(submitted) {
      form.removeEventListener("submit", onSubmit);
      cancel.removeEventListener("click", onCancel);
      dialog.close();
      resolve(submitted ? reason.value.trim() : null);
    }
    function onSubmit(e) {
      // Native dialog form submission already closes the dialog; we just
      // need to capture the value before unmount.
      if (!reason.value.trim()) {
        e.preventDefault();
        return;
      }
      cleanup(true);
    }
    function onCancel() {
      cleanup(false);
    }
    form.addEventListener("submit", onSubmit);
    cancel.addEventListener("click", onCancel);
    dialog.showModal();
    setTimeout(() => reason.focus(), 30);
  });
}

async function skipTask(taskId, taskTitle) {
  const reason = await promptSkipReason(taskId, taskTitle);
  if (!reason) return false;
  await api(`/api/tasks/${taskId}/skip`, {
    method: "POST",
    body: JSON.stringify({ reason }),
  });
  return true;
}

function _hasTextSelection() {
  // Treat a non-empty active selection as "user is reading / copying" —
  // gestures should not commit. Without this, dragging horizontally to
  // select a long title would also fire a swipe action.
  const sel = window.getSelection && window.getSelection();
  return !!(sel && sel.toString().length > 0);
}

function attachCardTap(card, openFn) {
  // Tap-to-open: distinct from swipe. We listen for `click` (fires on
  // both touch and mouse after a no-movement release). Skip if a button
  // or link inside the card was the actual target, if the user is in
  // the middle of selecting text, or if a swipe just committed (the
  // swipe path sets `data-swiped` on the card before the click event
  // fires).
  if (!openFn) return;
  card.addEventListener("click", (e) => {
    if (e.target.closest("button, a, input, textarea")) return;
    if (card.dataset.swiped === "true") {
      card.dataset.swiped = "";
      return;
    }
    if (_hasTextSelection()) return;
    openFn();
  });
}

function attachCardSwipe(card, item) {
  // Two-finger-friendly horizontal swipe on a single card. `item` is
  // `{ id, title, onRight, onLeft }` — `onRight` returns void, `onLeft`
  // returns a boolean (false if the user cancelled the reason dialog,
  // in which case we restore the card). Right (dx > +THRESHOLD) commits
  // the right action, left commits the left action. Vertical drag falls
  // through to the page (pull-to-refresh, scrolling).
  const TH = 70;
  let startX = 0;
  let startY = 0;
  let startT = 0;
  let active = false;
  let committed = null;

  function reset() {
    card.classList.remove("swiping", "swipe-left", "swipe-right");
    card.style.transform = "";
    card.style.transition = "";
    active = false;
    committed = null;
  }

  card.addEventListener(
    "touchstart",
    (e) => {
      if (e.touches.length !== 1) return;
      if (e.target.closest("button, a, input")) return;
      active = true;
      committed = null;
      startX = e.touches[0].clientX;
      startY = e.touches[0].clientY;
      startT = Date.now();
      card.style.transition = "";
    },
    { passive: true }
  );

  card.addEventListener(
    "touchmove",
    (e) => {
      if (!active) return;
      const dx = e.touches[0].clientX - startX;
      const dy = e.touches[0].clientY - startY;
      if (Math.abs(dy) > Math.abs(dx) && Math.abs(dy) > 10) {
        // Vertical scroll wins — bail.
        active = false;
        card.style.transform = "";
        return;
      }
      card.classList.add("swiping");
      card.style.transform = `translateX(${dx}px)`;
      card.classList.toggle("swipe-right", dx > TH / 2);
      card.classList.toggle("swipe-left", dx < -TH / 2);
    },
    { passive: true }
  );

  card.addEventListener(
    "touchend",
    async (e) => {
      if (!active) return;
      const t = e.changedTouches[0];
      const dx = t.clientX - startX;
      const dy = t.clientY - startY;
      const dt = Date.now() - startT;
      active = false;

      const horizontal =
        Math.abs(dx) > Math.abs(dy) && Math.abs(dx) >= TH && dt < 800;
      if (!horizontal) {
        reset();
        return;
      }
      if (_hasTextSelection()) {
        reset();
        return;
      }
      card.dataset.swiped = "true";
      // Slide the card off-screen in the swipe direction; once it's
      // gone we either snap it back as an inline-undo placeholder, or
      // (on cancel/error) bounce back to the original slot.
      const off = dx > 0 ? window.innerWidth : -window.innerWidth;
      card.style.transition = "transform 160ms ease-out, opacity 160ms ease-out";
      card.style.transform = `translateX(${off}px)`;
      card.style.opacity = "0";
      try {
        if (dx > 0) {
          await item.onRight();
          showUndoToast("marked done", item.undoRight);
          // Refresh now so the next card slides into the slot while
          // the toast is still on screen — without this the focus area
          // is blank for the duration of the toast (regression the
          // user hit when acking the break card).
          await refresh();
        } else {
          const result = await item.onLeft();
          if (result === false) {
            card.style.transition = "transform 200ms ease-out, opacity 200ms ease-out";
            card.style.transform = "";
            card.style.opacity = "";
            card.classList.remove("swiping", "swipe-left", "swipe-right");
          } else if (result === "moved") {
            await refresh();
          } else {
            showUndoToast("skipped", item.undoLeft);
            await refresh();
          }
        }
      } catch (err) {
        card.style.transition = "transform 200ms ease-out, opacity 200ms ease-out";
        card.style.transform = "";
        card.style.opacity = "";
        card.classList.remove("swiping", "swipe-left", "swipe-right");
      }
    },
    { passive: true }
  );

  card.addEventListener("touchcancel", reset);

  // Mouse-drag fallback for desktop / Playwright. Same semantics as the
  // touch handlers above — we just listen for mousedown/move/up. Pointer
  // events would unify the two, but mixing them with the existing touch
  // handlers leads to double-firing on iOS Safari, so we keep them split.
  let mouseDown = false;
  card.addEventListener("mousedown", (e) => {
    if (e.button !== 0) return;
    if (e.target.closest("button, a, input")) return;
    mouseDown = true;
    active = true;
    committed = null;
    startX = e.clientX;
    startY = e.clientY;
    startT = Date.now();
    card.style.transition = "";
    // Intentionally NOT calling e.preventDefault() — letting the
    // browser handle native text selection so the user can mouse-drag
    // to highlight a title or note and copy it. The swipe path below
    // checks `_hasTextSelection()` at mouseup and aborts if the drag
    // ended up producing a selection.
  });
  card.addEventListener("mousemove", (e) => {
    if (!mouseDown || !active) return;
    const dx = e.clientX - startX;
    const dy = e.clientY - startY;
    if (Math.abs(dy) > Math.abs(dx) && Math.abs(dy) > 10) {
      active = false;
      card.style.transform = "";
      return;
    }
    card.classList.add("swiping");
    card.style.transform = `translateX(${dx}px)`;
    card.classList.toggle("swipe-right", dx > TH / 2);
    card.classList.toggle("swipe-left", dx < -TH / 2);
  });
  async function endMouse(e) {
    if (!mouseDown) return;
    mouseDown = false;
    if (!active) {
      reset();
      return;
    }
    const dx = e.clientX - startX;
    const dy = e.clientY - startY;
    const dt = Date.now() - startT;
    active = false;
    const horizontal =
      Math.abs(dx) > Math.abs(dy) && Math.abs(dx) >= TH && dt < 1500;
    if (!horizontal) {
      reset();
      return;
    }
    // The drag could be a real swipe OR a desktop text selection that
    // happened to span > TH pixels horizontally. Selection wins —
    // committing a swipe under the user's copy gesture would be
    // surprising and lossy.
    if (_hasTextSelection()) {
      reset();
      return;
    }
    card.dataset.swiped = "true";
    const off = dx > 0 ? window.innerWidth : -window.innerWidth;
    card.style.transition = "transform 160ms ease-out, opacity 160ms ease-out";
    card.style.transform = `translateX(${off}px)`;
    card.style.opacity = "0";
    try {
      if (dx > 0) {
        await item.onRight();
        showUndoToast("marked done", item.undoRight);
        await refresh();
      } else {
        const result = await item.onLeft();
        if (result === false) {
          card.style.transition = "transform 200ms ease-out, opacity 200ms ease-out";
          card.style.transform = "";
          card.style.opacity = "";
          card.classList.remove("swiping", "swipe-left", "swipe-right");
        } else if (result === "moved") {
          await refresh();
        } else {
          showUndoToast("skipped", item.undoLeft);
          await refresh();
        }
      }
    } catch (err) {
      card.style.transition = "transform 200ms ease-out, opacity 200ms ease-out";
      card.style.transform = "";
      card.style.opacity = "";
      card.classList.remove("swiping", "swipe-left", "swipe-right");
    }
  }
  card.addEventListener("mouseup", endMouse);
  card.addEventListener("mouseleave", (e) => {
    if (mouseDown) endMouse(e);
  });
}

function renderTaskCard(task, { pushable = true } = {}) {
  const card = document.createElement("div");
  card.className = "task-card";
  card.dataset.testid = `task-card-${task.id}`;
  card.dataset.taskId = task.id;
  card.dataset.kind = "task";
  const isCompleted = task.completed_at != null;
  if (isCompleted) {
    card.classList.add("completed");
    card.dataset.completed = "true";
  }

  const title = document.createElement("div");
  title.className = "task-title";
  title.dataset.testid = `task-title-${task.id}`;
  title.textContent = task.title;
  card.appendChild(title);

  const meta = document.createElement("div");
  meta.className = "task-meta";
  const est = task.estimated_minutes ? `${task.estimated_minutes} min` : "";
  const completedTag = isCompleted ? "done" : "";
  meta.textContent = [task.tier, task.time_window, est, completedTag]
    .filter(Boolean)
    .join(" · ");
  card.appendChild(meta);

  if (task.notes) {
    const notes = document.createElement("div");
    notes.className = "task-notes";
    notes.dataset.testid = `task-notes-${task.id}`;
    notes.textContent = task.notes;
    card.appendChild(notes);
  }

  if (task.url) {
    const link = document.createElement("a");
    link.className = "task-url";
    link.href = task.url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = task.url;
    // Stop swipe gestures from being interpreted by the card when the
    // user is tapping the link itself.
    link.addEventListener("pointerdown", (e) => e.stopPropagation());
    card.appendChild(link);
  }

  attachCardTap(card, () => openDetailDialog("task", task));

  if (isCompleted) return card;

  // Same explicit-button row tasks get in the all-tasks list. The
  // focus card keeps its swipe gestures; the multi-row list views are
  // where accidental swipes during scroll were costing the user work.
  const actions = document.createElement("div");
  actions.className = "card-actions";
  actions.appendChild(
    _actionButton({
      label: "done",
      kind: "primary",
      testid: `task-done-${task.id}`,
      handler: async () => {
        await completeTask(task.id);
        showUndoToast("marked done", () => uncompleteTask(task.id));
        await refresh();
      },
    })
  );
  actions.appendChild(
    _actionButton({
      label: "change",
      kind: "ghost",
      testid: `task-change-${task.id}`,
      handler: () => openDetailDialog("task", task),
    })
  );
  actions.appendChild(
    _actionButton({
      label: "cancel",
      kind: "danger",
      testid: `task-cancel-${task.id}`,
      // Same as the event cancel/change path: hand the intent to the
      // chat assistant. The user types why they want to cancel/change
      // (or anything else) in the chat bar; the assistant decides
      // delete vs. reschedule vs. update. Results show as popup cards
      // above the chat input.
      handler: () => focusChatWithChangeRequest("task", task.id, task.title),
    })
  );
  card.appendChild(actions);
  return card;
}

function dayLabel(offset, isoDate) {
  if (offset === 0) return "today";
  if (offset === -1) return "yesterday";
  if (offset === 1) return "tomorrow";
  const d = new Date(isoDate + "T00:00:00");
  const opts = { weekday: "short", month: "short", day: "numeric" };
  return d.toLocaleDateString(undefined, opts).toLowerCase();
}

function clearTaskSections() {
  document.getElementById("calendar-events").innerHTML = "";
  document.getElementById("calendar-empty").hidden = true;
}

function _taskSortMinutes(task) {
  // Minutes-since-midnight for ordering inside a day. Tasks with a
  // `start_after` clock gate sort to that time of day; untimed tasks
  // sort to the end of the day so they appear AFTER all scheduled
  // events — "fit me in around your meetings".
  if (!task.start_after) return 24 * 60 + 1;
  const m = String(task.start_after).match(/^(\d{2}):(\d{2})/);
  if (!m) return 24 * 60 + 1;
  return parseInt(m[1], 10) * 60 + parseInt(m[2], 10);
}

function _eventSortMinutes(event) {
  const d = new Date(event.start);
  return d.getHours() * 60 + d.getMinutes();
}

async function renderCalendar() {
  // The home screen. Calendar events AND open tasks for every day
  // Vessel knows about, grouped by date, ordered chronologically inside
  // each day. Tasks slot in alongside events using their `start_after`
  // gate as a time-of-day; untimed tasks sink to the bottom of the day.
  //
  // Fluid rollover: an open task whose `due_date` is in the past gets
  // displayed under today's group instead. Pure client-side projection
  // — `due_date` on the server is unchanged, so completing the task
  // closes it cleanly without any data migration.
  const data = await api("/api/tasks/all");
  const todayIso = data.now.slice(0, 10);

  const events = (data.events || []).filter(
    (e) => e.completed_at == null && e.skipped_at == null
  );
  const openTasks = (data.tasks || []).filter(
    (t) => t.completed_at == null && t.skipped_at == null
  );

  document.getElementById("day-label").textContent = "calendar";
  document.getElementById("day-label").hidden = false;
  const win = document.getElementById("window-label");
  const totalItems = events.length + openTasks.length;
  win.textContent = `${totalItems}`;
  win.hidden = totalItems === 0;

  clearTaskSections();
  const container = document.getElementById("calendar-events");
  const empty = document.getElementById("calendar-empty");

  if (totalItems === 0) {
    empty.hidden = false;
    return;
  }

  // Bucket events by their start date.
  const eventsByDay = new Map();
  for (const ev of events) {
    const k = ev.start.slice(0, 10);
    if (!eventsByDay.has(k)) eventsByDay.set(k, []);
    eventsByDay.get(k).push(ev);
  }
  // Bucket tasks by effective date (rolled forward if past-due).
  const tasksByDay = new Map();
  for (const t of openTasks) {
    const effective = t.due_date < todayIso ? todayIso : t.due_date;
    if (!tasksByDay.has(effective)) tasksByDay.set(effective, []);
    tasksByDay.get(effective).push(t);
  }

  const allKeys = new Set([...eventsByDay.keys(), ...tasksByDay.keys()]);
  const sortedKeys = [...allKeys].sort();
  // Anchor on today (or the first day on or after today, falling back
  // to the latest known day) so the scroll lands on the current day.
  const anchorIso =
    sortedKeys.find((iso) => iso >= todayIso) || sortedKeys[sortedKeys.length - 1];

  for (const iso of sortedKeys) {
    const dayEvents = (eventsByDay.get(iso) || [])
      .slice()
      .sort((a, b) => a.start.localeCompare(b.start));
    const dayTasks = (tasksByDay.get(iso) || []).slice();
    const group = document.createElement("div");
    group.className = "all-day-group";
    group.dataset.testid = `calendar-day-${iso}`;
    if (iso === anchorIso) group.dataset.calendarAnchor = "true";
    const h = document.createElement("h3");
    const offset = Math.round(
      (new Date(iso + "T00:00:00") - new Date(todayIso + "T00:00:00")) /
        86400000
    );
    h.textContent = dayLabel(offset, iso);
    group.appendChild(h);

    // Cluster overlapping events first — conflict detection only makes
    // sense among events, which have explicit time ranges. Tasks have
    // no end time, so they never participate in clusters.
    const clusters = _clusterByOverlap(dayEvents);
    // Build a flat list of timeline items: each event-cluster gets the
    // start time of its earliest event; each task gets its start_after
    // minutes (or end-of-day for untimed). Sort, then render.
    const items = [];
    for (const cluster of clusters) {
      items.push({
        kind: "event-cluster",
        cluster,
        sort: _eventSortMinutes(cluster[0]),
      });
    }
    for (const t of dayTasks) {
      items.push({ kind: "task", task: t, sort: _taskSortMinutes(t) });
    }
    items.sort((a, b) => a.sort - b.sort);

    for (const item of items) {
      if (item.kind === "task") {
        group.appendChild(renderTaskCard(item.task));
      } else if (item.cluster.length === 1) {
        group.appendChild(renderEventCard(item.cluster[0]));
      } else {
        const wrapper = document.createElement("div");
        wrapper.className = "overlap-cluster";
        wrapper.dataset.testid = `overlap-cluster-${item.cluster[0].id}`;
        for (const ev of item.cluster) {
          wrapper.appendChild(renderEventCard(ev));
        }
        group.appendChild(wrapper);
      }
    }
    container.appendChild(group);
  }

  // Defer one frame so layout has settled before scrollIntoView fires.
  const anchor = container.querySelector('[data-calendar-anchor="true"]');
  if (anchor) {
    requestAnimationFrame(() => {
      anchor.scrollIntoView({ block: "start", behavior: "auto" });
    });
  }
}

function _formatTimeRange(startIso, endIso) {
  const fmt = (iso) =>
    new Date(iso).toLocaleTimeString(undefined, {
      hour: "numeric",
      minute: "2-digit",
    });
  return `${fmt(startIso)} – ${fmt(endIso)}`;
}

async function refresh() {
  const root = document.getElementById("app");
  root.dataset.loading = "true";
  try {
    await renderCalendar();
    root.dataset.mode = state.mode;
  } catch (err) {
    console.error(err);
    root.dataset.error = String(err && err.message ? err.message : err);
  } finally {
    root.dataset.loading = "false";
    root.dataset.lastRender = String(Date.now());
  }
}

async function checkForNewVersion() {
  // Ask the service worker to look for a new build. If one is waiting,
  // tell it to skipWaiting() — that swaps the controller, the
  // `controllerchange` listener below fires, and the page reloads with
  // the freshly-fetched JS / CSS. Without this, pull-to-refresh on iOS
  // would re-render the same old shell forever.
  if (!("serviceWorker" in navigator)) return false;
  try {
    const reg = await navigator.serviceWorker.getRegistration("/pwa/service-worker.js");
    if (!reg) return false;
    await reg.update();
    if (reg.waiting) {
      reg.waiting.postMessage("SKIP_WAITING");
      return true; // page will reload via controllerchange
    }
  } catch (err) {
    console.warn("sw update check failed", err);
  }
  return false;
}

async function manualRefresh(btn) {
  // Footer refresh button. First checks for a new service-worker build
  // so a tap actually ships the latest UI on iOS (otherwise the stale
  // shell sticks until the SW happens to update on its own); if a new
  // build is waiting, the page reloads via `controllerchange` and we
  // skip the data refresh because it'd be wasted work.
  if (btn) btn.classList.add("spinning");
  return withBusy(btn, async () => {
    try {
      const reloading = await checkForNewVersion();
      if (reloading) return;
      await refresh();
    } finally {
      if (btn) btn.classList.remove("spinning");
    }
  });
}

document.getElementById("refresh-btn").addEventListener("click", (e) => {
  manualRefresh(e.currentTarget);
});
document.getElementById("clear-token-btn").addEventListener("click", () => {
  setToken("");
  promptForToken();
});

// Pull-to-refresh on the scroll container only. Day navigation is in
// the footer — horizontal swipes on the document no longer change days,
// so per-card swipe (done / skip) can't be confused with a page swipe.
(function attachPullToRefresh() {
  const area = document.getElementById("swipe-area");
  const scroll = document.getElementById("task-scroll");
  const indicator = document.getElementById("pull-indicator");
  const PULL_THRESHOLD = 35;
  const PULL_MAX = 90;
  const PULL_TRIGGER_START = 6;

  let startX = 0;
  let startY = 0;
  let tracking = false;
  let pulling = false;
  let pullDy = 0;

  function isInteractive(el) {
    return !!(
      el &&
      el.closest &&
      el.closest("button, a, input, textarea, dialog, .task-card")
    );
  }

  scroll.addEventListener(
    "touchstart",
    (e) => {
      if (e.touches.length !== 1) return;
      if (isInteractive(e.target)) return;
      tracking = true;
      pulling = false;
      pullDy = 0;
      startX = e.touches[0].clientX;
      startY = e.touches[0].clientY;
    },
    { passive: true }
  );

  scroll.addEventListener(
    "touchmove",
    (e) => {
      if (!tracking) return;
      const dx = e.touches[0].clientX - startX;
      const dy = e.touches[0].clientY - startY;
      if (
        !pulling &&
        dy > PULL_TRIGGER_START &&
        Math.abs(dy) > Math.abs(dx) &&
        scroll.scrollTop <= 0
      ) {
        pulling = true;
        area.classList.add("dragging");
      }
      if (pulling) {
        pullDy = Math.min(PULL_MAX, dy * 0.8);
        area.style.transform = `translateY(${pullDy}px)`;
        indicator.classList.add("visible");
        indicator.classList.toggle("ready", pullDy >= PULL_THRESHOLD);
        indicator.textContent =
          pullDy >= PULL_THRESHOLD ? "release to refresh" : "pull to refresh";
      }
    },
    { passive: true }
  );

  function endPull(refreshed) {
    area.classList.remove("dragging");
    area.style.transform = "";
    if (refreshed) {
      indicator.classList.add("spinning");
      indicator.textContent = "refreshing";
      // Same dance as manualRefresh: check for a new SW build first so
      // pull-to-refresh actually ships the latest UI on iOS.
      (async () => {
        const reloading = await checkForNewVersion();
        if (reloading) return;
        await refresh();
      })().finally(() => {
        indicator.classList.remove("spinning", "visible");
      });
    } else {
      indicator.classList.remove("visible");
    }
  }

  scroll.addEventListener(
    "touchend",
    () => {
      if (!tracking) return;
      tracking = false;
      if (pulling) {
        const triggered = pullDy >= PULL_THRESHOLD;
        pulling = false;
        endPull(triggered);
      }
    },
    { passive: true }
  );

  scroll.addEventListener("touchcancel", () => {
    tracking = false;
    if (pulling) {
      pulling = false;
      endPull(false);
    }
  });
})();

// ---------- Press feedback for every button ----------
//
// Synthetic `pressed` class so a deliberate down→up sequence (mouse, touch,
// or Playwright) leaves a visible pressed-then-released signal. We can't
// rely on :active alone for testing — Playwright dispatches discrete
// mouse.down() / mouse.up() events and the browser's :active state isn't
// observable from outside the rendering pipeline.
(function attachPressFeedback() {
  document.addEventListener("pointerdown", (e) => {
    const btn = e.target.closest("button");
    if (btn) btn.classList.add("pressed");
  });
  function clearPressed() {
    document.querySelectorAll("button.pressed").forEach((b) =>
      b.classList.remove("pressed")
    );
  }
  document.addEventListener("pointerup", clearPressed);
  document.addEventListener("pointercancel", clearPressed);
  document.addEventListener("pointerleave", clearPressed);
})();

// ---------- Detail / edit dialog ----------
//
// Tap-to-open card view for both calendar events and tasks. Reads from
// the in-memory render data, lets the user edit any field, and PATCHes
// the matching CRUD endpoint on save. Delete is a button on the same
// modal — it hits DELETE on the same endpoint. Both flows finish with
// a refresh so the list re-renders from server state.
//
// We deliberately reuse /api/{tasks,calendar}/{id} (the same endpoints
// the chat assistant calls via tool-use) so the UI's edit path has no
// special-case server logic.
function _isoToLocalInput(iso) {
  // ISO 8601 → "YYYY-MM-DDTHH:MM" suitable for an <input type=datetime-local>.
  // datetime-local inputs are tz-naive — they show whatever wall-clock
  // string we hand them — so this projects the instant into the
  // browser's local TZ.
  if (!iso) return "";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return "";
  const pad = (n) => String(n).padStart(2, "0");
  return (
    `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}` +
    `T${pad(d.getHours())}:${pad(d.getMinutes())}`
  );
}

function _localInputToIso(value) {
  // datetime-local "YYYY-MM-DDTHH:MM" → ISO with the user's TZ offset.
  // Same shape as `_localISO()` so the server reads it as a wall-clock
  // moment in the user's TZ, never as UTC. Empty input → null so the
  // PATCH clears the field.
  if (!value) return null;
  const d = new Date(value);
  if (isNaN(d.getTime())) return null;
  const pad = (n) => String(n).padStart(2, "0");
  const offMin = -d.getTimezoneOffset();
  const sign = offMin >= 0 ? "+" : "-";
  const offH = pad(Math.floor(Math.abs(offMin) / 60));
  const offM = pad(Math.abs(offMin) % 60);
  return (
    `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}` +
    `T${pad(d.getHours())}:${pad(d.getMinutes())}:00${sign}${offH}:${offM}`
  );
}

function _syncDetailMapLink() {
  // Keep the "open in google maps" link beside the location input in
  // sync with whatever the user has typed. Hidden when the field is
  // empty so we don't show a link to a search for "".
  const input = document.getElementById("detail-location");
  const link = document.getElementById("detail-location-map");
  if (!input || !link) return;
  const value = input.value.trim();
  if (!value) {
    link.hidden = true;
    link.removeAttribute("href");
    return;
  }
  link.hidden = false;
  link.href = _mapsHref(value);
}

let _detailContext = null; // { kind: "event"|"task", id, original }

function openDetailDialog(kind, item) {
  const dialog = document.getElementById("detail-dialog");
  if (!dialog) return;
  _detailContext = { kind, id: item.id, original: item };

  const titleEl = document.getElementById("detail-title");
  const urlEl = document.getElementById("detail-url");
  const notesEl = document.getElementById("detail-notes");
  const timeFields = document.getElementById("detail-time-fields");
  const taskFields = document.getElementById("detail-task-fields");
  const errorEl = document.getElementById("detail-error");

  errorEl.hidden = true;
  errorEl.textContent = "";
  titleEl.value = item.title || "";
  urlEl.value = item.url || "";

  if (kind === "event") {
    timeFields.hidden = false;
    taskFields.hidden = true;
    document.getElementById("detail-start").value = _isoToLocalInput(item.start);
    document.getElementById("detail-end").value = _isoToLocalInput(item.end);
    document.getElementById("detail-arrive-by").value = _isoToLocalInput(
      item.arrive_by
    );
    const locInput = document.getElementById("detail-location");
    locInput.value = item.location || "";
    _syncDetailMapLink();
    locInput.oninput = _syncDetailMapLink;
    notesEl.value = item.description || "";
  } else {
    timeFields.hidden = true;
    taskFields.hidden = false;
    document.getElementById("detail-due-date").value = item.due_date || "";
    document.getElementById("detail-est").value =
      item.estimated_minutes != null ? item.estimated_minutes : "";
    document.getElementById("detail-start-after").value = item.start_after || "";
    notesEl.value = item.notes || "";
  }
  dialog.showModal();
}

(function attachDetailHandlers() {
  const dialog = document.getElementById("detail-dialog");
  if (!dialog) return;
  const form = document.getElementById("detail-form");
  const cancelBtn = document.getElementById("detail-cancel");
  const deleteBtn = document.getElementById("detail-delete");
  const errorEl = document.getElementById("detail-error");

  function showError(msg) {
    errorEl.textContent = msg;
    errorEl.hidden = false;
  }

  cancelBtn.addEventListener("click", () => {
    _detailContext = null;
    dialog.close();
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!_detailContext) return;
    const ctx = _detailContext;
    const title = document.getElementById("detail-title").value.trim();
    if (!title) {
      showError("title is required");
      return;
    }
    const url = document.getElementById("detail-url").value.trim() || null;
    const notesValue = document.getElementById("detail-notes").value;

    let endpoint, payload;
    if (ctx.kind === "event") {
      const start = _localInputToIso(
        document.getElementById("detail-start").value
      );
      const end = _localInputToIso(
        document.getElementById("detail-end").value
      );
      if (!start || !end) {
        showError("start and end are required");
        return;
      }
      payload = {
        title,
        start,
        end,
        arrive_by: _localInputToIso(
          document.getElementById("detail-arrive-by").value
        ),
        location: document.getElementById("detail-location").value.trim() || null,
        url,
        description: notesValue,
      };
      endpoint = `/api/calendar/${encodeURIComponent(ctx.id)}`;
    } else {
      const dueDate = document.getElementById("detail-due-date").value;
      if (!dueDate) {
        showError("due date is required");
        return;
      }
      const estRaw = document.getElementById("detail-est").value;
      const startAfter = document.getElementById("detail-start-after").value;
      payload = {
        title,
        due_date: dueDate,
        estimated_minutes: estRaw === "" ? null : parseInt(estRaw, 10),
        start_after: startAfter || null,
        url,
        notes: notesValue || null,
      };
      endpoint = `/api/tasks/${encodeURIComponent(ctx.id)}`;
    }

    try {
      await api(endpoint, {
        method: "PATCH",
        body: JSON.stringify(payload),
      });
      _detailContext = null;
      dialog.close();
      await refresh();
    } catch (err) {
      showError(`save failed: ${err.message || err}`);
    }
  });

  deleteBtn.addEventListener("click", async () => {
    if (!_detailContext) return;
    const ctx = _detailContext;
    const noun = ctx.kind === "event" ? "event" : "task";
    if (!confirm(`Delete this ${noun}? This can't be undone.`)) return;
    const endpoint =
      ctx.kind === "event"
        ? `/api/calendar/${encodeURIComponent(ctx.id)}`
        : `/api/tasks/${encodeURIComponent(ctx.id)}`;
    try {
      await api(endpoint, { method: "DELETE" });
      _detailContext = null;
      dialog.close();
      // Drop back to focus mode after a delete from the calendar/all
      // list — refreshing the same list with one fewer entry keeps the
      // user in context, but if they deleted the only item in view it
      // would render an empty section. Refresh in current mode is safer.
      await refresh();
    } catch (err) {
      showError(`delete failed: ${err.message || err}`);
    }
  });
})();

async function withBusy(btn, fn) {
  // Mark a button busy while an async action runs. Caller passes any
  // button reference (footer day-prev/next, show-all, refresh, etc.).
  // The class adds the spinner suffix and dims the button until the
  // promise settles — guaranteed even on error.
  if (!btn) return fn();
  btn.classList.add("busy");
  btn.disabled = true;
  try {
    return await fn();
  } finally {
    btn.classList.remove("busy");
    btn.disabled = false;
  }
}

// ---------- Chat box ----------
//
// Always-visible input above the footer. Submitting POSTs to /api/chat
// which runs the chat tool-loop assistant. The assistant's only job is
// CRUD on tasks / events / projects — never a general-purpose chatbot.
//
//   { applied: true,  diff: {...},  assistant: {summary, ...} }
//                                                  → state updated,
//                                                    refresh, show a
//                                                    popup card per
//                                                    added/changed item
//   { applied: false, diff: null,   assistant: {summary, ...} }
//                                                  → CRUD refusal —
//                                                    show summary as a
//                                                    one-line reply
//
// Cancel/change buttons on event and task cards funnel through the
// SAME path: they call `focusChatWithChangeRequest()` which pre-fills
// the input with `cancel/change "<title>" [id:<id>]: ` and focuses
// it. The user types the why, presses send, the chat assistant
// decides what to do. The bracketed id is the canonical reference —
// titles aren't unique (recurring events, duplicate task names) and
// the user can edit the title text before sending. The chat assistant
// reads the id to look up the entity, then falls back to the title
// only if the id is missing.

function focusChatWithChangeRequest(kind, id, title) {
  // kind is "event" | "task" — currently both share the same prefix;
  // kept as a parameter so we can specialize the prompt later without
  // touching call sites.
  const input = document.getElementById("chat-input");
  if (!input) return;
  const prefix = `cancel/change "${title}" [id:${id}]: `;
  input.value = prefix;
  input.focus();
  // Place the caret at the end so the user can immediately type the
  // reason without arrow-key navigation.
  try {
    input.setSelectionRange(prefix.length, prefix.length);
  } catch (_err) {
    /* setSelectionRange unsupported for some input types — ignore */
  }
}

function _chatResultsContainer() {
  return document.getElementById("chat-results");
}

function _clearChatResults() {
  const c = _chatResultsContainer();
  if (!c) return;
  c.innerHTML = "";
  c.hidden = true;
}

function _renderChatResultCard({ kind, action, item, title, meta }) {
  // Build one popup card. `kind` ∈ {"event", "task", "project", "info"}.
  // For event/task with an item dict, the card is clickable and opens
  // the detail dialog. For projects (no detail editor) or deletes
  // (item no longer exists), the card is non-clickable info-only.
  const card = document.createElement("div");
  card.className = "chat-result-card";
  card.dataset.testid = `chat-result-${kind}-${action}-${item ? item.id : "n"}`;
  const clickable = (kind === "event" || kind === "task") && action !== "deleted";
  card.dataset.clickable = clickable ? "true" : "false";

  const body = document.createElement("div");
  body.className = "chat-result-body";
  const titleEl = document.createElement("div");
  titleEl.className = "chat-result-title";
  titleEl.textContent = title;
  body.appendChild(titleEl);
  const metaEl = document.createElement("div");
  metaEl.className = "chat-result-meta";
  metaEl.textContent = meta;
  body.appendChild(metaEl);
  card.appendChild(body);

  const close = document.createElement("button");
  close.type = "button";
  close.className = "chat-result-close";
  close.setAttribute("aria-label", "dismiss");
  close.textContent = "×";
  close.addEventListener("pointerdown", (e) => e.stopPropagation());
  close.addEventListener("click", (e) => {
    e.stopPropagation();
    card.remove();
    const c = _chatResultsContainer();
    if (c && c.children.length === 0) c.hidden = true;
  });
  card.appendChild(close);

  if (clickable) {
    card.addEventListener("click", async () => {
      // Re-fetch the latest copy of the item before opening the modal
      // so edits done since the chat ran (or by the chat itself, in
      // case the diff snapshot drifted from current state after a
      // refresh) show through. Fall back to the diff snapshot if the
      // fetch fails or the item is gone.
      let fresh = item;
      try {
        const data = await api("/api/tasks/all");
        const list = kind === "event" ? data.events || [] : data.tasks || [];
        const found = list.find((x) => x.id === item.id);
        if (found) fresh = found;
      } catch (_err) {
        /* keep snapshot */
      }
      openDetailDialog(kind, fresh);
    });
  }
  return card;
}

function _eventMeta(ev) {
  try {
    return `${_formatHM(ev.start)} – ${_formatHM(ev.end)}`;
  } catch (_err) {
    return "event";
  }
}

function _taskMeta(t) {
  const due = t.due_date ? `due ${t.due_date}` : "";
  const after = t.start_after ? `after ${String(t.start_after).slice(0, 5)}` : "";
  return [due, after].filter(Boolean).join(" · ") || "task";
}

function renderChatResults(diff) {
  // Build the popup stack for whatever the chat assistant just did.
  // Order: events, tasks, projects; added before changed before
  // removed within each group so newly created items land at the top.
  const container = _chatResultsContainer();
  if (!container) return;
  container.innerHTML = "";

  const cards = [];
  const cal = (diff && diff.calendar) || { added: [], changed: [], removed: [] };
  const tasks = (diff && diff.tasks) || { added: [], changed: [], removed: [] };
  const projects = (diff && diff.projects) || { added: [], changed: [], removed: [] };

  for (const ev of cal.added) {
    cards.push(_renderChatResultCard({
      kind: "event", action: "added", item: ev,
      title: ev.title || "(untitled event)",
      meta: `new event · ${_eventMeta(ev)}`,
    }));
  }
  for (const ch of cal.changed) {
    const ev = ch.after;
    cards.push(_renderChatResultCard({
      kind: "event", action: "changed", item: ev,
      title: ev.title || "(untitled event)",
      meta: `updated event · ${_eventMeta(ev)}`,
    }));
  }
  for (const ev of cal.removed) {
    cards.push(_renderChatResultCard({
      kind: "event", action: "deleted", item: ev,
      title: ev.title || "(untitled event)",
      meta: "deleted event",
    }));
  }
  for (const t of tasks.added) {
    cards.push(_renderChatResultCard({
      kind: "task", action: "added", item: t,
      title: t.title || "(untitled task)",
      meta: `new task · ${_taskMeta(t)}`,
    }));
  }
  for (const ch of tasks.changed) {
    const t = ch.after;
    cards.push(_renderChatResultCard({
      kind: "task", action: "changed", item: t,
      title: t.title || "(untitled task)",
      meta: `updated task · ${_taskMeta(t)}`,
    }));
  }
  for (const t of tasks.removed) {
    cards.push(_renderChatResultCard({
      kind: "task", action: "deleted", item: t,
      title: t.title || "(untitled task)",
      meta: "deleted task",
    }));
  }
  for (const p of projects.added) {
    cards.push(_renderChatResultCard({
      kind: "project", action: "added", item: p,
      title: p.name || p.id, meta: "new project",
    }));
  }
  for (const ch of projects.changed) {
    const p = ch.after;
    cards.push(_renderChatResultCard({
      kind: "project", action: "changed", item: p,
      title: p.name || p.id, meta: "updated project",
    }));
  }
  for (const p of projects.removed) {
    cards.push(_renderChatResultCard({
      kind: "project", action: "deleted", item: p,
      title: p.name || p.id, meta: "deleted project",
    }));
  }

  if (cards.length === 0) {
    container.hidden = true;
    return;
  }
  for (const c of cards) container.appendChild(c);
  container.hidden = false;
}

(function attachChat() {
  const form = document.getElementById("chat-form");
  const input = document.getElementById("chat-input");
  const send = document.getElementById("chat-send");
  const reply = document.getElementById("chat-question");
  if (!form || !input || !send || !reply) return;

  function showReply(text) {
    reply.textContent = text;
    reply.hidden = !text;
    if (text) setTimeout(() => input.focus(), 30);
  }

  function clearReply() {
    reply.hidden = true;
    reply.textContent = "";
  }

  async function submit(text) {
    input.classList.add("busy");
    send.classList.add("busy");
    input.disabled = true;
    send.disabled = true;
    try {
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Vessel-Client-Now": _localISO(),
          Authorization: `Bearer ${getToken()}`,
        },
        body: JSON.stringify({ text }),
      });
      if (resp.status === 401) {
        promptForToken();
        return;
      }
      if (!resp.ok) {
        const detail = await resp.text();
        _clearChatResults();
        showReply(`error: ${detail.slice(0, 200)}`);
        return;
      }
      const data = await resp.json();
      const summary = (data.assistant && data.assistant.summary) || "";
      input.value = "";
      if (data.applied) {
        // CRUD happened: popups speak for the assistant. Drop any
        // prior text reply so we don't double-up the signal.
        clearReply();
        await refresh();
        renderChatResults(data.diff);
      } else {
        // No CRUD landed — either the assistant refused (general
        // chatter) or hit an error path. Surface the short text
        // reply and clear any stale popups.
        _clearChatResults();
        if (summary) showReply(summary);
        else clearReply();
      }
    } catch (err) {
      _clearChatResults();
      showReply(`error: ${err.message || err}`);
    } finally {
      input.classList.remove("busy");
      send.classList.remove("busy");
      input.disabled = false;
      send.disabled = false;
    }
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    submit(text);
  });
})();

if (!getToken()) {
  promptForToken();
} else {
  refresh();
}

if ("serviceWorker" in navigator) {
  let reloaded = false;
  navigator.serviceWorker.addEventListener("controllerchange", () => {
    if (reloaded) return;
    reloaded = true;
    location.reload();
  });
  navigator.serviceWorker
    .register("/pwa/service-worker.js")
    .then((reg) => {
      reg.update();
      document.addEventListener("visibilitychange", () => {
        if (document.visibilityState === "visible") reg.update();
      });
    })
    .catch((err) => console.warn("sw register failed", err));
}

setInterval(refresh, 60_000);
