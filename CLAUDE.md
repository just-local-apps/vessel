# Vessel — engineering rules

## State model invariant: derived fields are derived, not stored

The state model (`vessel/models/state.py`) is the contract between the
agent, the PWA, the scheduler, and the CRUD layer. Treat it as an API,
not a convenience.

**Rule: state fields are derived by default. Storing requires an ADR-style
justification with an explicit functional-dependency declaration.**

Concretely, when reviewing or writing changes to `vessel/models/state.py`:

1. For every new or changed field, ask: *can this be computed from
   existing fields?* If yes → use `@computed_field`, do not add a stored
   column. The bug we are preventing is **derived-state-as-canonical
   state**: `time_window` (a categorical bucket) was stored alongside
   `start_after` (a clock gate) and the two drifted, so the UI label and
   the scheduling logic disagreed.

2. Every new stored field needs a one-line FD comment in the model file:
   *"depends only on `id`"* (3NF). If it depends on another non-key
   field, it is not in 3NF — make it a `@computed_field`.

3. State-level invariants that span fields belong in `StateData`'s
   `@model_validator(mode="after")`. Add the check there, not in
   downstream callers. Examples already enforced: foreign-key references
   to `projects.id`, mutual exclusion of `completed_at` and `skipped_at`,
   `calendar.end >= calendar.start`.

4. There must be exactly one writer per field. If a field is set from
   more than one place (intake agent + CRUD + recurrence spawner), it is
   a derivation candidate — fold it into `@computed_field` and delete
   the writers.

5. Tripwire test: `tests/test_state.py::test_no_derived_columns` lists
   the fields that are intentionally derived. Adding any of those names
   back to `Task.model_fields` (or the equivalent on other models) fails
   CI. Extend the test when you remove a stored column.

## Default behavior, not optional

This is not advisory. The state model is read by every agent and every
HTTP route in the system, and any redundancy in it shows up as a UX bug
within a release. Pydantic's `@computed_field` makes derivation cheap;
there is no excuse to denormalize.

When in doubt, derive.

## No secrets in the repository

This is a public repo. Never commit secrets, credentials, or private data.

**Hard rules:**

1. **No real tokens or API keys** in any tracked file — not in source, not in
   tests, not in config, not in `.claude/settings.json` permission allowlists.
   Use environment variable references (`$VAR` or `os.getenv("VAR")`) only.

2. **No real connection strings** with embedded credentials
   (`postgresql://user:pass@host/db`). `.env.example` uses placeholders only.

3. **No personal data** — real names tied to private context, phone numbers,
   addresses, or appointment details must never appear in source files or test
   fixtures.

4. **`.claude/` is gitignored.** Do not add it back to tracking. Permission
   allowlists in `settings.json` must never contain inline token values.

5. **Test fixtures use fake values.** Any token, key, or credential in a test
   must be an obviously fake placeholder (`"test-token-of-sufficient-length"`,
   `"fake"`, generated in the test itself). Never copy a real value into a test.

**Before every commit:** if a value looks like a real secret — long random
string, matches a known key format (`sk-`, `eyJ`, `tgp_v1_`, base64 blob) —
stop and move it to `.env` instead.
