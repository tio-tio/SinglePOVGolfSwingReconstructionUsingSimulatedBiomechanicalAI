# Chat V2 Plan — session-aware coach, richer context, grounded posture coaching

> **STATUS 2026-08-01: IMPLEMENTED in code (6 commits on `aws-deploy-handoff`,
> 45ecb05..e5bbe63), all offline suites green (369 checks). NOT yet deployed.**
> Deploy checklist to go live:
> 1. Re-zip + update the **upload** and **jobs** Lambdas (each zip now bundles
>    `Scripts/session_meta.py` flat — see the READMEs' zip commands).
> 2. Grant the jobs role `s3:PutObject` on `03_outputs/*` and add the
>    **`POST /jobs`** route on API Gateway (PENDING_PERMISSIONS.md).
> 3. Rebuild + deploy the **chat image** (Dockerfile gained session/progress
>    modules + drill_cards.json) with new env: `JOBS_API_BASE` (same API origin),
>    `CHAT_LLM_TIMEOUT=18`, `CHAT_LLM_RETRIES=1`.
> 4. Rebuild + deploy the **processing image** (job_meta artifact + Open-Meteo
>    backfill; Scripts/ zipped recursively per the image-build memory).
> 5. Publish the web bundle (chat.js / portal.js / ballflight.js).
> 6. Smoke live: upload two swings ≥5 min apart → "compare this to my last
>    session", "what should I work on", "was it windy?" against real jobs.
> Not built (by design): md→json drill-card compiler (corpus is authored
> directly in drill_cards.json v0.1.0), conditions row on the ball-flight card
> (the coach covers conditions), P4 items (spoken recap, Aurora goals).

Scoping doc v2, 2026-08-01. v1 research summary retained in §0; §1–§6 updated with
Banjot's decisions:

1. Comparison unit is **session vs session** (not day vs day).
2. Weather + user context inputs: **approved**.
3. Coaching advice: **approved** — deprecate the blanket anti-prescriptive regex layer;
   advice focuses on **body posture/positioning** (what our indicators actually measure).
4. Backlog picks: **"what changed since last time" auto-brief**, **spoken range-session
   recap**, **DB-tracked goal tracking**.
5. New: **elicitation** — the coach proactively asks follow-ups (which club, what were
   the conditions, range or course) and captures answers as structured context.

> Note for the capstone writeup: item 3 consciously reverses the earlier "interpretive,
> NOT prescriptive" requirement (`coaching_indicators.py:9`). The reversal is a product
> decision by the team; the eval suite gets relabeled accordingly (§3.6). Document it as
> a requirements change, not a regression.

---

## 0. Where we are (research summary, unchanged from v1)

**Chat runtime.** Server-stateless Lambda (`deploy/chat_handler.py`) → Bedrock Haiku 4.5
(`bedrock-runtime`). The LLM has 8 tools (`Scripts/coaching_chat.py:320-395`), all reading
one preloaded `SwingContext` = scorecard A + optional scorecard B + ball_3d + KB. No S3
access, no DB; job artifacts fetched one-at-a-time over CloudFront and cached in `/tmp`.
History is client-held (last 20 msgs), sanitized server-side. Tool loop caps at 6
iterations; API Gateway imposes a hard **29 s** end-to-end budget.

**Swing browsing already half-exists.** `GET /jobs` (`deploy/jobs/jobs_handler.py`) lists
every processed job with `{job_id, date, ready, name}` — date = newest S3 `LastModified`.
The portal syncs this into a localStorage library and feeds chat the active job + up to
12 ready siblings (`portal.js:527-540`), but the live backend only ever loads 1–2
scorecards. The offline mock (`chat.js:199-223`) already does library-wide superlatives.

**Dates are weak.** No artifact carries a capture time; only S3 LastModified and the
client's upload-time clock. No per-user identity anywhere live (shared team token).

**Comparison machinery exists offline.** `Scripts/coaching_progress.py::delta_truth`
classifies each indicator improved/regressed/unchanged relative to the pro band, with an
evaluated prompt (`pg_readable`, composite 0.9998). Chat's `compare_indicator` is a thin
single-pair version.

**The system is mechanically anti-prescriptive** in five places: `PRESCRIPTIVE` regexes
in `coaching_eval_harness.py:51-56`, `coaching_chat.py:740-742`,
`coaching_qa.py:77` / `coaching_progress_eval.py:42` / `coaching_llm_eval.py:58`, and the
JS mirror `deploy/web/chat.js:266`; every KB card bans drills; "how do I fix my swing" is
`refuse_scope` gold at 1.0 refusal accuracy. No drill/coaching corpus exists in the repo.

**Ball flight has zero environment support.** `AIR_DENSITY = 1.225` hardcoded
(`ball_flight.py:39`, duplicated `ballflight.js:24`); `wind=` planned in
BALL_FLIGHT_PLAN.md:58 but never implemented.

**External.** [Open-Meteo](https://open-meteo.com/): free, keyless, hourly historical
weather to 1940 — backfills conditions for past swings from timestamp + location, plain
`urllib` from Lambda. Competitive scan: the 2026 differentiator is the closed loop
**diagnose → prescribe drill → verify on next upload**, which day/session comparison
enables and most apps lack.

---

## 1. Sessions: the unit of comparison and memory

### What is a session?

A practice visit: a run of uploads close together in time. Two-layer definition:

- **Derived (v1, zero migration)**: server clusters swings by `uploaded_at` gaps —
  swings within **2 h** of the previous swing belong to the same session. Deterministic,
  computed at listing time from the same data `GET /jobs` already aggregates; works
  retroactively on every existing job (LastModified as fallback date).
- **Declared (overrides derived)**: `session_id` + optional `session_label`
  ("Tuesday range, driver work") stamped into the new `job_meta.json` (§2) at upload —
  the portal keeps a "current session" for 2 h after an upload and offers
  "same session / new session" on the next one. User label wins over the clusterer.

### Chat tools (replaces v1's day-based design)

```
list_sessions()                  -> [{session_id, label?, date, start, end, n_swings,
                                     swings: [{swing_id, name, club?, flags}]}]
                                    newest first, capped ~20 sessions / 50 swings
load_swing(swing_id)             -> loads that scorecard (+ball) into context;
                                    returns its get_swing_summary. Max 4 live at once
                                    (LRU eviction), keeps the 29 s budget safe.
compare_swings(a, b)             -> full delta_truth output per reliable indicator:
                                    {improved|regressed|unchanged, delta, band status}.
compare_sessions(sa, sb)         -> per-indicator session aggregate: median of each
                                    reliable indicator across the session's swings,
                                    then delta_truth on the medians; also n_swings,
                                    flags cleared/appeared, best swing per session.
                                    Medians of body-scale-normalized angles are
                                    legitimate; ball-flight carries are NOT averaged
                                    across quality tiers — report best-measured only.
```

"How did today compare to Tuesday?" → `list_sessions` → `compare_sessions`. Follow-up
"which swing dragged it down?" → `load_swing` on the outlier. The single-swing
`compare_clip_id` path and `compare_indicator` stay for backward compat.

Aggregation for `compare_sessions` happens **in code, server-side** (a
`session_rollup(scorecards)` helper next to `delta_truth`) — the model never eyeballs
raw lists, same discipline as today.

### Plumbing

| Where | Change |
|---|---|
| `deploy/chat_handler.py` | Library-scope data layer for the 4 tools; `/tmp` cache keyed by job id; session clustering helper shared with jobs_handler. |
| `Scripts/coaching_chat.py` | `SwingContext` → dict of loaded scorecards + session index; new tool specs; `MAX_TOOL_ITERS` 6→8. |
| `deploy/upload/upload_handler.py` | Write `job_meta.json` (§2) with `uploaded_at`, `session_id`, `session_label`. |
| `deploy/jobs/jobs_handler.py` | Include `session_id`/`label` in `/jobs` when meta exists; expose derived sessions. |
| Web | Portal groups the library list by session; chat header shows active session; `#chat-compare` select demoted to a hint chip; TTS cache key → sha of answer text (answers now span swings). |
| Verifier | Add the new tools to the reliable-tool whitelist in `verify_chat_grounding`; numbers pool already recurses tool results. |
| Eval | New multi-session QA set (~30 items): session superlatives, ambiguous session refs, single-swing sessions, mixed-quality ball data, "am I improving" over 3+ sessions. Ship gate: composite ≥ 0.8. |

Honest limitations to label in UI: team-wide library until Cognito; derived session
boundaries are heuristic for pre-meta jobs.

## 2. Context: `job_meta.json`, weather, and elicitation (approved)

One new tiny artifact per job, `03_outputs/<job_id>/job_meta.json`, written at upload
and **updatable** via a new token-gated `POST /jobs/meta` (same `MC_RESULTS_TOKEN` gate;
schema-validated, whitelisted keys only, values length-capped):

```json
{ "uploaded_at": "...", "session_id": "...", "session_label": "...",
  "club": "driver", "location": {"lat": ..., "lon": ...},
  "setting": "range|course|sim", "ball": "range|premium",
  "weather": {"temp_c":..., "wind_mph":..., "wind_dir_deg":..., "humidity_pct":...,
               "pressure_hpa":..., "source": "open-meteo|user"},
  "notes": "user free text ≤200 chars" }
```

### 2a. Weather physics

- `ball_flight.py`: `air_density(temp_c, pressure_hpa, humidity_pct, elevation_m)`
  (ideal gas + vapor correction, pure stdlib — preserves the "safe to import in chat
  Lambda" rule) replaces the constant; `simulate_flight(..., wind=(speed_mph, dir_deg))`
  implements the parameter BALL_FLIGHT_PLAN.md:58 planned. Mirror both in
  `deploy/web/ballflight.js`. Sanity tests vs. rules of thumb (≈1 % carry per 5 °C;
  head/tailwind asymmetry).
- Auto-backfill: on upload, if geolocation was granted, the processing Lambda makes one
  Open-Meteo archive call for the upload hour → `weather` block. Degrade silently.
- Chat: `get_conditions(swing_id|session_id)`; `estimate_ball_flight` grows optional
  condition params. Rule: conditions adjust **what-if sims and cross-session fairness**
  ("Thursday was into a 12 mph wind") — never "correct" a measured carry.

### 2b. Elicitation — the coach asks, and remembers the answer

New tool making the interactivity real:

```
save_context(swing_id|session_id, {club?, setting?, ball?, notes?,
                                   weather_user?: {...}}) -> updated meta
```

- **When the model should ask**: `SYSTEM_RULES` gets an elicitation section — if a
  question's answer depends on missing context, ask **one** short follow-up instead of
  assuming. Concrete triggers: ball-flight question with `club` unknown (club drives
  `CLUB_DEFAULTS` tier selection, so this directly improves answers); session
  comparison where one session lacks weather; "why was I worse Thursday" with no
  setting recorded (range vs course).
- When the user answers ("it was my 9-iron, windy day"), the model calls
  `save_context` — the answer persists in `job_meta.json`, so next session's chat
  already knows. This is the difference between chat *using* context and chat
  *collecting* it.
- Trust boundary: user text enters prompts only via tool results the server rebuilds
  (existing design); `save_context` whitelists keys, caps lengths, strips markup.
  `weather_user` never overwrites an `open-meteo` block — stored alongside with
  `source: "user"`.
- Pose-quality tie-in (from v1 backlog #4, folded in here): `get_capture_quality`
  surfaces `pose_diag.json`, and its follow-ups are elicitation too — "was the camera
  waist-height?" → `save_context(notes=...)`, plus filming advice, which is
  uncontroversial under any advice policy.

## 3. Grounded posture coaching (approved — regex layer deprecated)

Focus per Banjot: **how to posture the body better** — which is exactly what the
pipeline measures. All 16 indicators are body-position/kinematics; the 9 with flag
rules are the coachable set:

| Flagged indicator | Fault language already in RULES (`coaching_scorecard.py:41-90`) |
|---|---|
| `shoulder_turn_top_deg` | limited coil / loses power |
| `x_factor_top_deg` | shoulder–hip separation |
| `hip_turn_impact_deg` | hips not cleared |
| `posture_loss_deg` | early extension / standing up |
| `head_sway_max_deg→pct` | swaying off the ball |
| `head_lift_max_pct` | lifting out of posture |
| `left_arm_bend_top_deg` | lead-arm collapse |
| `hip_lateral_shift_pct` | sway rather than rotate |
| `tempo_ratio` | rushed transition |

### 3.1 Drill/posture-cue corpus (the real work)

`Data/coaching/drill_cards/*.md`, compiled into the KB by extending
`build_indicator_kb_from_markdown.py`. ~15–20 cards (high-side and low-side variants
where both flag). Card schema:

```
indicator_key, direction (above|below band), title, setup, movement, reps,
feel_cue (body-position language), what_should_change {indicator_key, direction},
contraindications (e.g. skip if capture quality low), source_citation
```

Authored from standard instruction, cited, human-reviewed. Posture-first register:
cues are about body positions ("keep your trail hip depth through the downswing"),
matching both the ask and what we can verify.

### 3.2 `get_drills(indicator_key)` — retrieved, not generated

Returns matching cards **only if** that indicator is currently flagged
(`severity=="review"`) **and** reliable (tier high/med). Otherwise a structured refusal
("nothing measured out of range there"). Advice becomes exactly as grounded as numbers
are today: the model can only relay what a tool returned.

### 3.3 Verifier: replace, don't delete

Deprecate the blanket `prescriptive` violation; replace with
**`ungrounded_prescription`**: prescriptive language is a violation **unless** this
turn's tool results include a `get_drills` card whose title/cue matches the
recommendation (string-match, same technique as `_display_match` for numbers).
`ungrounded_number`, `low_confidence_leak`, `ungrounded_range_claim` all stay. Sites to
change (all five + mirror): `coaching_eval_harness.py:51-56`,
`coaching_chat.py:740-742,877-878`, `coaching_qa.py:77`,
`coaching_progress_eval.py:42`, `coaching_llm_eval.py:58`, `deploy/web/chat.js:266`.
Non-chat surfaces (`explanation.json` template, scorecard) keep the old ban — advice
lives in conversation, where the tool-grounding check exists.

### 3.4 KB + persona

Add `permitted_advice` to the 9 coachable cards; their `prohibited_inferences` drill
bans become "prohibited unless retrieved via get_drills". Billy Baroo gets an advice
register extending the existing sanctioned "mental cue" slot; keep the guardrail
"don't call it an improvement unless supported".

### 3.5 Close the loop (sessions × advice)

`get_drills` output carries `what_should_change`; on a later `compare_sessions` the
model checks whether the prescribed indicator moved as predicted ("we worked on hip
sway — 9.1 % → 6.3 %, inside the tour band"). This is the differentiator the
competitive scan says almost nobody ships. Persist the prescription in the session's
`job_meta.json` (`notes` or a `prescribed` key) so a later Lambda invocation knows what
was assigned.

### 3.6 Eval relabel + re-gate

- Relabel the `refuse_scope` gold for "how do I fix my swing" / "what should I work on"
  → `answer_with_drills`.
- Composite: drop the dead `nonprescriptive` 0.15 weight → new axis
  `advice_grounded_rate` (every recommended drill traces to a flagged reliable
  indicator + a corpus card; target 1.0).
- New advice QA set ~25 items: flagged → advise; unflagged → decline; low-confidence →
  decline; "give me a drill for club path" (unmeasured) → refuse; multi-flag
  prioritization (advise on the worst percentile first).
- Ship bar unchanged: composite ≥ 0.8 on the held-out set, 1.0 on advice grounding.

## 4. Approved backlog items

**4a. "What changed since last time" auto-brief** (~1 d after §1): on results load,
one server-side chat turn comparing the new swing vs. its session baseline (or previous
session's rollup) → banner + optional TTS. Pure reuse of `compare_swings`/`compare_sessions`.

**4b. Spoken range-session recap** (~1–2 d): "recap today" → `compare_sessions`
(current vs. previous) + within-session trend (first-half vs second-half medians —
computed in the rollup helper, not by the model) → TTS via the existing coach voice.
End-of-session nudge in the portal when a session has ≥3 swings.

**4c. DB-tracked goal tracking** (~3–4 d, the only item needing infra):
- Apply the authored Aurora migration (`deploy/db/migrations/0001_init.sql` — schema
  validated, never applied; `full_app.yaml` has the cluster at MinCapacity 0) and add
  `0002_goals.sql`:
  `goals(goal_id uuid PK, user_id uuid NULL, indicator_key text, target_value numeric,
  target_direction text CHECK (up|down|into_band), target_date date, status text,
  created_at timestamptz)` + index on `(user_id, status)`.
- Chat tools: `set_goal(indicator_key, target, by?)` (asks a confirming follow-up —
  elicitation again), `get_goals()`, `check_goal(goal_id)` — progress computed from
  the existing `results_history` view (`delta_vs_prev` per indicator) or, until
  identity lands, from the session rollups.
- Identity caveat: `user_id` stays NULL pre-Cognito → goals are team-scoped in the
  pilot, one shared locker. Label it in UI; the schema is ready for the flip.
- Processing already writes `swings`/`analyses`/`indicators` rows when
  `DB_CLUSTER_ARN` is set (`processing_handler.py:261-284`) — setting that env var is
  part of this item, and the chat Lambda gains its first (read-mostly) DB dependency
  via the Data API.

## 5. Sequencing

| Phase | Contents | Effort | Notes |
|---|---|---|---|
| P1 | §1 sessions (tools, job_meta, clustering, UI grouping) + §4a auto-brief | ~5 d | Foundation for everything else |
| P2 | §2 weather physics + Open-Meteo backfill + elicitation/save_context + capture-quality tool | ~4–5 d | Parallel-safe with P1 except job_meta (land that first) |
| P3 | §3 coaching: corpus authoring, get_drills, verifier replacement, eval relabel + re-gate | ~7–8 d | Corpus authoring dominates; start card drafts during P1/P2 |
| P4 | §4b spoken recap + §4c goals (Aurora apply, 0002 migration, goal tools) | ~5 d | Goals also unblocks future identity work |

## 5b. Research-loop corrections (2026-08-01, verified against code)

Five research iterations over the actual code corrected the design in these ways:

1. **Session metadata rides as S3 object metadata on the video, not a sidecar file.**
   The uploads bucket fires EventBridge on *every* Object Created (`full_app.yaml:187,243`),
   so a sidecar `job_meta.json` under `01_inputs/uploads/<job>/` would trigger a bogus
   processing run. Instead: the client sends session fields to `/upload-url`; the
   presigned-POST `Fields` carry them as `x-amz-meta-mc-*`; the processing Lambda
   `head_object`s the video, and writes `03_outputs/<job>/job_meta.json` with the other
   artifacts (it already has that write perm). No IAM change, no extra trigger.
2. **`list_sessions` reuses `GET /jobs` over HTTPS.** The chat Lambda has no boto3/S3
   access at all (by design); rather than grant IAM, it calls the existing jobs endpoint
   (new env `JOBS_API_BASE`) with `MC_RESULTS_TOKEN`, caching the listing in `/tmp` for
   ~60 s. Session clustering happens in the chat data layer from the listing's dates.
3. **The verifier needs date handling.** `_numbers_in` only pools numeric leaves, so an
   answer saying "your July 28 session" trips `ungrounded_number` ("28" ≥ 13). Fix:
   numbers inside date-like strings in tool results (ISO dates, "Jul 28") join the
   grounded pool; tokens in the answer that are part of a date pattern match against them.
4. **`delta_truth(scA, scB)` is (earlier, later)** — opposite of `SwingContext(a=current,
   b=earlier)`. The compare tools map explicitly and return `direction` from the LATER
   swing's perspective to avoid inverted narration.
5. **jobs listing date must exclude `job_meta.json`** from the newest-LastModified
   aggregate, or editing a session note would shift the job's date (and its session).
6. **TTS cache concern was a non-issue** — keys are content-hashed; `swing_id` only
   namespaces. Dropped from scope.
7. **NN ball-flight engine stays pinned to standard conditions.** Phys-NN was trained at
   ρ=1.225 with no wind; when wind or non-standard density is requested the sim falls
   back to Phys-Q and the result discloses it (`engine: "phys_q"`, `conditions_applied`).
   JS mirror: `ballflight.js:24` RHO/G duplication gets the same treatment.
8. **Prescriptive-regex sites number 13 files, not 6.** Runtime: `coaching_chat.py`,
   `deploy/web/chat.js`. Eval: `coaching_eval_harness.py`, `coaching_qa.py`,
   `coaching_progress_eval.py`, `coaching_llm_eval.py` (+ `coaching_chat_eval.py`,
   `coaching_qa_eval.py` graders). Report/viz tooling: `make_coaching_showcase.py`,
   `make_llm_eval_visual.py`. Docstrings: `coaching_scorecard.py`, `coaching_indicators.py`.
   Scope: invert the CHAT verifier + chat/qa evals; the one-shot explanation layer
   (scorecard → explanation.json) stays anti-prescriptive; report tooling untouched.
9. **Chat image must gain modules**: `deploy/chat/Dockerfile` COPYs Scripts files
   individually — add `coaching_progress.py` (compare), the new session/meta module, and
   the drill-cards data file when coaching lands.
10. **Backend timeout already exceeds the gateway**: AnthropicBackend default
    `timeout=40 s, retries=2` behind API Gateway's 29 s. Make both env-tunable
    (`CHAT_LLM_TIMEOUT`, `CHAT_LLM_RETRIES`) and set 18 s / 1 retry in the Lambda env.
11. **Auto-brief is client-fired.** On `openJob`, the portal sends one canned question
    ("what changed since my previous session?") through the normal `/chat` path and
    renders the answer as a banner — no new Lambda surface.
12. **Offline mock stays per-swing.** It only runs when `API_BASE` is unset (local dev);
    session tools are live-only. The mock keeps working unchanged.

## 6. Risks

- **29 s API Gateway ceiling** with 8 tool iterations + session rollups: mitigate with
  /tmp caching, 4-swing load cap, rollups precomputed per session on first touch;
  Lambda response streaming is the escape hatch if we hit the wall.
- **Advice quality rides on the corpus**, not the model — human review of every card is
  the quality gate; the verifier only proves provenance, not correctness.
- **Eval churn**: relabeled gold + new axes means historical composite numbers aren't
  comparable; snapshot the old leaderboard before relabeling (for the writeup's
  before/after story).
- **Team-wide data until Cognito**: sessions, goals, and the library are shared; every
  surface that says "your" should say "the team's" until identity lands.
- **Requirement reversal** (§ note at top): make sure the professor conversation is
  reflected in the final report; the eval history documents both regimes.
