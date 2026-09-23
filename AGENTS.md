# AGENTS.md

## Project overview

This repository contains a Python Telegram AI communication agent.

The long-term product goal is:

- the user starts the program once;
- after startup, normal operation is controlled from a private Telegram control bot;
- the user can choose any Telegram conversation/person;
- the system creates or loads an isolated workspace for that dialog;
- the agent reads and maintains that dialog's history, episodes, embeddings, long-term memory, profiles, and reply context;
- the agent proposes replies in the user's communication style;
- the user remains in control of sending; no automatic sending by default;
- over time, the agent should learn from the user's real choices and edits.

This is a generic product architecture. Never hardcode a specific person such as Nikko, Sean, or any other dialog into product logic.

---

## User workflow goal

The desired end-user workflow is:

```text
start the app once
    ↓
Telegram account client connects
    ↓
private Telegram control bot starts
    ↓
all normal control happens in the bot
```

The user should not need to manually run indexing, export, memory, migration, or analysis scripts during normal use.

Python files under `scripts/` may remain as development, debugging, recovery, migration, or maintenance utilities, but they are not intended to be the final user interface.

Prefer one application entry point with many internal modules, not one giant Python file.

---

## Current project state

The project has already undergone a large refactor from a single-dialog prototype toward per-dialog workspaces.

Important current capabilities:

- Telethon is used for the real Telegram account and dialog history/live events.
- A Telegram Bot API bot is used as the private control interface.
- The bot is restricted to the owner and private chat.
- Reply generation returns three variants.
- Session-only mood exists and affects generation.
- `/show`, `/reply`, `/mood`, `/clear_mood`, `/update_memory` exist.
- `/dialogs` exists and reads Telegram dialogs through Telethon.
- `/select <dialog_id>` exists as an early dialog-management step.
- `AgentManager` exists in an early form and can find/select a dialog by stable Telegram dialog ID.
- `/select` currently does **not** switch the active workspace/runtime yet. This is intentional at the current checkpoint.
- The current runtime still begins by selecting one dialog before the control bot becomes the full controller.
- Per-dialog workspace paths are already wired into normal history, episodes, memory search, reply context, and memory update code.
- A migrated, mature existing dialog is the canonical regression-test workspace. Do not hardcode its IDs; derive workspace data normally.
- Live incoming and outgoing messages are saved to history and included in context.
- The control bot currently mirrors incoming counterpart messages, but intentionally does not mirror the user's own outgoing messages yet. This is a known UX gap to fix later.
- `live_agent.py` has been refactored so major responsibilities are separated into runtime preparation, live events, terminal/UI concerns, etc. Do not grow it back into a large monolith.

### Current stopping point

The last completed/pushed development step was dialog discovery/selection support in the Telegram bot:

```text
/dialogs
/select <stable_dialog_id>
AgentManager
```

The next planned UI step was **not implemented yet**:

- replace command-centric control with inline Telegram buttons;
- introduce a main menu;
- show dialogs as paginated buttons;
- select dialogs through callback buttons;
- retain commands temporarily as fallback/debug controls.

Do not assume that `bot_keyboards.py`, inline-button menus, or callback handlers already exist unless you inspect the repository and confirm them.

---

## Key architecture

Expected package shape is approximately:

```text
MyBot/
├── live_agent.py
├── mybot/
│   ├── config.py
│   ├── ai/
│   │   ├── client.py
│   │   ├── context_builder.py
│   │   └── style_analyzer.py
│   ├── app/
│   │   ├── session_state.py
│   │   └── terminal_interface.py
│   ├── telegram/
│   │   ├── client.py
│   │   ├── dialogs.py
│   │   ├── exporter.py
│   │   ├── bot_interface.py
│   │   └── live_events.py
│   ├── storage/
│   │   ├── history.py
│   │   ├── deletions.py
│   │   ├── workspace.py
│   │   └── dialog_guard.py
│   ├── episodes/
│   │   ├── builder.py
│   │   ├── incremental.py
│   │   └── search.py
│   ├── memory/
│   │   ├── manager.py
│   │   ├── merger.py
│   │   ├── incremental.py
│   │   └── search.py
│   └── services/
│       ├── reply_service.py
│       ├── dialog_runtime.py
│       └── agent_manager.py
└── scripts/
```

Inspect the actual repository before editing; this document describes the intended/current architecture but the code is the source of truth.

---

## Workspace isolation: critical invariant

Each Telegram account/dialog must have its own isolated workspace.

Conceptually:

```text
data/
└── account_<account_id>/
    ├── global/
    └── dialogs/
        ├── dialog_<dialog_id>/
        ├── dialog_<dialog_id>/
        └── ...
```

A dialog workspace may contain:

```text
workspace.json
chat_history.jsonl
deleted_message_ids.json
ai_request_preview.txt
full_history_analysis.txt
episodes.jsonl
episode_embeddings.json
episode_embeddings_live.jsonl
agent_memory.json
agent_memory_live.jsonl
memory_embeddings.json
memory_embeddings_live.jsonl
memory_update_state.json
user_profile.json
person_profile.json
chunk_analysis/
```

### Never violate this rule

Person/dialog-specific information must never leak between dialogs or accounts.

For example:

```text
facts about person A
must never be retrieved while replying to person B
```

Whenever a new feature reads or writes:

- history;
- deleted message IDs;
- episodes;
- episode embeddings;
- memory;
- memory embeddings;
- person profile;
- request previews;
- indexing/update state;

it must use the active workspace paths rather than global legacy paths.

---

## Future account-level/global data

The system should eventually distinguish:

### Account-global knowledge

Examples:

- user's general writing style;
- broad communication preferences;
- correction patterns learned across dialogs;
- general reply preferences.

Likely location:

```text
data/account_<id>/global/
```

### Dialog-local knowledge

Examples:

- facts about a specific person;
- relationship history;
- dialog episodes;
- dialog memories;
- person profile;
- temporary conversation context.

Do not blindly promote dialog-local facts into global style memory.

The current `user_profile` location may still be dialog-local for compatibility; do not redesign this casually without first understanding its contents.

---

## Current storage/history quirks

### Legacy "JSONL"

The historical `chat_history.jsonl` format is not guaranteed to be strict one-JSON-object-per-line JSONL.

Legacy files may contain pretty-printed JSON objects separated by blank lines.

Preserve compatibility with the existing history parser/writer unless deliberately migrating the format with tests and a safe migration path.

### Deleted messages

Deleted-message handling exists and must remain workspace-specific.

Never apply deleted IDs from one dialog to another.

A previous cross-dialog failure demonstrated that this can corrupt history state.

---

## Episode system

The episode system has already been made workspace-aware.

Important behavior:

- episode embeddings are created in batches;
- base and live embedding stores may contain overlapping episode IDs;
- semantic search deduplicates by episode ID;
- overlap between base/live data can therefore be expected;
- a missing nonzero last processed message ID is an error and must not silently trigger rebuilding from zero.

Do **not** restore behavior like:

```text
tracker state cannot find last ID
→ silently rebuild all history
```

That previously caused duplicate episodes and oversized embedding requests.

Initialization should create embeddings before persisting corresponding episode/index state where possible, to reduce partial-state corruption.

Future indexing should be resumable and checkpointed.

---

## Memory system

Long-term memory is workspace-aware.

Memory retrieval should use the active workspace's base/live embedding files.

Memory updates should use the active workspace's:

- history;
- deleted-message file;
- episode data;
- live memory;
- live memory embeddings;
- update-state file.

The system currently supports manual memory update. Long-term direction is automatic background maintenance after meaningful batches of new messages/episodes, not an API call for every message.

Future memory should distinguish at least:

```text
person facts
relationship/dialog facts
user style
behavior patterns
correction patterns
temporary/session context
important episodes
```

Do not turn temporary state (for example a mood tonight) into a permanent user fact.

---

## Session mood

Mood is session-only.

It should influence tone, energy, length, emotionality, and wording, but should not automatically be stored as a permanent fact.

Existing behavior worked well in live testing and should be preserved.

---

## Reply generation

`ReplyService` is responsible for constructing/generating reply variants.

Current behavior:

- generates three variants;
- uses recent context;
- uses relevant long-term memories;
- uses semantically similar episodes;
- accepts session mood;
- writes an AI request preview to the active workspace.

Preserve the working behavior when refactoring.

No automatic send by default.

---

## Full prompt inspection: planned feature

A future control-bot feature must allow the user to inspect the **complete final prompt/request that would be sent to the model**.

Desired UX may include:

```text
[Show prompt]
```

or a command such as:

```text
/prompt
```

The bot should be able to:

- send the complete prompt as a `.txt` file;
- optionally show it in Telegram messages/chunks;
- ideally show the exact assembled request before generation when requested.

Purpose:

- debugging practical prompt quality;
- understanding why a reply was generated;
- collecting feedback from future users;
- improving prompts without guessing.

The existing workspace `ai_request_preview` should be reused where appropriate rather than creating duplicate prompt-building logic.

---

## Telegram control bot: product direction

The control bot is intended to become the primary product UI.

Commands may remain as fallback/debug tools, but normal use should become button-driven.

### Near-term desired main menu

Conceptually:

```text
AI Agent

[💬 Dialogs]      [✍️ Reply]
[📖 Context]      [🎭 Mood]
[🧠 Memory]       [⚙️ Settings]
```

Do not over-optimize the final menu yet. It is acceptable to expose extra controls during development so the user can learn what is useful.

### Dialog menu

Desired:

- inline buttons;
- pagination;
- button callback uses stable `dialog_id`, not the temporary list position;
- choosing a dialog must not automatically trigger expensive indexing before confirmation.

For many dialogs, show a limited number per page.

### Existing commands

Keep working commands during the transition unless there is a good reason to remove them.

They are useful for debugging even when buttons become the main UI.

---

## AgentManager: planned central role

`AgentManager` should evolve into the central runtime coordinator.

Long-term responsibilities:

```text
Telegram account
active dialog
active workspace
active runtime
listener lifecycle
dialog switching
workspace initialization status
background jobs
```

Target dialog-switch flow:

```text
user selects dialog in control bot
    ↓
AgentManager resolves stable dialog ID
    ↓
checks workspace state
    ↓
if ready: switch runtime/listener/context
    ↓
if missing: offer workspace initialization
```

Eventually the application should no longer require console `choose_dialog()` for normal startup.

Do not rush to support multiple simultaneous active dialogs until single-active-dialog switching is safe.

---

## New dialog/workspace initialization: planned flow

Selecting a completely new dialog must not immediately cause an uncontrolled large OpenAI job.

Desired flow:

```text
select dialog
    ↓
workspace missing
    ↓
show dialog/history information
    ↓
ask user to create workspace
    ↓
import history
    ↓
build episodes
    ↓
show estimated/actual indexing work
    ↓
confirm/start embeddings
    ↓
extract/index memory
    ↓
mark workspace ready
```

All of this should be initiated and monitored through the Telegram control bot, not by manually running Python scripts.

### Requirements

- resumable;
- checkpointed;
- crash-safe;
- no silent rebuilding from zero;
- progress visible in bot;
- background work must not freeze the control bot;
- allow cancellation/pause later;
- avoid surprise API costs.

---

## Learning from real user replies: high-priority future feature

This is one of the most important future product features.

The agent should learn from what the user **actually sends**, especially when the user:

- chooses an AI variant;
- edits it;
- sends it;
- rejects all variants and writes something else.

### Desired feedback record

Conceptually:

```json
{
  "context_id": "...",
  "generation_id": "...",
  "ai_variants": ["...", "...", "..."],
  "chosen_variant": 2,
  "actual_reply": "...",
  "result": "accepted_with_edit",
  "timestamp": "..."
}
```

Possible labels:

```text
accepted_without_edit
accepted_with_edit
rejected
independent_reply
```

Do not use only raw edit distance as the learning signal. Later analysis should infer useful preference patterns.

Examples:

- AI replies are too long;
- user often removes poetic language;
- user prefers certain greetings;
- user changes emotional intensity;
- user avoids certain phrases;
- user chooses shorter/direct variants in certain contexts.

### Retrieval of correction examples

Future generation context should be able to retrieve semantically relevant past corrections:

```text
current dialog context
+ relevant memories
+ relevant episodes
+ relevant correction examples
+ style/profile
```

This provides personalization without requiring fine-tuning.

---

## Reply-selection UX: planned

Eventually each generated variant should have an identity:

```text
generation_id
variant_id
```

The control bot should provide actions such as:

```text
[Send]
[Edit]
```

or equivalent.

If the user edits before sending, store both:

```text
AI candidate
actual sent reply
```

This is the basis for correction learning.

No automatic sending without explicit confirmation unless the product requirements are deliberately changed later.

---

## Mirror both sides of the live conversation

Known UX issue:

The live history/context already sees the user's own messages, but the control bot currently mainly pushes incoming counterpart messages.

Future control-bot timeline should show both directions, for example:

```text
⬅️ Other person
How was your day? ❤️

➡️ You
My day was not bad...
```

This is important both for usability and for visually understanding correction-learning behavior.

Do not confuse this UI issue with storage: outgoing messages are already being captured by the live runtime.

---

## Background jobs and progress

Long-running work such as:

- initial history import;
- episode construction;
- embedding generation;
- memory extraction;
- style analysis;

should eventually run as background jobs.

The control bot should stay responsive.

Desired progress UX:

```text
Preparing dialog

History       ✅
Episodes      ✅ 2143
Embeddings    ⏳ 1200 / 2143
Memory        ⏸
```

Use persistent state/checkpoints so a restart continues rather than repeats work.

---

## Multi-dialog future

Later, the agent may monitor multiple dialogs at the same time and act as an AI communication inbox.

Do not implement this before:

1. workspace isolation is proven;
2. safe single-dialog switching works;
3. runtime lifecycle is centralized in AgentManager;
4. background initialization is reliable.

---

## Bot UI code organization

Avoid turning `mybot/telegram/bot_interface.py` into another monolith.

It is already substantial.

As button/menu functionality grows, prefer extracting components such as:

```text
mybot/telegram/bot_keyboards.py
mybot/telegram/bot_callbacks.py
mybot/services/workspace_initializer.py
```

Exact names are flexible; preserve clear responsibility boundaries.

---

## `live_agent.py`

Keep `live_agent.py` small.

It should act as a composition/bootstrap layer, not contain:

- large history-import workflows;
- episode internals;
- bot menu implementation;
- memory logic;
- giant event handlers.

The project intentionally refactored it down from 700+ lines.

Do not reintroduce those responsibilities.

---

## Code-change style

The user prefers incremental, understandable changes.

When working:

1. inspect current code first;
2. change one coherent subsystem at a time;
3. preserve existing working behavior;
4. explain which files were changed and why;
5. avoid large rewrites unless clearly justified;
6. distinguish refactor work from new feature work;
7. avoid hardcoded dialog/person assumptions.

Do not create clever abstractions solely for abstraction's sake.

---

## Validation

For normal Python changes, use lightweight checks such as:

```bash
python -m py_compile path/to/changed_file.py
python -m compileall mybot
```

When appropriate, run focused tests.

Do not run expensive OpenAI operations, bulk embeddings, destructive migrations, or live Telegram workflows merely to "test" unless the task explicitly requires them or the user approves.

A real live regression test should usually use the already-mature existing workspace before trying a new dialog.

---

## Safety around runtime data

Treat the following as sensitive/runtime state:

```text
.env
*.session
*.session-journal
data/
memory/
exports/
```

Do not:

- commit secrets;
- print API keys/tokens;
- delete or rewrite runtime data casually;
- move real dialog data between workspaces;
- use one dialog's state to initialize another.

The repository's `.gitignore` is expected to ignore runtime data such as:

```gitignore
.env
/venv/
__pycache__/
*.pyc
*.session
*.session-journal
/data/
/memory/
/exports/
.DS_Store
```

Root anchoring of `/memory/` matters because `mybot/memory/` is source code and must not be ignored.

---

## Git behavior

The user controls Git workflow.

Do not commit, push, reset, rebase, delete branches, or rewrite history unless explicitly asked.

When asked to prepare a checkpoint, first inspect status/diff and mention unexpected changes.

Generated/runtime data under ignored folders should not enter commits.

---

## Important past failure / regression to avoid

A previous version used global legacy data while the user accidentally selected a different dialog.

This caused:

- one person's message to be appended into another person's history;
- false deletion reconciliation;
- missed later history;
- episode tracker mismatch;
- silent rebuild from zero;
- thousands of duplicate episodes;
- an oversized embedding request.

This incident is the reason for:

- per-dialog workspaces;
- strict path isolation;
- stable dialog IDs;
- no silent reset/rebuild;
- cautious new-dialog initialization.

Any design that could recreate cross-dialog state mixing is unacceptable.

---

## Current recommended development sequence

Continue from the current pushed checkpoint in this order:

### 1. Button-first Telegram UI

Implement inline-button navigation while keeping commands as fallback.

Start with:

- main menu;
- dialogs button;
- paginated dialog buttons;
- stable dialog IDs in callback data;
- back/home navigation.

Do not switch workspaces yet merely by clicking a dialog until the lifecycle is implemented safely.

### 2. Evolve AgentManager

Make it own active-dialog/runtime state rather than merely storing a selected dialog.

### 3. Remove console dialog selection from normal startup

The program should launch the control bot first and wait for bot interaction.

### 4. Safe dialog switching

Switch listeners/context/workspace without restarting Python.

### 5. Workspace initialization through bot

Create/import/index a new dialog from the bot with confirmation and progress.

### 6. Mirror both incoming and outgoing live messages in the bot

Make the control bot a useful conversation timeline.

### 7. Reply-variant buttons

Each generated reply gets selectable actions.

### 8. Correction-learning data capture

Relate selected variant to actual sent reply.

### 9. Retrieve correction examples during generation

Use learned choices as additional context.

### 10. Prompt inspection/export

Expose the complete final prompt/request as text and `.txt`.

### 11. Automatic/background memory and indexing maintenance

Remove routine manual maintenance commands from the normal workflow.

### 12. Multi-dialog monitoring later

Only after the previous lifecycle/isolation work is robust.

---

## First task for the new Codex session

Before editing anything:

1. read this `AGENTS.md`;
2. inspect the repository tree and current Git status;
3. inspect:
   - `mybot/telegram/bot_interface.py`
   - `mybot/telegram/dialogs.py`
   - `mybot/services/agent_manager.py`
   - `mybot/services/dialog_runtime.py`
   - `live_agent.py`
4. verify that the current checkpoint matches the description above;
5. do not assume the unimplemented inline-button work already exists.

Then implement the next small feature:

> Introduce the first button-driven Telegram UI without changing runtime/workspace selection semantics yet. Keep existing commands as fallback. Prefer extracting keyboard/menu definitions from `bot_interface.py` so it does not continue growing.

A suitable first iteration is:

```text
/start
→ main inline menu

[💬 Dialogs] [✍️ Reply]
[📖 Context] [🎭 Mood]
[🧠 Memory]  [⚙️ Settings]
```

`Dialogs` should open paginated inline buttons. Selecting a dialog should initially only select/preview it through the current AgentManager and clearly state that the active workspace has not yet been switched.

After that works, stop and report:

- files changed;
- behavior added;
- validation run;
- recommended next step.

Do not jump ahead into workspace initialization or multi-dialog runtime switching in the same patch.

---

## Working with OpenAI/Codex documentation

When a task depends on current OpenAI API or Codex behavior, use current official OpenAI documentation rather than relying on stale assumptions.

If an OpenAI developer documentation MCP server is configured in Codex, prefer it for OpenAI-specific documentation questions.

---

## Product principle

The project is not merely "an LLM that sees chat history."

The intended product is a user-controlled AI communication system that:

```text
understands each relationship separately
remembers relevant history
retrieves context intelligently
learns the user's real communication preferences
improves from accepted/edited/rejected suggestions
remains transparent and inspectable
is controlled through a convenient Telegram UI
never mixes one person's private context with another person's
```

Preserve that direction when making local implementation decisions.
