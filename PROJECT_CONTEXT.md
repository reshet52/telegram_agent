# PROJECT_CONTEXT.md

Read this file only when the current task needs broader project history/rationale. `AGENTS.md` contains the active operating rules and roadmap; repository code is always the source of truth.

## Product goal
MyBot is a Telegram AI communication agent. Current development mode is assisted: the system reads a selected Telegram dialog and proposes three replies; the user decides what to send. Long term, the agent should understand each relationship, remember context, learn the user's style/edits, understand why a dialog is being maintained, and eventually manage explicitly enabled conversations autonomously.

Final deployment direction: server-hosted multi-user Telegram service for approved users. Each user's Telegram accounts/dialogs must remain fully isolated. A future local/custom model may replace or supplement OpenAI.

## Current audited runtime
Latest audit described this flow:

```text
live_agent.py
  -> load allowed/local accounts
  -> AccountBot starts control bot
  -> validate Telethon session / QR auth
  -> restore active dialog
  -> RuntimeController opens workspace/listener
  -> LiveEvents stores incoming + outgoing
  -> ReplyService builds context/memory/episodes
  -> OpenAI currently returns 3 variants
  -> user chooses/edits/sends
  -> reply journal records decision/sent-message events
```

Audited capabilities:
- account routing and separate Telethon sessions;
- active dialog restore/switching;
- per-dialog workspaces;
- live incoming/outgoing capture;
- history import;
- workspace preparation with progress/resume;
- episodes + embeddings;
- long-term memory + embeddings;
- 3 reply variants;
- session mood;
- reply journal;
- Telegram button/callback UI;
- experimental search.

The audit reported 73 offline tests passing at that checkpoint; it did not run live Telegram/OpenAI tests.

## Important storage architecture
Current conceptual storage:

```text
data/
└── account_<account_id>/
    ├── global/
    └── dialogs/
        └── dialog_<dialog_id>/
```

Dialog workspaces may contain history, deletions, episodes, embeddings, memory, profiles, request previews, initialization/checkpoint state, reply journal, etc.

Critical rule: person/dialog facts/history/memory must never leak to another dialog/account/user.

Legacy history may not be strict one-object-per-line JSONL; preserve existing parser compatibility unless deliberately migrating with tests.

A past global-state failure mixed dialogs and caused false deletions, missing history, checkpoint mismatch, silent rebuilds, duplicate episodes, and a huge embedding request. This is why stable IDs, workspace paths, explicit checkpoint failures, and resumable work are mandatory.

## Important current modules from audit
- `live_agent.py`: small bootstrap/composition.
- `telegram/account_bot.py`: control-bot/account startup.
- `services/runtime_controller.py`, `dialog_runtime.py`, `agent_manager.py`: active runtime/dialog lifecycle.
- `telegram/live_events.py`, `telegram/exporter.py`, `services/history_import.py`: history/live ingestion.
- `storage/workspace.py`, `history.py`, `deletions.py`, `atomic.py`, `control_state.py`: storage/state.
- `services/workspace_initializer.py`, `workspace_compaction.py`: dialog preparation/compaction.
- `episodes/incremental.py`, `checkpoint.py`, `maintenance.py`, `search.py`: episode pipeline/retrieval.
- `memory/incremental.py`, `manager.py`, `search.py`: memory pipeline/retrieval.
- `ai/client.py`, `context_builder.py`, `style_analyzer.py`, `services/reply_service.py`: AI request/generation.
- `services/global_style.py`, `correction_examples.py`: style/correction context.
- `storage/reply_journal.py`, `telegram/bot_reply_actions.py`, `bot_feedback.py`: generation/choice/edit/sent tracking.
- `telegram/bot_interface.py`, `bot_callbacks.py`, `bot_keyboards.py`, `bot_menu.py`, `bot_panel.py`, `bot_operations.py`: control-bot UI.
- `services/history_search.py`: experimental search at audit time.

Legacy/review candidates from audit: `episodes/builder.py`, `memory/merger.py`, `app/terminal_interface.py`, `storage/dialog_guard.py`, old `telegram/client.py`, and some scripts. Do not bulk-delete.

## Search decision
Current search quality is unsatisfactory: it can miss obvious facts, return strange candidates, and interpret ambiguous questions too literally. Existing retrieval sees a limited candidate set, so failure to retrieve is not proof that the fact does not exist.

Approved target:

```text
question
-> understand active dialog/person
-> optionally rewrite ambiguous query
-> retrieve from suitable sources
-> verify evidence
-> one best supported answer
-> confidence
-> source messages
-> alternatives only when genuinely ambiguous
```

Example: in an active person's workspace, "Как зовут собаку?" should be interpreted in that relationship context rather than blindly as "my dog".

Random Fact comes before Search v2. It should return a grounded fact + source evidence so the user can rediscover facts and later build a trusted benchmark of ~20 questions. Benchmark data is private/local, not Git data.

## Learning decision
The user wants learning based on the full behavior chain, not only a clicked variant.

Cases:
- `accepted_without_edit`
- `accepted_with_edit`
- `rejected`
- `independent_reply`
- `independent_after_generation`

Store both selected candidate and actual sent message. A clicked variant is intent; actual outgoing Telegram message is stronger evidence. User-authored replies without AI generation are valuable style examples too.

For edited replies, later learning should infer semantic reasons, not only raw text diff: formality, length, emotional intensity, phrase choice, follow-up questions, rhythm, initiative, etc. One event creates a hypothesis; repeated evidence increases confidence.

## Style decision
Use two levels:

```text
GLOBAL USER STYLE + PERSON-SPECIFIC STYLE
```

Person-specific style wins on conflicts.

Global style should be inferred by comparing 3-5+ diverse dialogs. Analyze each dialog separately, then compare recurring traits. User messages are style evidence; counterpart messages only provide context. Never promote person-specific facts to global style.

## Dialog Goal decision
The agent must eventually understand not only how to reply, but why the conversation exists.

Keep separate:
- Memory: what is known.
- Dialog Goal: desired outcome/purpose.
- Conversation State: what is happening now.

Possible future goal/state data:
- long-term objective;
- current objective;
- priorities;
- constraints;
- allowed initiative;
- open loops;
- unresolved questions;
- promised follow-ups.

Initially goal/state only influences suggestions. Autonomous action selection comes much later.

## Media decision
Current media often reaches prompts only as markers such as `[PHOTO]`/`[GIF]`.

Approved near-term policy:
- analyze only NEW media when needed for current reply;
- cache analysis locally;
- keep original media locally during prototype stage;
- do not bulk-process old media yet.

Before broader processing, run a controlled experiment such as 10 photos + 10 GIFs and measure cost, usefulness, latency, and GIF frame strategy. Development API budget target is roughly up to $1/day, but expensive jobs still require explicit approval.

Historical media backfill may later help learn how the user reacted to images/GIFs, but it is deferred.

## Model direction
Current provider is OpenAI. Long term, architecture should allow OpenAI/local/custom models without rewriting Telegram/history/memory/learning systems. Introduce a provider abstraction only when roadmap Step 13 is reached; avoid premature framework work.

The user expects future experiments with local pretrained models, RAG, LoRA/fine-tuning, or other custom approaches before choosing a production model.

## Prompt transparency
The user wants to inspect the exact effective request sent to the model, including separate `instructions`, `input`, retrieved context, and other real call inputs. A preview that omits effective fields is insufficient. Future UI should show/export this without duplicating prompt assembly.

## Bot UX
Current UI works but is too technical. Future UX should be organized around user tasks such as Dialogs, Reply, Search, Random Fact, Memory, Learning, Mood, Dialog Goal, Prompt Preview, Settings, Account. Do not redesign all UX before Search/Learning semantics stabilize.

## Final autonomous direction
Final autonomous mode is real autonomy for explicitly enabled dialogs: read incoming text/media, use memory/style/goals/state, choose whether/how/when to act, generate/send without confirmation, observe reaction, update state/learning.

Do not implement early. Prerequisites include reliable retrieval, memory, learning, style, dialog goals/state, media handling where needed, model quality, audit logging, server reliability, per-dialog opt-in, and immediate stop controls.

## Server direction
Final product is not merely a local Mac app. It should become a controlled server service for approved users.

Before server implementation, design separately:
- authentication/allowlist;
- user identity;
- Telegram credentials/session storage;
- secret management;
- user/account/dialog isolation;
- persistent database/storage;
- background jobs;
- quotas/API budgets;
- model access;
- logging/audit;
- backup/recovery;
- deployment.

Do not move to server architecture before the local product behavior is proven.

## Approved roadmap summary
0. Re-establish baseline and account for dirty changes/tests.
1. Grounded Random Fact.
2. Private ~20-question search benchmark.
3. Search v2.
4. Reliable learning-event capture.
5. Preference hypotheses from corrections.
6. Global style analysis across dialogs.
7. Integrate global + local style.
8. Dialog Goal + Conversation State.
9. Bot UX redesign.
10. Exact prompt inspection.
11. Media cost/quality experiment.
12. New-media understanding/cache.
13. ModelProvider boundary.
14. Local/custom model experiments.
15. Server/multi-user architecture.
16. Autonomous conversation engine.

Every step is gated: Codex implements one step, validates, gives exact manual test instructions, then stops until explicit user approval.

## Current continuation point
Start a fresh Codex context at Step 0.

At the latest audit, Git `main` matched `origin/main` at the current pointer, but the working tree contained modified Telegram UI files plus new search/test files. Treat the repository you open now as authoritative: inspect current `git status`/`git diff` rather than relying on old filenames/counts.

Step 0 must not change runtime code. It only establishes what is actually present now, runs safe offline checks where appropriate, reports discrepancies, and stops.
