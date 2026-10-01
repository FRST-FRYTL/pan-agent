# Evaluation

This page explains how PAN is measured, what the results are, and how to read them.

> **Results as of the September 2026 improvement round.** The round changed retrieval, the handling
> of what the assistant reports, user facts and preferences, and token overhead. Every
> stock-vs-PAN number from the multi-session scenarios was measured with harness 3.0 (see
> [Harness 3.0](#harness-30-stock-hermes-can-search-its-sessions)). The held-out test sets were run
> once, after the round, for three systems.

## How to read these numbers

- **They are the author's own measurements on one stack.** The model under test was Qwen3.8-27B
  (NVFP4) on vLLM 0.30.1rc1 + PR #50021 (MTP speculative decoding, prefix caching) on a single
  NVIDIA DGX Spark. PAN works with any OpenAI-compatible model, but a different model,
  quantisation or server will give different results.
- **The tested PAN used hybrid retrieval with the `pan-embedd` sidecar**, which this release does
  not package (see the README's known limitations). Without the sidecar, PAN retrieves with FTS5
  only. That configuration was not benchmarked on its own.
- **They are not leaderboard numbers.** The public datasets were converted into multi-session
  Hermes conversations, and they were scored with a different judge, model and metric than their
  papers use.
- **The held-out test sets are the independent numbers.** Nothing was tuned on them. Earlier,
  the LongMemEval test sets also ran at milestones as a regression guard, which never picked a
  change. In the final test each set ran once per system. The validation sets were used to accept or reject changes, so
  they are less independent.
- **The comparison is with stock Hermes on the same model.** It is not a comparison with other
  memory systems.
- **They measure recall across sessions, not use over weeks.** See
  [What is not measured](#what-is-not-measured).
- **The harness is private.** It is a private harness (not yet public), so the numbers cannot be
  reproduced yet. A public harness is on the roadmap.

## Method

### What is compared

- **Three systems on the same model.**
  1. **Stock Hermes:** unmodified Hermes Agent at the same pinned commit, installed into its own
     venv, with its built-in memory (`MEMORY.md` / `USER.md` and the `memory` tool) and
     `session_search`.
  2. **PAN, the version before the round:** the best memory pipeline before the improvement round
     (FTS5 retrieval).
  3. **PAN, final:** the memory pipeline of this release, with hybrid retrieval through the
     `pan-embedd` sidecar.
- Hermes + PAN is the same Hermes install with `memory.provider: pan` and `pan-memoryd` processing
  each session. All systems use the same model, server, sampling settings and tools. PAN's memory
  gate calls the same model.
- **Release work after the benchmark** does not change the memory pipeline: installer checks,
  `pan doctor`, the separate server cookbook, and secret redaction in rules-only mode.

### Multi-session scenarios through real Hermes

- A scenario is a script of 2–6 sessions with 1–2 user turns each.
- Every session is a fresh headless Hermes process, so only memory carries over.
- Each scenario gets a fresh profile and an empty workspace outside any git repository. Some
  scenarios place a config file or a log in the workspace first.
- The harness flushes curation between sessions, so the daemon's timing is idealised.

### Harness 3.0: stock Hermes can search its sessions

Earlier harness versions started Hermes with `--source tool`, which hides earlier sessions from
Hermes' `session_search`. Stock Hermes could therefore not search its own past sessions, and every
PAN-vs-stock gap measured that way was overstated. Harness 3.0 removes the flag. All
multi-session stock-vs-PAN numbers on this page were re-measured with harness 3.0, and the earlier
ones are withdrawn. The version before
the round was re-run in the same runs, so the round's effect is measured on the same harness and
server. The exceptions are marked: the ported agent-task evals never used the flag, so their
results stand.

### A server check before and after every run

At least 5 of 6 structured tool calls must succeed at both ends of a run, or the run is invalid. A
serving bug once corrupted tool calls without any visible server error, and it depressed both
systems ([local-models.md](local-models.md#tool-calls-that-break-only-on-repeated-requests)). The same check
ships as `pan doctor`.

### Judging

- A **blinded Claude judge** (Opus 5.5) scores each answer pass or fail against a per-check rubric.
  The model under test never judges.
- Each rubric is a conjunctive checklist, with no partial credit. Three fail conditions apply
  everywhere: a stale value stated as current, hedging between the old and the new value, and an
  invented value. "I don't know" fails a recall check and passes a noise or not-recorded check.
- A judge packet holds the user turns, the judged answers, the rubric and a reference answer. It
  carries no system name, run id or file name. For the test sets, the packets of all three systems
  were shuffled together before judging.
- Borderline verdicts and a random sample are reviewed by the evaluating agent session.
- Percentages are the judge's pass rates over the memory checks.
- **Known limits:** the judge is blind to the system, not to the content; PAN answers sometimes
  mention "the wiki". Judges also differ on one borderline abstention pattern ("none recorded" as
  the answer to "how many times did I …?"). Packets were shuffled across systems, so this adds noise
  but no bias towards one system.

### Data policy

Scenario data is split three ways (ADR-009 in [decisions.md](decisions.md)):

| Split | Used for | Who sees it |
|---|---|---|
| Dev | analysing failures, writing rules and fixtures, offline replay | the agents doing the tuning |
| Validation | accepting or rejecting one candidate change | only the evaluating session; the tuning agent gets back generic bug classes, never scenarios |
| Test | a milestone check, run only at milestones | nobody reads the items; only aggregate scores are looked at |

- **Validation sets wear out.** The rule is a fresh set every 2–3 decisions, written by an
  isolated agent with no access to PAN's code, results or other sets. The round started on a fresh
  set; the previous one had been used for four decisions and was retired to a regression guard.
- **Some round fixes were motivated by failed validation cases.** They were written as generic
  rules, but the validation gains may be partly fitted to those sets. The held-out test sets are
  the clean check.
- **Test sets are converted public data,** never tuned on, and never read item by item.
- **Contamination probe.** The LongMemEval-derived test questions were asked with no history at
  all. Leaving out the questions where "I don't know" is the correct answer, stock Hermes got one
  right, and that one was a guess.

### Generic rules only

PAN is meant as a general memory layer, so no rule may be keyed on benchmark content. A rule must be
language-level (syntax, discourse markers, value shapes), work in English and German, have a
precision guard, and use fixtures written independently of any scenario. Two findings show why:

- **Overfitting.** Of the first round of rule tuning, about 63–74 % of the gain carried over to
  unseen scenarios. The second round, tuned on a different set, regressed on unseen data. It also
  introduced a curator bug that blended an old and a new number into a value nobody had stated.
- **Dates.** Dates come only from event timestamps or plain-language date statements, not from
  benchmark converter markers; this costs about 2 pp on validation.

### Acceptance rules

- **The plateau rule** (ADR-010): rule tuning stopped after two consecutive iterations without a
  gain of at least 3 pp on fresh validation data. Below about 3 pp, differences were
  indistinguishable from noise at these set sizes. The LLM gate was adopted under that rule.
- **The improvement round** gave each task one acceptance rule, set before its validation run,
  plus guards: noise precision not below the reference, no drop of more than 2 pp on the other
  validation set, agent-task accuracy within its confidence interval, and append-only prompt-cache
  prefixes.

### Repetitions and intervals

- Every scenario ran 2–6 times per system, depending on the set (the Sonnet reference: once).
- Pooled pass rates carry Wilson 95 % intervals, shown in brackets.
- Differences on the test sets are paired per question (mean over the repetitions): the mean
  difference with a 95 % interval, a paired z test, and an exact sign test on wins and losses.

### The judge-drift check

Judge packets are kept, so old answers can be re-judged blind with the current judge prompt. That
separates a real regression from a change in the judge. After a server upgrade, re-judging an
internal set's old answers moved its score by +1.0 pp, with 99 % verdict agreement (n = 195). The
judge had not drifted.

### The agent-task regression guard

To check that PAN does not hurt ordinary agent work, five upstream Hermes evals were ported to the
harness and run against both systems. They use the upstream programmatic graders; only the
compaction eval is judged. Each cell ran 3 times, or 6 where the first 3 disagreed. Tokens, latency
and tool calls are reported next to accuracy. These evals do not use the harness flag above, so the
harness fix did not affect them.

## Results

### Held-out test sets (LongMemEval-derived)

Converted from the cleaned oracle split of
[LongMemEval](https://github.com/xiaowu0162/LongMemEval)
([paper](https://arxiv.org/abs/2410.10813)) into multi-session Hermes conversations. The oracle
split has short histories (1–6 evidence sessions per question), so this measures cross-session
recall, not retrieval from a long haystack. Each set ran once per system, 2 repetitions, judge-only
pass rate.

| Test set | Stock Hermes | PAN, before the round | **PAN, final** |
|---|---|---|---|
| English, 150 questions × 2 | 38.0 % [33, 44] | 69.7 % [64, 75] | **76.0 %** [71, 80] |
| German / mixed, 30 questions × 2 | 40.0 % [29, 53] | 68.3 % [56, 79] | **76.7 %** [65, 86] |

| Paired difference (per question) | English | German / mixed |
|---|---|---|
| final − stock | +38.0 pp [+29.6, +46.4], p < 0.001 | +36.7 pp [+19.1, +54.2], p < 0.001 |
| final − before the round | +6.3 pp [+0.4, +12.3], p = 0.037 | +8.3 pp [−3.3, +19.9], p = 0.16 (n.s.) |

- **The round's gain shows on the test sets too.** On English it is significant on its own. On the 30 German / mixed questions it points the same way but is not significant.
- **The version before the round reproduces its earlier score** on the new server and harness, so
  the gain is not a server or harness effect.
- **Cross-language** (German facts with an English question, or the reverse; 20 questions): stock
  30.0 %, before the round 67.5 %, final 77.5 %. Hybrid retrieval targeted this case.
- **Tokens:** on the English set, input tokens per scenario were 60.2k for stock Hermes, 49.2k
  before the round and 46.1k final. Stock Hermes spends more because it searches and reads its
  past sessions.

By LongMemEval question type (English set, pass rate in %):

| Question type (questions) | Stock | Before the round | Final |
|---|---|---|---|
| knowledge update (29) | 34.5 | 82.8 | **86.2** |
| temporal reasoning (29) | 51.7 | 82.8 | **86.2** |
| single-session, user (24) | 43.8 | 70.8 | **75.0** |
| single-session, assistant (14) | 28.6 | **75.0** | 71.4 |
| multi-session (40) | 43.8 | 62.5 | **66.2** |
| single-session, preference (14) | 0.0 | 28.6 | **67.9** |
| abstention (30, across types) | **91.7** | 85.0 | 81.7 |

Cells of 14–40 questions are noisy; the abstention and single-session-assistant drops are not
significant.

### Validation sets

Harness 3.0, same server. These sets were used to accept or reject the round's changes, so they are
less independent than the test sets. The final version ran in a later session than the other two.

| Set | Reps | Stock Hermes | PAN, before the round | **PAN, final** |
|---|---|---|---|---|
| Internal validation set (48 independently written scenarios, 260 checks per system) | 5 | 55.4 % [49, 61] | 87.7 % [83, 91] | **93.1 %** [89, 96] |
| Noise precision on that set (30 checks) | 5 | 86.7 % | 96.7 % | 96.7 % |
| LongMemEval-derived validation split (15 questions) | 3 (final: 6) | 55.6 % [41, 69] | 64.4 % [50, 77] | **83.3 %** [74, 90] |

- **The internal set** is multi-session scenarios in English, German and mixed, written by an
  isolated agent with no access to PAN's code or results.
- **Facts mentioned in passing** ("implicit" facts on that set, 20 checks): stock Hermes 15 %, PAN
  before the round 90 %. This is the gap PAN's capture targets.
- **Noise precision** measures whether irrelevant chatter is kept out of memory. With harness 3.0
  PAN is no longer behind stock Hermes here.
- **The LongMemEval validation split is small** (15 questions, two thirds of them single-session
  assistant questions), so it is a secondary signal only.

### What the round changed

Each task was accepted on its own validation run against the version before it. The per-task gains
below are PAN against PAN; the stock comparison is in the tables above.

- **Hybrid retrieval.** FTS5 plus multilingual embeddings (Qwen3-Embedding-0.6B), fused with
  reciprocal-rank fusion and reranked by a cross-encoder (bge-reranker-v2-m3), through the local
  `pan-embedd` sidecar. The curator finds the page to update with the same fusion, which turns
  duplicate pages into updates.
  - End to end on the internal validation set: 87.7 % → 88.1 % (n.s.); the cross-language category
    rose 75 % → 90 % (20 checks). Most remaining cross-language misses are capture misses, not
    retrieval misses.
- **Assistant-reported claims.** What the assistant said it did, decided or set up is kept as an
  `Assistant reported:` claim instead of being dropped (see
  [architecture.md](architecture.md#per-episode)).
  - On the ported Hermes session-search eval with Hermes' `session_search` turned off, so PAN's
    memory alone has to answer: 5/12 → 10/12 (4 tasks × 3 repetitions).
  - Internal validation set: 88.1 % → 88.1 %, with noise precision held at 29/30. A tool-fact
    category (facts read from a file or log) rose 35 % → 75 % (20 checks), from a rule that keeps
    dated tool records long-term.
  - The curator no longer strikes a still-valid claim when a second claim about the same subject
    lands on its page.
  - A hint that sent the agent to `session_search` after an empty `memory_search` gained 5.4 pp
    but cost noise precision (83 %). It was removed.
- **User facts and preferences.** Gate post-checks that wrongly dropped user facts (the gate's own
  "their" rewrite of "my", hedges in another clause, facts inside a question), a gate prompt that
  treats standing constraints as preferences, and a cleaner presentation of recalled facts.
  - Internal validation set: 88.1 % [84, 91] → 93.5 % [90, 96] (+5.4 pp; paired sign test
    p = 0.03). Its user-fact categories rose 85.8 % → 93.3 % (120 checks). Noise precision held at
    29/30.
  - On the test set, single-session-preference questions rose from 28.6 % to 67.9 % over the round
    (14 questions; table above).
  - Some of these fixes were motivated by validation cases, so the validation gain may be partly
    fitted. The test-set gain is the clean check.
- **Token overhead.** A focused prefetch query (absolute paths shortened, multi-paragraph messages
  also scored per paragraph), a relative rerank floor, at most 4 prefetched pages, marker
  explanations only where a marker appears, and compact tool schemas.
  - PAN's fixed tokens per request: 676 → 443 on an empty wiki, 737 → 483 on a 5-page wiki.
  - On the 13 ordinary agent tasks, unrelated pages were injected in 12 of 13 tasks before and in 0
    of 13 after (offline replay).
  - Paired median token ratio against the previous version, same session, 6 repetitions: 0.964
    [0.910, 1.022] on the 13 ordinary agent tasks, 0.966 on the tool-performance traps. Accuracy
    held (97.3 % vs 96.4 % on the ordinary tasks).
  - Internal validation set: 93.5 % → 93.1 % (n.s.), noise precision held at 29/30.

### Agent tasks and cost

| Measurement | Stock Hermes | Hermes + PAN |
|---|---|---|
| 13 ordinary agent tasks, version before the round, 3 reps | 96.7 % | 97.7 % (+1.0 pp [−2.3, +4.4], on par) |
| Tool-performance traps (tasks built to provoke tool errors), before the round, 3 reps | 74.1 % | 81.5 % (within noise) |
| Recall after context compaction, before the round, previous server (vLLM 0.22.1) | 54.3 % | **77.8 %** (+23.5 pp, p = 0.002) |
| Append-only prompt-cache prefixes, final version | 100 % | 100 % |

- **No accuracy regression on ordinary agent work.** The confidence interval of the difference
  includes 0. Each round task re-checked agent-task accuracy against its predecessor in paired
  runs, and every check passed.
- **Tokens per task.** Before the round, PAN used a paired median of about 1.07× stock Hermes'
  tokens on the 13 ordinary tasks; the mean was 1.40×, driven by a few long tool loops that stock
  Hermes also shows and that are not caused by memory. The longer prompt of the user-facts work
  raised the paired median to about 1.12× stock, and the token-overhead work then cut it to 0.96× of
  that. The final version was not run against stock Hermes on agent tasks in the same session, so
  its ratio to stock, about 1.08×, is inferred from these steps rather than measured.
- **Compaction is the clearest agent-side result.** The items are LongMemEval-derived, from the
  development split, not the test set. Answering from the compacted context alone is a tie, so the
  gain comes from PAN's prefetch and tools, not from the compressor both systems share. It was not
  re-run after the round.
- **Prompt caching is intact.** Prompt-cache prefixes were 100 % append-only in both systems.
- **Sample sizes are small** (3–6 repetitions, 4–27 items per eval). Only the compaction result is
  statistically clear.

### Frontier-model reference: Sonnet on a subset

**Sonnet via Claude Code headless, not the Anthropic API; 1 rep; a 25-scenario subset** of the
internal validation set (27 judged checks per repetition), harness 3.0. Only stock Hermes was run
with Sonnet; there is no Sonnet + PAN arm.

| System | Reps | Judge-only |
|---|---|---|
| Stock Hermes + Sonnet | 1 | 66.7 % [48, 81] |
| Stock Hermes + Qwen3.8-27B | 5 | 57.8 % [49, 66] |
| Hermes + PAN (before the round) + Qwen3.8-27B | 5 | 90.4 % [84, 94] |

- Sonnet with stock Hermes is not significantly better than Qwen with stock Hermes on this subset;
  the Qwen + PAN interval does not overlap with it.
- The judge is a Claude model and the system under test is Sonnet, a possible bias, mitigated by
  the blinding and the rubric.
- Tokens and latency are not comparable with the Qwen runs.

### Latency

- **Gate latency:** roughly 5–10 s per episode under benchmark load on the current server. The
  gate runs in the background daemon, so the agent does not wait for it, but new memories arrive
  with that delay.
- **Prefetch:** hybrid retrieval with reranking takes about 0.2 s per turn (FTS5 alone, a few ms).
- **Server:** moving to the current server (vLLM 0.30.1rc1 + PR #50021, MTP and prefix caching on)
  cut agent-task latency by 38–71 %, and answer quality stayed flat.

### Known weaknesses

- **Abstention:** on the LongMemEval-derived test set, correct "that was never mentioned" answers
  fell from 85.0 % before the round to 81.7 % (stock Hermes: 91.7 %; 30 questions, n.s.). The more
  PAN remembers, the more often it builds a concrete answer from nearby facts.
- **Multi-session aggregation** (counting or summing across sessions): 66.2 %.
- **Turn errors:** the final version still had 13 failed turns out of 1,084 on the English test
  set (stock Hermes: 92).
- **Retrieval at scale** is untested: the dense index is scored in pure Python, which is fine up
  to a few thousand units.

### ConvoMem-derived sets

These sets are derived from [ConvoMem](https://huggingface.co/datasets/Salesforce/ConvoMem), whose
data is licensed CC-BY-NC-4.0; only aggregate scores are reported here.

| ConvoMem-derived measurement (harness 3.0) | Stock Hermes | PAN, before the round | **PAN, final** |
|---|---|---|---|
| Held-out test set, 100 questions × 2, run once | 39.0 % [33, 46] | 72.5 % [66, 78] | **78.5 %** [72, 84] |
| All three test sets pooled (LongMemEval English and German/mixed, ConvoMem; 280 questions) | 38.6 % [35, 43] | 70.5 % [67, 74] | **77.0 %** [73, 80] |

- **Pooled, final − before the round: +6.4 pp** (paired over 280 questions, 95 % CI [+2.0, +10.9],
  p = 0.004; sign test 61 wins / 37 losses, p = 0.02). Final − stock: +38.4 pp.
- **ConvoMem test alone:** final − before the round +6.0 pp [−1.9, +13.9] (n.s.); final − stock
  +39.5 pp [+30.2, +48.8].
- **By category** (final): changing facts 79.0 %, user facts 62.5 % (before the round 37.5 %),
  implicit connections 63.2 % (flat), abstention 100 % for all three systems. The stock score
  includes the abstention items, so on the other items stock Hermes is well below 39 %.
- **Input tokens per scenario** fell about 5 % over the round.
- **Offline retrieval checks of the round** ran on dev data that includes ConvoMem-derived items
  (the larger, pooled wikis are built from them). Recall@5 of the right page on the pooled wiki
  (about 100–150 pages), FTS5 vs hybrid with reranking: English 0.79 → 0.91, German question over
  English pages 0.40 → 0.86. Prefetch on per-scenario wikis: the right page was injected for
  0.72 → 0.81 of answerable English questions and 0.09 → 0.67 of German ones, at the same precision.
  A curator replay of 468 dev decisions changed 24, mostly duplicate pages that became updates;
  22 were judged better and none worse.
- **Validation:** a ConvoMem-derived validation split was used during development, but it was not
  re-measured with harness 3.0, so no validation number is given.

## What is not measured

- weeks of real use with hundreds of facts;
- background timing: the harness flushes curation between sessions, which idealises it;
- the current code without the sidecar (FTS-only fallback). The version before the round was
  FTS-only, but it also lacks the round's other changes;
- other models, thinking mode, and retrieval at scale.

## Data and licences

LongMemEval is MIT-licensed. The ConvoMem data is licensed CC-BY-NC-4.0. It is not redistributed;
only aggregate scores are reported. No benchmark questions, answers or scenario files are included
in this repository.
