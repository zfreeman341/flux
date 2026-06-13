# flux

A Rust CLI that runs multi-agent LLM research workflows defined in TOML.

Workflows describe a DAG of steps. Each step is an LLM call — with a prompt,
a model, optional web tools, and dependencies on earlier steps. flux executes
the DAG: sequential where steps depend on each other, parallel fan-out where
they don't, with a hard budget cap in dollars enforced at every step.

The premise: you get better answers to research questions by decomposing them,
investigating the parts in parallel, and synthesizing the results — than you
ever get from a single prompt.

```
$ flux explain examples/company-research.toml

Workflow: Company Research
Budget: $2.00

Step 1: planner
  Model:    claude-haiku-4-5-20251001
  Est cost: ~$0.01

Step 2: researcher
  Provider: claude-code
  Fan-out:  over 'planner' (max 3 concurrent)
  Est cost: ~$0.60

Step 3: synthesize
  Model:    claude-sonnet-4-6
  Est cost: ~$0.03

Estimated total: ~$0.64 of $2.00 budget
```

## Quickstart

```sh
# Build (Rust 1.85+, edition 2024)
cargo build --release

# flux talks to the Anthropic API directly
export ANTHROPIC_API_KEY=sk-ant-...

# Always look before you spend: validate, then estimate
flux validate examples/paper-compare.toml
flux explain  examples/paper-compare.toml

# Run it
flux run examples/paper-compare.toml --input topic="state-space models vs transformers"

# Research isn't just papers — plan a vacation
flux run examples/resort-scout.toml --input "two adults, beachfront, great food, quiet"
```

The final output lands in `outputs/`, and full per-step debug artifacts
(prompts, outputs, token counts, cost, timing) in `.flux-runs/<timestamp>/`.

## A workflow, annotated

```toml
[workflow]
name = "paper-compare"
description = "Find and compare recent papers on a research topic"

[budget]
max_usd = 1.50              # hard cap — the run aborts rather than exceed it

[inputs]
topic = { description = "Research topic to search for and compare papers on" }

[[steps]]
id = "find"
model = "claude-sonnet-4-6"
tools = ["web_search"]      # Anthropic's managed server-side search
max_tool_calls = 5
prompt = """
Search for two or three recent influential papers on: {{ topic }}
For each paper, note: title, authors, year, venue, and core contribution.
"""

[[steps]]
id = "compare"
depends_on = ["find"]       # runs after `find`; its output is available below
model = "claude-sonnet-4-6"
prompt = """
You found these papers on {{ topic }}:

{{ find.output }}

Compare them: problem solved, methodology, key results, limitations.
"""

[output]
step = "compare"            # this step's output is the workflow's result
```

Prompts are [minijinja](https://github.com/mitsuhiko/minijinja) templates.
CLI inputs are top-level variables (`{{ topic }}`); upstream step outputs are
namespaced (`{{ find.output }}`).

## CLI

| Command | What it does |
|---|---|
| `flux run <file>` | Execute a workflow. `--input k=v` and `--input-file k=path` bind inputs; `--output path` redirects the result. |
| `flux validate <file>` | Parse + static checks (unique ids, dangling deps, cycles, provider names) without spending anything. |
| `flux explain <file>` | Show the execution plan with a per-step and total cost estimate. No API calls. |
| `flux list-models` | The supported model shorthand. |

## Step reference

| Field | Meaning |
|---|---|
| `id` | Unique step name; other steps reference it in `depends_on` and templates. |
| `model` | Anthropic model id (default `claude-sonnet-4-6`). |
| `prompt` | minijinja template for the step's prompt. |
| `depends_on` | Steps that must finish first; their outputs become template variables. |
| `tools` | Managed server tools: `web_search`, `web_fetch`. |
| `max_tool_calls` | Cap on tool invocations per step (default 5). |
| `max_tokens` | Output token budget per call (default 8096). Raise for long synthesis steps — hitting the cap is flagged with a visible `[NOTE: ...]` marker. |
| `provider` | `anthropic` (default, direct API) or an agent CLI: `claude-code`, `hermes`. |
| `parallel_over` | Fan out one call per line of the named upstream step's output. |
| `parallel_items` | Static fan-out list, when the items are known at authoring time. |
| `max_items` | Cap the fan-out size (protects the budget from a chatty planner). |
| `max_concurrent` | In-flight cap for fan-out calls (default 3), via a semaphore. |
| `cost_per_call_usd` | Pre-charged estimate per agent call; reconciled to the actual cost the agent CLI reports. |
| `timeout_secs` | Per-agent-call deadline (default 300). |
| `reads_from` | Map of template variable → state file path, loaded before the step runs. |
| `appends_to` | State file the step's output is appended to (timestamped) after it runs. |

## Architecture

```
                       ┌──────────────┐
                       │   planner    │  cheap model decomposes the question
                       └──────┬───────┘
                 parallel_over│(bounded by semaphore)
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
      ┌───────────┐    ┌───────────┐     ┌───────────┐
      │researcher │    │researcher │     │researcher │   one per item,
      └─────┬─────┘    └─────┬─────┘     └─────┬─────┘   failures skipped
            └─────────────────┼─────────────────┘
                              ▼
                       ┌──────────────┐
                       │  synthesize  │  stronger model, sees all survivors
                       └──────┬───────┘
                              ▼
                    outputs/  +  .flux-runs/<ts>/
```

**Two kinds of steps.** Direct API steps call Anthropic's Messages API
(`LlmProvider` trait), with web research handled by Anthropic's managed
server-side `web_search`/`web_fetch` tools. Agent steps (`AgentProvider`
trait) shell out to an agent CLI (`claude -p ...`) for tasks that benefit
from a full tool-using agent — e.g. crawling a site's structure rather than
issuing one search. The traits keep providers swappable without touching the
engine — and make the whole engine testable against mock providers.

**Choosing a fan-out type.** Both kinds can fan out in parallel; the
`provider` field on the fan-out step picks which:

| | API fan-out (default) | Agent fan-out (`provider = "claude-code"`) |
|---|---|---|
| Per-item cost | ~$0.05–0.15 (sonnet + a few searches) | ~$0.10–1.50 (subprocess with its own context) |
| Latency | one API round-trip, server-side search | agent boots, plans, and crawls |
| Best for | search-and-summarize per item | multi-hop crawling (follow links, check sub-pages) |
| Cost accounting | exact token billing per call | actual cost reported by the claude CLI |

The ~10× per-item price difference compounds: a cheaper fan-out means you can
afford more items and more searches per item for the same budget — usually
better coverage than fewer, heavier agents. `examples/resort-scout.toml` uses
the API fan-out; `examples/company-research.toml` uses agents.

**Budget enforcement is first-class.** Every workflow declares `max_usd`.
Token usage is priced per step (including the per-call web-search surcharge);
parallel fan-outs are pre-charged before any call fires so a 20-item fan-out
can't blow through the cap mid-flight, and refunded for calls that never ran.
If the budget trips, remaining steps are skipped — except the final
synthesizer, which runs in an "emergency" mode so you get a partial result
instead of nothing.

**Partial failure is expected, not fatal.** A fan-out call that fails after
retries is skipped; the synthesizer is told which items failed and writes
around the gap. Rate limits (429) get exponential backoff that honors the
server's `retry-after`.

**Everything is observable — and verifiable.** Structured `tracing` logs per
step (tokens, cost, duration), and every run writes `.flux-runs/<timestamp>/`
with each step's exact prompt, output, and a cost summary table — which is
what makes prompt iteration tractable. Web-search citations are captured as a
per-step `Sources:` section, so claims in the final output trace back to real
URLs. Anomalous stop reasons are surfaced, not swallowed: output truncation
gets a visible marker, and paused server tool turns are automatically resumed.

**State lives in files, not a database.** Workflows are version-controlled
TOML. Run state is ephemeral markdown per run. Long-lived context (notes
accumulated across runs) is plain markdown that steps opt into via
`reads_from`/`appends_to` — declared in the schema, so a workflow's file
access is auditable.

## Design decisions

- **TOML, not YAML/JSON/DSL** — Rust-native (Cargo uses it), hand-writable,
  and deliberately too simple to grow accidental control flow. A real
  workflow language is future work, on purpose.
- **Files, not a database** — every artifact is greppable, diffable, and
  version-controllable. No daemon, no migrations, no ops.
- **Estimate before run** — `flux explain` prices the whole DAG offline and
  states each assumption it makes, because the worst failure mode of an LLM
  orchestrator is a surprise bill.
- **Cheap models plan, strong models synthesize** — Haiku decomposes and
  extracts, Sonnet reasons over the assembled evidence. The schema makes the
  per-step model choice explicit.

## Roadmap

- Deterministic replay of recorded runs
- Context truncation / summarization when assembled inputs grow large
- Workflow composition (workflows as steps)
- Streaming output for long synthesis steps

## Development

```sh
cargo check && cargo clippy -- -D warnings && cargo fmt && cargo test
```

Clippy warnings are errors here. The codebase stays formatted.
