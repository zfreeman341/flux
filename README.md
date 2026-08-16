# flux

[![CI](https://github.com/zfreeman341/flux/actions/workflows/ci.yml/badge.svg)](https://github.com/zfreeman341/flux/actions/workflows/ci.yml)

A Rust CLI that runs multi-agent LLM research workflows defined in TOML.

Workflows describe a DAG of LLM, HTTP, MCP, and local transform steps. flux
executes the DAG in dependency waves: steps that share no dependency edge run
concurrently, fan-out steps spread one call per item, and a hard model-spend
budget is enforced at every step.

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
# Build (Rust 1.88+, edition 2024)
cargo build --release

# Direct API steps need a key. Workflows built entirely from agent steps
# (provider = "claude-code") need no key at all — see Billing below.
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

### Billing: API credits or a Claude subscription

Steps that set `provider = "claude-code"` run through the `claude` CLI and bill
whatever that CLI is logged into — a Claude Pro/Max subscription, typically. To
keep that promise honest, flux removes `ANTHROPIC_API_KEY` from the subprocess
environment: the CLI silently prefers an inherited key over its stored login,
which would quietly bill API credits for work the subscription already covers.

A workflow whose every step is an agent step therefore needs no key at all, and
flux never constructs an API client for one:

```sh
env -u ANTHROPIC_API_KEY flux run my-agent-workflow.toml
```

Mixed workflows work too — API steps use the key, agent steps don't see it.

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

## Step kinds

Not every node in a research DAG should be a model call. A step declares its
`kind`, defaulting to `agent` so existing workflows are unaffected.

| Kind | What it does | Cost |
|---|---|---|
| `agent` | An LLM call: prompt, model, optional web tools. The original behavior. | Metered or notional |
| `http` | A plain request. The response is passed downstream in the upstream API's own shape. | Free |
| `mcp` | An explicit tool call to a named local stdio MCP server. | Free in Flux |
| `transform` | A named pure function over records — dedupe, group, rank. | Free |

Free steps are exempt from the budget reservation guard. That exemption is the
point: an HTTP fan-out over 70 endpoints draws nothing, and gating it on
remaining headroom would skip cheap deterministic work for no saving.

### MCP steps

MCP servers are declared once and referenced by name. Flux starts one child
process per MCP step, performs the protocol handshake, verifies the tool exists,
reuses that session across fan-out calls, and closes it when the step completes.

```toml
[mcp.servers.github]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-github"]
pass_env = ["GITHUB_TOKEN"]

[[steps]]
id = "issues"
kind = "mcp"
server = "github"
tool = "list_issues"
arguments = { owner = "{{ owner }}", repo = "{{ repo }}", state = "open" }
timeout_secs = 60
```

`arguments` is a typed TOML object: strings are minijinja templates, while
numbers, booleans, arrays, and nested objects retain their JSON types. Each
result is a validated `mcp_response` record with `server`, `tool`, `item`,
`is_error`, and `result` fields. Fan-out call failures become error records so
surviving calls continue; spawn, handshake, and missing-tool failures stop the
step before calls begin.

MCP steps are deterministic in orchestration—Flux selects the server and tool
rather than asking a model to choose—not necessarily pure. A tool can mutate
external state or incur its own service charges, neither of which is represented
by Flux's model budget. Workflow files can execute local programs; review them
before running. Child environments contain only `HOME`, `PATH`, `TMPDIR`, and
variables explicitly named by `pass_env`, and secret values are never logged.
Only local stdio servers are supported; remote transports and model-directed MCP
tool selection remain future work.

### Typed artifacts

Steps used to pass prose. A step that fetches JSON produces records instead, so
a DAG edge carries an `Artifact`:

```rust
enum Artifact {
    Text    { body: String, sources: Vec<Source> },
    Records { schema: Schema, rows: Vec<Value> },
}
```

`Records` carries the schema its rows satisfy, and rows are revalidated at every
step boundary against it. A row that fails is dropped
and logged, and the run continues — the same graceful degradation already used
for budget exhaustion. In templates, `{{ step.output }}` is the prose form (records
render as pretty JSON, so a model can read them), while `{{ step.rows }}` and
`{{ step.count }}` expose the records directly.

Fan-out generalizes to records: `parallel_over` a `Records` step runs one branch
per row, and the prompt or URL template can reach into it with `{{ item.field }}`.

### Extending flux

Schemas, transforms, and probes are looked up by name in a `Registry`. flux
registers a small generic set; a crate built on flux adds its own before
constructing the engine, so domain-specific steps live outside this repo:

```rust
let mut registry = Registry::builtin();
registry.register_schema(Schema::of::<Invoice>("invoice"));
registry.register_transform("overdue", |inputs: &[&Artifact], _: &TransformArgs| {
    all_rows(inputs).into_iter().filter(is_overdue).collect()
});

let outcome = flux::runner::run(&args, Arc::new(registry)).await?;
```

An unknown `output_type`, `transform`, or `probe` name is rejected by
`flux validate`, before anything is spent. Built in: the `record`,
`http_response`, and `mcp_response` schemas, and the `dedupe`, `difference`,
`top_n`, and `group_by` transforms.

## Step reference

| Field | Meaning |
|---|---|
| `id` | Unique step name; other steps reference it in `depends_on` and templates. |
| `kind` | `agent` (default), `http`, `mcp`, or `transform`. |
| `url` / `method` / `body` / `headers` | Request definition for `kind = "http"`. `url` and `body` are templates. |
| `probe` | A registered multi-request routine for `kind = "http"`, for lookups that are not one request per item. None are built in. |
| `server` / `tool` / `arguments` | Named server, exact tool, and typed templated argument object for `kind = "mcp"`. |
| `transform` | Which registered pure function to apply for `kind = "transform"`. |
| `args` | Keyword lists the transform reads, so filter policy lives in the workflow rather than in compiled code. |
| `output_type` | Name of the schema this step's rows are validated against (`record` for any JSON object). Also promotes an agent step's JSON reply to `Records`. |
| `model` | Anthropic model id (default `claude-sonnet-4-6`). |
| `prompt` | minijinja template for the step's prompt. |
| `depends_on` | Steps that must finish first; their outputs become template variables. |
| `tools` | Managed server tools: `web_search`, `web_fetch`. |
| `max_tool_calls` | Cap on tool invocations per step (default 5). |
| `max_tokens` | Output token budget per call (default 8096). Raise for long synthesis steps — hitting the cap is flagged with a visible `[NOTE: ...]` marker. |
| `provider` | `anthropic` (default, direct API) or an agent CLI: `claude-code`, `cursor`, `hermes`. Agent steps bill the CLI's own login, not your API key. |
| `parallel_over` | Fan out one call per item from the named upstream step's output. If that output contains a fenced block, only the block is read as the item list — so a planner's prose can't become an item. |
| `parallel_items` | Static fan-out list, when the items are known at authoring time. |
| `max_items` | Cap the fan-out size (protects the budget from a chatty planner). |
| `max_concurrent` | In-flight cap for fan-out calls (default 3), via a semaphore. |
| `cost_per_call_usd` | Pre-charged estimate per agent call; reconciled to the actual cost the agent CLI reports. Ignored for subscription-billed agents, which are never pre-charged. |
| `timeout_secs` | Per-call deadline: agent default 300 seconds, MCP default 60 seconds. |
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

**Scheduling is by dependency wave, not one step at a time.** Steps are grouped
into waves — a wave being every step whose dependencies are already satisfied —
and each wave runs concurrently. A DAG that opens with two independent steps
takes as long as the slower one, not their sum. The shared budget sits behind a
`tokio::sync::Mutex` so concurrent steps serialize only on the accounting, not
on the network calls.

**Multiple execution backends.** Direct API steps call Anthropic's Messages API
(`LlmProvider` trait), with web research handled by Anthropic's managed
server-side `web_search`/`web_fetch` tools. Agent steps (`AgentProvider` trait)
shell out to an agent CLI for tasks that benefit from a full tool-using agent.
HTTP, MCP, and transform executors handle explicitly selected deterministic
operations without involving a model.

**Choosing a fan-out type.** Both kinds can fan out in parallel; the
`provider` field on the fan-out step picks which:

| | API fan-out (default) | Agent fan-out (`provider = "claude-code"`) |
|---|---|---|
| Per-item cost | ~$0.05–0.15 (sonnet + a few searches) | ~$0.10–1.50 (subprocess with its own context) |
| Latency | one API round-trip, server-side search | agent boots, plans, and crawls |
| Best for | search-and-summarize per item | multi-hop crawling (follow links, check sub-pages) |
| Cost accounting | exact token billing per call | actual cost reported by the claude CLI |
| Billed to | your `ANTHROPIC_API_KEY` | the agent CLI's own login |

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

**Metered and notional spend are tracked separately.** A subscription-billed
agent still reports what its work *would* have cost on the API. Charging that
figure to the cap would abort runs that cost nothing, so it accumulates as
notional spend, reported for visibility and never enforced. The provider
declares which kind it is (`Billing::Metered` or `Billing::Notional`); only
metered spend is real money and only metered spend can trip the ceiling.

**Partial failure is expected, not fatal.** A fan-out call that fails after
retries is skipped; the synthesizer is told which items failed and writes
around the gap. Rate limits (429), overload responses (529), other 5xx, and
transient connect/timeout failures all get exponential backoff — jittered, so
a concurrent fan-out doesn't retry in lockstep — and a server-supplied
`retry-after` overrides the computed delay.

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
