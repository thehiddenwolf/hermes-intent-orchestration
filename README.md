# Hermes Intent Orchestration Plugin

This plugin links the Thinker and Doer/Research agents, integrating short-term and long-term memories with MediaWiki (Spectra Wiki) notes.

## Features

- **Streaming Tag Transformation**: Dynamically replaces XML thought tags (`<research>`, `<do>`, `<shortTermMemorize>`, etc.) with rich character action/emotions inline.
- **Short-Term Memory Integration**: Updates `MEMORY.md` automatically via `shortTermMemorize` and `shortTermForget` tools.
- **Long-Term Memory Integration**: Interacts with memos via `longTermMemorize`, `longTermForget`, and `longTermRecall`.
- **Subagent Delegation**: Routes tasks to specialized research (`spectra-research-agent`) or execution (`spectra-action-agent`) profiles.
- **Wiki Notes Integration**: Integrates page lookups and page reads from your MediaWiki notes.

## Installation / Deployment

To copy changes from this repository to your Hermes plugins:

```bash
cp __init__.py plugin.yaml ~/.hermes/plugins/intent_orchestration/
```
