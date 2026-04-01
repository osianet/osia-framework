# Git Workflow & Autonomous Developer Practice

Kiro operates as a fully autonomous developer on this project. All code changes follow a feature branch → PR workflow. **Never commit directly to `main`.**

---

## Branch Naming

```
feat/<short-description>      # new features
fix/<short-description>       # bug fixes
chore/<short-description>     # tooling, deps, config
refactor/<short-description>  # code restructuring without behaviour change
```

Examples: `feat/rss-entity-dedup`, `fix/qdrant-timeout-handling`, `chore/bump-anthropic-sdk`

---

## Pre-Commit Checklist (MANDATORY)

Run all of these locally before committing. A CI failure after push means something was skipped.

### 1. Format
```bash
uv run --extra dev ruff format src/ scripts/ hf-spaces/
```
Applies formatting in-place. Run this first — it modifies files.

### 2. Lint
```bash
uv run --extra dev ruff check src/ scripts/ hf-spaces/
```
Must exit 0. Fix all errors before proceeding. Do not suppress rules without a documented reason in `pyproject.toml`.

### 3. Tests
```bash
uv run --extra dev pytest --tb=short -q
```
Must exit 0 (or exit 5 = "no tests collected", which is acceptable until the test suite grows).

### 4. Verify format check passes (mirrors CI exactly)
```bash
uv run --extra dev ruff format --check src/ scripts/ hf-spaces/
```
Must report "N files already formatted". If it would reformat anything, go back to step 1.

All four steps mirror the CI workflow in `.github/workflows/ci.yml` exactly. If they pass locally, CI will pass.

---

## Full Autonomous Workflow

### Step 1 — Create a feature branch from main
```bash
git checkout main
git pull origin main
git checkout -b feat/<description>
```

### Step 2 — Make changes, run pre-commit checks
After all code changes are complete:
```bash
uv run --extra dev ruff format src/ scripts/ hf-spaces/
uv run --extra dev ruff check src/ scripts/ hf-spaces/
uv run --extra dev pytest --tb=short -q
uv run --extra dev ruff format --check src/ scripts/ hf-spaces/
```
Fix any issues before continuing.

### Step 3 — Commit with a conventional commit message
```bash
git add -A
git commit -m "<type>(<scope>): <short summary>

<optional body explaining why, not what>"
```

Commit types: `feat`, `fix`, `chore`, `refactor`, `style`, `test`, `docs`

Examples:
- `feat(rss): add entity deduplication before research queue push`
- `fix(desk-registry): handle missing fallback model config gracefully`
- `chore(deps): bump qdrant-client to 1.13.0`

### Step 4 — Push the branch
```bash
git push origin feat/<description>
```

### Step 5 — Open a pull request

**Preferred: use the `git-mcp-server` tool** (platform: `gh-badrory`, project: `osianet/osia-framework`):

Use the `create_merge_request` MCP tool with:
- `platform`: `gh-badrory`
- `project_id`: `osianet/osia-framework`
- `source_branch`: the feature branch name
- `target_branch`: `main`
- `title`: conventional commit style title
- `description`: PR body with Summary, Changes, and Testing sections

This is authenticated as the `BadRory` collaborator account and avoids any `gh` CLI auth issues.

**Fallback: `gh` CLI** (if MCP is unavailable):
```bash
gh pr create \
  --repo osianet/osia-framework \
  --base main \
  --head feat/<description> \
  --title "<type>(<scope>): <short summary>" \
  --body "## Summary
<what this PR does and why>

## Changes
- <bullet list of key changes>

## Testing
<how this was verified — unit tests, manual checks, etc.>"
```

> **Note:** `gh` must be authenticated as `BadRory` — a collaborator account on the `osianet/osia-framework` repo. Check with `gh auth status`. If the active account is wrong, run `gh auth switch --user BadRory` first.

---

## GitHub MCP Integration

The `git-mcp-server` is configured with a `gh-badrory` platform providing full GitHub API access as the `BadRory` collaborator. Use this for all GitHub operations:

- **List PRs**: `list_merge_requests` (platform: `gh-badrory`, project_id: `osianet/osia-framework`)
- **Create PRs**: `create_merge_request`
- **View PR details/diffs**: `get_merge_request_details`, `get_merge_request_diff`
- **Comment on PRs**: `create_mr_comment`
- **List/create issues**: `list_issues`, `create_issue`
- **PR templates**: `get_mr_templates` (check for repo templates before creating PRs)

Always use `gh-badrory` as the platform name and `osianet/osia-framework` as the project ID.

### Step 6 — Report to the user
After the PR is open, provide:
- The PR URL
- A brief summary of what was changed and why
- Any follow-up items or known limitations

---

## Dependency Changes

When adding or updating a dependency:

1. Use `uv add <package>` (or `uv add --dev <package>` for dev-only tools)
2. This updates both `pyproject.toml` and `uv.lock` atomically
3. Commit both files together — never commit one without the other
4. For `hf-spaces/research-worker/requirements.txt`, pin to exact versions (`==x.y.z`)

---

## Rules

- **Never push to `main` directly** — branch protection requires a PR with 1 approving review and passing CI
- **Never force-push** to any branch
- **One logical change per PR** — keep PRs focused and reviewable
- **If CI fails after push**, fix it on the same branch with a follow-up commit — do not open a new PR
- **Pyright failures are informational** (`continue-on-error: true`) — they won't block CI but should be noted in the PR body if present
