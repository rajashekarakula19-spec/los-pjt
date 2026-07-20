# Deploy the dashboard to a free Hugging Face Static Space

Hugging Face currently offers Static Spaces free to all users. Docker and
Gradio Spaces require a paid plan, so this deployment publishes the dashboard
and its precomputed, de-identified analysis without running FastAPI.

## What the free edition includes

- Facility benchmarking and ranked opportunities
- Case-mix-adjusted LOS and cost results
- Confidence, robustness, and outlier-concentration metrics
- De-identified historical examples and dashboard filters

The experimental live prediction form is hidden because it requires a Python
backend. The analysis remains usable because it reads committed aggregate JSON.

## 1. Create the Space

1. Sign in at <https://huggingface.co/>.
2. Open <https://huggingface.co/new-space>.
3. Choose a name such as `los-opportunity-analyzer`.
4. Set visibility to **Public**.
5. Select **Static** and the **Plain HTML** template.
6. Create the Space.

The Space `README.md` should begin with metadata similar to:

```yaml
---
title: Finger Lakes Inpatient Opportunity Analyzer
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
pinned: false
short_description: Case-mix-adjusted LOS and cost opportunity analysis
---
```

## 2. Publish the static files

From the directory containing this project, replace `YOUR_HF_USERNAME`:

```bash
git clone https://huggingface.co/spaces/YOUR_HF_USERNAME/los-opportunity-analyzer hf-space

mkdir -p hf-space/data
cp urmc-los-cost/frontend/index.html hf-space/index.html
cp urmc-los-cost/backend/artifacts/config.json hf-space/data/config.json
cp urmc-los-cost/backend/artifacts/metrics.json hf-space/data/metrics.json

cd hf-space
git add README.md index.html data
git commit -m "Deploy static opportunity dashboard"
git push
```

Replace `urmc-los-cost` if the local project folder has a different name.

## 3. Authentication

If Git requests a password, use a Hugging Face write token from
<https://huggingface.co/settings/tokens>. Enter the Hugging Face username and
use the token as the password. Never place a token in a command, remote URL,
file, commit, screenshot, or chat.

## 4. Live URL

After the Space finishes building:

```text
Space page: https://huggingface.co/spaces/YOUR_HF_USERNAME/los-opportunity-analyzer
Direct app: https://YOUR_HF_USERNAME-los-opportunity-analyzer.hf.space
```

Use the link shown on the Space page if Hugging Face normalizes it differently.

## Updating later

Regenerate the project artifacts; copy `index.html`, `config.json`, and
`metrics.json` into the Space again; then commit and push. Do not upload the raw
SPARCS CSV because the static dashboard does not need it.

## Troubleshooting

If the Space reports a configuration error, confirm its YAML header contains
`sdk: static` and `app_file: index.html`. If the dashboard has no results,
confirm the Space contains:

```text
index.html
data/config.json
data/metrics.json
```
