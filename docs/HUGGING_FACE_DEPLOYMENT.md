# Deploy the full application to Hugging Face Spaces

Hugging Face Spaces can run the FastAPI backend and experimental prediction
sandbox on the free CPU Basic tier. The Space sleeps when unused and wakes when
visited.

## 1. Create the Space

1. Sign in at <https://huggingface.co/>.
2. Open <https://huggingface.co/new-space>.
3. Choose a Space name such as `los-opportunity-analyzer`.
4. Set visibility to **Public**.
5. Select **Docker** as the SDK.
6. Choose the free **CPU Basic** hardware.
7. Create the Space.

Hugging Face creates a small Space repository containing a `README.md`. Preserve
that file because its YAML header configures the Docker runtime.

The header should look like:

```yaml
---
title: Finger Lakes Inpatient Opportunity Analyzer
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Case-mix-adjusted LOS and cost opportunity analysis
---
```

## 2. Copy the application into the Space repository

Run these commands from the directory that contains this project. Replace
`YOUR_HF_USERNAME` with the username shown in the Hugging Face profile URL.

```bash
git clone https://huggingface.co/spaces/YOUR_HF_USERNAME/los-opportunity-analyzer hf-space

cp -R los-pjt/backend hf-space/backend
cp -R los-pjt/frontend hf-space/frontend
cp los-pjt/Dockerfile hf-space/Dockerfile

cd hf-space
git add README.md Dockerfile backend frontend
git commit -m "Deploy opportunity analyzer"
git push
```

If the local project folder has a different name, replace `los-pjt` in the copy
commands with its path.

## 3. Authentication

When Git requests a password, use a Hugging Face **write access token**, not the
account password:

1. Open <https://huggingface.co/settings/tokens>.
2. Create a fine-grained or write token that can modify the Space.
3. Enter the Hugging Face username when Git asks for a username.
4. Paste the token when Git asks for a password.

Never put the token in a file, command, Git remote URL, commit, screenshot, or
chat message. Revoke it from the token settings if it is accidentally exposed.

## 4. Build and live URL

Each push triggers a Docker rebuild. Follow the build logs from the Space page.
After the status changes to **Running**, the public URLs are:

```text
Space page: https://huggingface.co/spaces/YOUR_HF_USERNAME/los-opportunity-analyzer
Direct app: https://YOUR_HF_USERNAME-los-opportunity-analyzer.hf.space
```

Hugging Face normalizes some characters in the direct-app subdomain. Use the
link displayed by the Space page as the authoritative URL.

## Why only selected files are copied

- The raw SPARCS CSV is not required and is not uploaded.
- The trained `.joblib` artifacts and aggregate JSON are already in `backend/`.
- The generated Space `README.md` retains the required Docker metadata.
- GitHub workflows, Git history, local virtual environments, and tests are not
  required in the runtime repository.

## Troubleshooting

### Space shows a configuration error

Confirm that the Space `README.md` begins with `sdk: docker` and
`app_port: 7860`.

### Container starts but the page is unavailable

Confirm that the Docker command binds to `0.0.0.0` on port `7860`. The supplied
Dockerfile already does this.

### Build is slow

The first build installs pandas and scikit-learn and may take several minutes.
Later builds can reuse cached layers when requirements do not change.

### App sleeps

Sleeping is expected on free CPU hardware. The first visit after inactivity can
take extra time while the Space wakes.
