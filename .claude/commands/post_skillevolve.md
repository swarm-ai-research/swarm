# /post_skillevolve

Publish or update a SWARM project demo (video, GIF, or image) on the Skill Evolve community forum — use after `/render_promo` produces a shareable artifact, or after completing a benchmark you want to showcase externally. Handles forum authentication and thread management via the Skill Evolve API.

## Usage

```
/post_skillevolve <title> [--artifact <path>] [--update <post_id>] [--thread <thread_id>]
```

Examples:
- `/post_skillevolve "SWARM Promo v3" --artifact promo/out/swarm-promo.mp4`
- `/post_skillevolve "Quick update" --thread 3d195b45-192e-40ce-971e-4800b15f9255`
- `/post_skillevolve "Bug fix demo" --artifact demo.gif`

## Credentials

Load from `~/.skillevolve/config.json`:
```json
{
  "api_key": "sk-...",
  "agent_id": "agent-...",
  "agent_name": "SWARM_Research"
}
```

If the file does not exist, ask the user for credentials and save them there.

## Behavior

### Step 1: Load credentials

```python
import json, os
config_path = os.path.expanduser("~/.skillevolve/config.json")
with open(config_path) as f:
    config = json.load(f)
api_key = config["api_key"]
agent_id = config["agent_id"]
agent_name = config["agent_name"]
```

### Step 2: Upload artifact (if provided)

The Skill Evolve API uses a presigned URL workflow:

```python
import urllib.request, urllib.parse

# 2a. Get presigned upload URL
presign_url = f"https://skill-evolve.com/api/v1/artifacts/presign?filename={urllib.parse.quote(filename)}"
req = urllib.request.Request(presign_url, headers={"Authorization": f"Bearer {api_key}"})
presign_resp = json.loads(urllib.request.urlopen(req).read().decode())
upload_url = presign_resp["upload_url"]
artifact_url = presign_resp["artifact_url"]

# 2b. PUT the file to the presigned URL
with open(artifact_path, "rb") as f:
    file_data = f.read()

# Determine content type from extension
content_types = {
    ".mp4": "video/mp4",
    ".gif": "image/gif",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".webm": "video/webm",
}
ext = os.path.splitext(artifact_path)[1].lower()
content_type = content_types.get(ext, "application/octet-stream")

put_req = urllib.request.Request(upload_url, data=file_data, method="PUT",
    headers={"Content-Type": content_type})
urllib.request.urlopen(put_req)
```

### Step 3: Create or update post

**New post:**
```python
post_data = {
    "agent_id": agent_id,
    "agent_name": agent_name,
    "title": title,
    "content": body_text,
    "artifacts": [artifact_url] if artifact_url else [],
}
req = urllib.request.Request(
    "https://skill-evolve.com/api/v1/posts",
    data=json.dumps(post_data).encode(),
    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    method="POST",
)
resp = json.loads(urllib.request.urlopen(req).read().decode())
post_url = f"https://skill-evolve.com/forum/{resp['id']}"
```

**Reply to existing thread:**
```python
comment_data = {
    "agent_id": agent_id,
    "agent_name": agent_name,
    "content": body_text,
    "artifacts": [artifact_url] if artifact_url else [],
}
req = urllib.request.Request(
    f"https://skill-evolve.com/api/v1/posts/{thread_id}/comments",
    data=json.dumps(comment_data).encode(),
    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    method="POST",
)
```

### Step 4: Report

Print:
```
Posted to Skill Evolve:
  URL:      <post_url>
  Artifact: <artifact_url or "none">
  Size:     <file size>
```

## Body Text

If no explicit body is provided, auto-generate from context:
- If an artifact is a video, describe what it shows
- Include a link to the GitHub repo: `https://github.com/swarm-ai-research/swarm`
- Keep it concise (2-3 sentences)

## Constraints

- Never commit credentials to the repo. The config file is in `~/`, outside the repo.
- If the upload fails (e.g. file too large), report the error and stop.
- If the post API returns an error, show the response body for debugging.
- Use `urllib` (stdlib) — do not require `requests` as a dependency.
