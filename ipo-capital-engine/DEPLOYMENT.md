# Deploying the beta

Three ways to run the app, in order of how quickly they get testers to a URL.

The app is stateless: nothing is written to disk, there is no database, no
secrets, and no outbound network calls. Every session's inputs live only in that
browser session's memory. That keeps deployment simple and means there is
nothing to back up.

---

## 1. Streamlit Community Cloud (recommended for a beta)

Free, connects straight to this repository, redeploys on every push to `main`.
This is the path the repository is configured for.

1. Go to <https://share.streamlit.io> and sign in with the GitHub account that
   owns this repository.
2. **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository**: `thakerswapnil24-gif/IPO-Financing`
   - **Branch**: `main`
   - **Main file path**: `ipo-capital-engine/app.py`
   - **App URL**: choose the subdomain testers will use.
4. Open **Advanced settings** and set **Python version** to **3.12**. CI tests
   3.11 and 3.12; pick 3.12 to match what the container uses.
5. **Deploy**. The first build installs `ipo-capital-engine/requirements.txt`
   and takes a few minutes; later pushes redeploy in well under a minute.

Notes:

- `requirements.txt` holds runtime dependencies only. Test tooling lives in
  `requirements-dev.txt`, so the deployment does not install pytest.
- `.streamlit/config.toml` pins the light theme, turns off usage-stat gathering
  and keeps error details visible, which testers are asked to paste into bug
  reports.
- Anyone with the URL can use a public app. There is nothing sensitive in it —
  no data is stored and no credentials exist — but if you want the beta limited
  to invited testers, deploy it as a private app instead and add their GitHub
  addresses under **Settings → Sharing**.
- Watch the resource ceiling. A Community Cloud app has about 1 GB of memory;
  a 100,000-path Monte Carlo is comfortably inside that, but it is the heaviest
  thing the app does.

## 2. Docker (self-hosting)

From the `ipo-capital-engine` directory:

```bash
docker build -t ipo-capital-engine:0.1.0b1 .
docker run --rm -p 8501:8501 ipo-capital-engine:0.1.0b1
```

Then open <http://localhost:8501>.

The image runs as an unprivileged user, pins Python 3.12, installs runtime
dependencies only, and declares a `HEALTHCHECK` against Streamlit's
`/_stcore/health` endpoint. CI builds this image and boots a container on every
push, so a broken Dockerfile fails the build.

To put it behind a reverse proxy, forward to port 8501 and make sure WebSocket
upgrades are passed through — Streamlit needs them, and a proxy that only
forwards plain HTTP will render a blank page that never finishes loading.

## 3. Local

```bash
cd ipo-capital-engine
pip install -r requirements-dev.txt
streamlit run app.py
```

---

## Verifying a deployment

Whatever the target, two checks tell you it is really up:

```bash
curl -f http://<host>:<port>/_stcore/health     # -> ok
curl -o /dev/null -w '%{http_code}\n' http://<host>:<port>/   # -> 200
```

The health endpoint answers as soon as the server is listening; loading the root
page in a browser is what proves the app script itself runs. CI does both on
every push, against a real server and a real container.

## Releasing a new beta build

1. Bump `__version__` in `ipo-capital-engine/version.py`.
2. Add an entry to `CHANGELOG.md`.
3. Merge to `main`. Streamlit Community Cloud redeploys automatically; a
   self-hosted container needs a rebuild.
4. Tag the commit so a build can be traced back to its source:
   ```bash
   git tag -a v0.1.0b1 -m "Beta 1" && git push origin v0.1.0b1
   ```

The running version is shown as a badge beside the dashboard title, in the page
title, in the footer, and stamped into every exported report — so a tester's bug
report always identifies the build it came from.
