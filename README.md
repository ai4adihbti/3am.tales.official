# 🕯️ Night Stories: a free, fully automated multi-agent Instagram Reels system

Every day, a team of AI agents researches, writes, critiques, narrates, edits and posts a
horror/mystery story Reel to your Instagram, then studies how it performed and gets better.
Everything runs for free on GitHub Actions; no computer needs to stay on.

```
 Analyst ─▶ Researcher ─▶ Writer ⇄ Critic ─▶ Voice ─▶ Visual Director ─▶ Editor ─▶ Caption ─▶ Publisher
    ▲                                                                                          │
    └──────────────────────────────── Memory (state/history.json) ◀───────────────────────────┘
```

| Agent | Job |
|---|---|
| **Analyst** | Pulls views, shares, saves and comments for past Reels; writes "what works / what flops" lessons |
| **Researcher** | Picks today's format (rotates, then favours winners) and a fresh idea that has never been used |
| **Writer** | Writes a 45–60 s script: scroll-stopping hook, escalating scenes, twist ending |
| **Critic** | Independent editor: scores hook, tension, twist, flow and safety; sends weak scripts back |
| **Voice** | Deep neural narrator plus a studio-style mastering chain (EQ, compression, room ambience, 48 kHz) |
| **Visual Director** | Finds **real camera footage** for every scene: vertical, Full HD or better, never reused |
| **Editor** | Cuts footage to the narration, crossfades, word-timed subtitles, eerie drone, −14 LUFS master |
| **Caption** | Caption with a comment-bait question, call to action and niche hashtags |
| **Publisher** | Posts through the **official** Instagram API (no passwords, no scraping, no bans) |

**Output quality:** 1080×1920, 30 fps, H.264 High at CRF 18 (slow preset), AAC 256 kbps 48 kHz stereo,
loudness-normalised to −14 LUFS, Instagram's playback level.

---

## What it costs: ₹0

| Piece | Free option used |
|---|---|
| AI brain | Google Gemini API free tier (a few calls a day; far below the limits) |
| Footage | Pixabay API, free with an instant key; clips are free for commercial use with no attribution needed |
| Voice | Microsoft neural voices through `edge-tts`, free with no key |
| Editing | ffmpeg, open source |
| Posting and analytics | Official Instagram API, free |
| Scheduler and server | GitHub Actions, free (a private repo gets 2,000 min/month; this uses about 300) |

---

## Setup (about 30–40 minutes, one time)

### 1. Make your Instagram a Professional account
Instagram app → **Settings → Account type and tools → Switch to professional account → Creator**.
It's free and takes a minute. The posting API doesn't work with personal accounts.

### 2. Get your Instagram token (Meta developer app)
1. Go to **developers.facebook.com**, log in, then **My Apps → Create app**.
2. Choose the use case **"Manage messaging & content on Instagram"** (this is the *Instagram API with Instagram Login*; no Facebook Page needed).
3. In the app, open **Instagram → API setup with Instagram login**.
4. Under **Generate access tokens**, click **Add account** and log in with your Instagram account.
   If asked, also add yourself under **App roles → Roles → Instagram Testers**, then accept the invite in the
   Instagram app (**Settings → Apps and websites → Tester invites**).
5. Make sure these permissions are enabled: `instagram_business_basic`,
   `instagram_business_content_publish`, `instagram_business_manage_insights`.
6. Click **Generate token** and copy it. This is your **`IG_ACCESS_TOKEN`** (valid 60 days).
7. Find your **`IG_USER_ID`**: it's shown next to the account on the same page, or run
   `IG_ACCESS_TOKEN=... python scripts/check_token.py`, which prints it.

> You don't need Meta's App Review: an app in Development mode can post to **your own** account.

### 3. Get the two free API keys
- **Gemini:** aistudio.google.com → **Get API key** → **Create API key**. This is `GEMINI_API_KEY`.
- **Pixabay (footage):** sign up free at pixabay.com, then open **pixabay.com/api/docs**. Your key is shown
  in the *Parameters* section under `key` once you're logged in. This is `PIXABAY_API_KEY`.
  (Already have a Pexels key? You can add it as `PEXELS_API_KEY` instead. Pexels has paused new keys.)

### 4. Put the project on GitHub
1. Create a free GitHub account, then **New repository** → name it `night-stories` → choose **Private** → Create.
2. Upload all the files in this folder, including the hidden **`.github`** folder (it holds the daily schedule).
   - Easiest: install **GitHub Desktop**, clone the empty repo, copy these files into it, then Commit and Push.
   - Web upload sometimes skips hidden folders. If `.github/workflows/daily-reel.yml` is missing after uploading,
     use **Add file → Create new file**, type the path `.github/workflows/daily-reel.yml` and paste in the file's contents.

### 5. Add your secrets
Repo → **Settings → Secrets and variables → Actions → New repository secret**. Add these four:

| Name | Value |
|---|---|
| `GEMINI_API_KEY` | from step 3 |
| `PIXABAY_API_KEY` | from step 3 |
| `IG_ACCESS_TOKEN` | from step 2 |
| `IG_USER_ID` | from step 2 |

Optional: `GEMINI_MODEL` (to force a specific model) and `IG_API_HOST` (set to `graph.facebook.com`
only if you used the Facebook-Login version of the API).

### 6. Test it (nothing gets posted)
Repo → **Actions** tab → enable workflows if asked → **Daily horror reel** → **Run workflow**.
Leave **dry run** ticked and click Run. When it finishes (about 8–12 minutes), open the run and download
**reel-N** under **Artifacts**. It contains the video plus a JSON file with the script, the Critic's score and the caption.

### 7. Go live
Run it once more with **dry run unticked**, which posts for real. From then on it posts **every day at
about 6:47 PM IST** with no input from you. To change the time, edit the `cron` line in
`.github/workflows/daily-reel.yml` (times are in UTC; IST = UTC + 5:30).

### 8. Once a month: refresh the token
Instagram tokens expire after 60 days. Monthly, run:
```
IG_ACCESS_TOKEN=your_current_token python scripts/check_token.py --refresh
```
and paste the new token into the `IG_ACCESS_TOKEN` secret. If you forget, the daily run fails and
GitHub emails you, so you'll know.

---

## Customising (edit `config.yaml`)
- **Niche and audience:** `channel` and `formats`. Add or replace formats, e.g. a true-crime-free "dark history" format.
- **Voice:** `voice.name`. Try `en-US-GuyNeural`, `en-GB-RyanNeural` or `en-US-AndrewNeural`; adjust `rate` and `pitch`.
- **Look:** `video.grade` (colour), `subtitle_size`, `hook_color`, `drone_volume`.
- **Story length and quality bar:** `story.min_words`, `max_words`, `min_critic_score`.
- **Posting twice a day:** add a second `- cron:` line in the workflow file.

## Running on your own computer (optional)
Needs Python 3.10+ and ffmpeg.
```
pip install -r requirements.txt
python main.py --mock       # offline test, no keys needed (placeholder footage and tone)
python main.py --dry-run    # real story, voice and footage, saved to out/ and not posted
python main.py              # create and post
```
Set the keys as environment variables (see `.env.example`).

## Troubleshooting
| Problem | Fix |
|---|---|
| `missing secrets` | A secret name is misspelled or wasn't added (step 5) |
| Gemini `HTTP 429` | Free-tier rate limit; the system retries automatically. If it persists, set `GEMINI_MODEL` to `gemini-2.5-flash-lite` |
| Gemini `HTTP 404` on every model | Google renamed its models: put a current Flash model name from aistudio.google.com into `GEMINI_MODEL` |
| `create container failed` / `OAuthException` | Token expired or missing a permission; regenerate it (step 2) |
| `Instagram could not process the video` | Rare; it usually succeeds on the next day's run. Check the run's artifact plays correctly |
| Voice missing in a reel | The free voice service was briefly down; the reel falls back to subtitles plus ambience |
| Workflow didn't run on schedule | GitHub can delay scheduled runs by a few minutes; the Actions tab shows every run |

## Ground rules built in
- Official API only: no passwords stored, no scraping, no follow/like/DM bots, so your account stays safe.
- The Critic rejects gore, self-harm, real victims, named private people and anything unsafe for a 16+ audience.
- Fiction is labelled "(fiction)" in captions; the real-mystery format must be accurate.
- Instagram allows up to 100 API posts a day; this posts 1.
