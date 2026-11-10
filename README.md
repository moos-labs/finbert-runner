# finbert-runner

Runs **FinBERT** (`ProsusAI/finbert`) financial-sentiment analysis via GitHub
Actions — part of a one-public-repo-per-tool pattern: each tool gets its own
repo, the commands to execute live in a repository variable, runs are triggered
manually, and outputs are collected as artifacts.

## How to run an analysis

1. **Edit the commands**: repo **Settings → Secrets and variables → Actions →
   Variables** → set `COMMANDS` to the shell commands you want executed, e.g.
   ```
   mkdir -p results
   python scripts/finbert_sentiment.py --input samples/poul_headlines.txt --label poul-pilot 2>&1 | tee results/run.log
   ```
2. **Trigger manually**: **Actions** tab → *Run analysis* → *Run workflow*
   (or dispatch it through the GitHub API).
3. **Collect results**: download the `results` artifact from the finished run.
   It contains `sentiment.json` (per-item labels + probabilities, label counts,
   mean probabilities), `report.md`, and the run log.

## Design notes

- **Generic and multi-market.** `scripts/finbert_sentiment.py` scores arbitrary
  English financial texts — any stock, any market, any news source. Nothing in
  the workflow or script is tied to one exchange; the POUL headlines in
  `samples/` are only a pilot input.
- **Cheap repeat runs.** The pip cache plus an `actions/cache` entry for
  `~/.cache/huggingface` (key `finbert-model-v1`) mean later runs skip both the
  package downloads and the ~440MB model download.
- **CPU only.** `requirements.txt` pulls torch from the PyTorch CPU wheel index.
- **English only.** FinBERT was fine-tuned on English financial text. Arabic
  headlines must be translated before scoring (the pilot headlines are English
  renderings of Arabic news items).
- **Licence note.** The FinBERT training code is Apache-2.0, but since
  October 2026 the pre-trained **weights** sit under the Hugging Face
  model-card terms, not plain Apache-2.0.
