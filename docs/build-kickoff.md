# personalive-blackwell: joining the BUILD office

The owner (james@chappa.ai) added the existing repo chappa-ai-llc/personalive-blackwell to the office on 2026-11-08. This file is the first session's brief.

## Design doc

The owner did not name it: look in README.md and `docs/` for the design document.

## About

Not given: write `about` (what, why, who) in `build/board.json` from the design doc.

## Owner's notes

Adopted from the Security page on 2026-11-08, because of its security findings.

## Security when adopted

The office's dependency scan of 2026-11-08T19:11:12Z (dep-audit 0.16.5) read chappa-ai-llc/personalive-blackwell at main@b2cd648:

- Advisories: 373 open (12 critical, 156 high, 147 medium, 38 low, 20 unknown), 0 excepted.
- Dependabot: 320 open alerts.
- Unmaintained or archived direct dependencies: 2.

## First steps

Work in this order:

1. Read the global rules and the playbook's chapters for a new project (MDD first, tools before code).
2. Read the design doc and the rest of the repo.
3. Write the MDD, `docs/MDD-personalive-blackwell.md`, from the design doc (or make the design doc the MDD if it already is one). List it under `docs` in `.build-board.json`.
4. Make sure `scratchpad/` is in `.gitignore`. Write or update README.md from the MDD.
5. Fill `build/board.json`: `about`, the timeline, and a card for everything only the owner can do or decide. Push it: `build_board.py push`.
6. Security first: `build_board.py security --phases`, then `build_board.py push`. Work the critical and high rows before anything else (due 2 and 7 days from first seen).
7. Bring the repo up to the office standards: work through `/srv/tool-bin/build-onboard/STANDARDS.md` (security self-check for a web front end, Dependabot alerts, BUILD link, no npm, no Actions, secrets, release by tag). Fix what is small now; file an issue and a card for the rest.
8. Report in on the forum: a topic whose title starts with "onboarded: personalive-blackwell", addressed to owner. Say which standards are met and which are filed.
9. Remove `kickoff` from `.build-board.json` and this file from its `docs`, commit, and push.

Board: https://build.chappa.ai/p/personalive-blackwell
