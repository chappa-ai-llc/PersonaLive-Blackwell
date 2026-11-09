"""Tests for deploy/release_page.py, with a fake GitHub (no network)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))

import release_page  # noqa: E402

CHANGELOG = """# Changelog

## [Unreleased]

## [0.2.1] - 2026-11-09

**Purpose:** documentation only.

## [0.2.0] - 2026-11-08

**Purpose:** appearance.

## [0.1.0] - 2026-11-08

**Purpose:** the first release.
"""


class FakeGitHub:
    repo = "o/r"

    def __init__(self, tags, have):
        self.tags, self.have, self.made = tags, set(have), []

    def release_tags(self):
        return list(self.tags)

    def tags_with_pages(self):
        return set(self.have)

    def make(self, tag, body, latest):
        self.made.append((tag, body, latest))
        self.have.add(tag)
        return f"https://example.test/{tag}"


class ReleasePageTest(unittest.TestCase):
    def run_page(self, target, gh, text=CHANGELOG):
        return release_page.run(target, gh, text, out=lambda *_: None)

    def test_section_is_the_named_version_only(self):
        self.assertEqual(release_page.changelog_section(CHANGELOG, "0.2.0"), "**Purpose:** appearance.")
        self.assertIsNone(release_page.changelog_section(CHANGELOG, "0.2"))
        self.assertIsNone(release_page.changelog_section(CHANGELOG, "9.9.9"))

    def test_backfills_older_tags_oldest_first_and_marks_only_the_target_latest(self):
        gh = FakeGitHub(["v0.2.1", "v0.2.0", "v0.1.0", "not-a-release"], have=[])
        self.assertEqual(self.run_page("v0.2.1", gh), 0)
        self.assertEqual([(t, latest) for t, _, latest in gh.made],
                         [("v0.1.0", False), ("v0.2.0", False), ("v0.2.1", True)])

    def test_body_has_the_notes_and_how_to_verify(self):
        gh = FakeGitHub(["v0.2.1"], have=[])
        self.run_page("v0.2.1", gh)
        body = gh.made[0][1]
        self.assertTrue(body.startswith("**Purpose:** documentation only."))
        self.assertIn("tag -v v0.2.1", body)
        self.assertNotIn("0.2.0", body)

    def test_second_run_changes_nothing(self):
        gh = FakeGitHub(["v0.2.1", "v0.1.0"], have=["v0.2.1", "v0.1.0"])
        self.assertEqual(self.run_page("v0.2.1", gh), 0)
        self.assertEqual(gh.made, [])

    def test_newer_tags_are_not_touched(self):
        gh = FakeGitHub(["v0.3.0", "v0.2.1"], have=[])
        self.run_page("v0.2.1", gh)
        self.assertEqual([t for t, _, _ in gh.made], ["v0.2.1"])

    def test_unknown_tag_and_missing_section_are_refused(self):
        with self.assertRaises(release_page.Refused):
            self.run_page("v0.2.1", FakeGitHub(["v0.2.0"], have=[]))
        with self.assertRaises(release_page.Refused):
            self.run_page("v9.9.9", FakeGitHub(["v9.9.9"], have=[]))

    def test_older_tag_without_a_section_is_skipped(self):
        gh = FakeGitHub(["v0.2.1", "v0.0.9"], have=[])
        self.assertEqual(self.run_page("v0.2.1", gh), 0)
        self.assertEqual([t for t, _, _ in gh.made], ["v0.2.1"])

    def test_bad_arguments(self):
        self.assertEqual(release_page.main(["release_page.py"]), 2)
        self.assertEqual(release_page.main(["release_page.py", "0.2.1"]), 2)


if __name__ == "__main__":
    unittest.main()
