"""Prepare a review branch by merging a published upstream Dispatcharr release.

Run from a clean checkout of the fork's default QoE branch. This script never
pushes, opens a PR, or changes production; the workflow handles those actions.
"""

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys


MARKER = Path(".qoe-upstream-tag")
DEV_BASE = "7bf5cff4be4697c629faaf267575761e48142705"
QOE_BRANCH = "feat/qoe-observability"
TAG_RE = re.compile(r"v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\Z")


def git(*args, check=True):
    return subprocess.run(["git", *args], text=True, capture_output=True, check=check)


def output(status, tag, detail=""):
    print(f"status={status} tag={tag} {detail}".strip())
    destination = os.environ.get("GITHUB_OUTPUT")
    if destination:
        with open(destination, "a", encoding="utf-8") as stream:
            stream.write(f"status={status}\nrelease_tag={tag}\n")
            if status == "updated":
                stream.write(f"candidate_branch=qoe-upstream-{tag}\n")


def is_ancestor(older, newer):
    return git("merge-base", "--is-ancestor", older, newer, check=False).returncode == 0


def fetch_tag(tag):
    result = git(
        "fetch", "--no-tags", "upstream",
        f"refs/tags/{tag}:refs/tags/{tag}", check=False,
    )
    return result.returncode == 0


def bootstrap(tag):
    """Replay the fixed QoE development branch onto the first compatible tag."""
    if not fetch_tag(tag):
        output("blocked", tag, "Could not fetch upstream release tag")
        return 0
    release = git("rev-parse", f"refs/tags/{tag}^{{commit}}").stdout.strip()
    if not is_ancestor(DEV_BASE, release):
        output("waiting", tag, "Release does not contain the pinned dev base")
        return 0
    fetched = git(
        "fetch", "origin", f"refs/heads/{QOE_BRANCH}:refs/remotes/origin/{QOE_BRANCH}",
        check=False,
    )
    if fetched.returncode:
        output("blocked", tag, "Could not fetch QoE development branch")
        return 0
    qoe_head = f"refs/remotes/origin/{QOE_BRANCH}"
    if not is_ancestor(DEV_BASE, qoe_head):
        output("blocked", tag, "QoE branch no longer descends from pinned dev base")
        return 0
    commits = git("rev-list", "--reverse", f"{DEV_BASE}..{qoe_head}").stdout.splitlines()
    if not commits:
        output("blocked", tag, "QoE branch has no commits to replay")
        return 0
    branch = f"qoe-upstream-{tag}"
    if git("show-ref", "--verify", f"refs/heads/{branch}", check=False).returncode == 0:
        output("blocked", tag, "Candidate branch already exists locally")
        return 0
    # Preserve the scheduler/bootstrap implementation from the fork default
    # branch; the development branch may contain an older copy of these files.
    automation_paths = (
        "scripts/qoe_release_sync.py", ".github/workflows/qoe-release-sync.yml",
    )
    automation = {path: git("show", f"HEAD:{path}").stdout for path in automation_paths}
    git("switch", "-c", branch, release)
    for commit in commits:
        picked = git("cherry-pick", commit, check=False)
        if picked.returncode:
            git("cherry-pick", "--abort", check=False)
            output("conflict", tag, f"QoE cherry-pick conflict at {commit[:12]}")
            return 0
    for path, content in automation.items():
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    MARKER.write_text(tag + "\n", encoding="utf-8")
    git("add", str(MARKER), *automation_paths)
    git("commit", "-m", f"chore(qoe): bootstrap on Dispatcharr {tag}")
    output("updated", tag, f"candidate={branch} upstream_sha={release}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="Published stable release tag from GitHub API")
    args = parser.parse_args()
    tag = args.tag
    if not TAG_RE.fullmatch(tag):
        output("blocked", tag, "Tag is not a stable semantic version")
        return 0
    if git("status", "--porcelain").stdout.strip():
        output("blocked", tag, "Checkout is not clean")
        return 0
    if not MARKER.is_file():
        return bootstrap(tag)
    old_tag = MARKER.read_text(encoding="utf-8").strip()
    if not TAG_RE.fullmatch(old_tag):
        output("blocked", tag, "Invalid integrated-release marker")
        return 0
    if tag == old_tag:
        output("unchanged", tag)
        return 0

    # Fetch only the two exact upstream tag refs. A release name is validated
    # above, and git refuses to replace an existing tag that changed remotely.
    for version in (old_tag, tag):
        if not fetch_tag(version):
            output("blocked", tag, f"Could not fetch upstream {version}")
            return 0
    old_commit = git("rev-parse", f"refs/tags/{old_tag}^{{commit}}").stdout.strip()
    new_commit = git("rev-parse", f"refs/tags/{tag}^{{commit}}").stdout.strip()
    head = git("rev-parse", "HEAD").stdout.strip()
    if not is_ancestor(old_commit, new_commit):
        output("blocked", tag, "New release does not descend from integrated release")
        return 0
    if not is_ancestor(old_commit, head):
        output("blocked", tag, "Fork branch does not descend from integrated release")
        return 0

    branch = f"qoe-upstream-{tag}"
    if git("show-ref", "--verify", f"refs/heads/{branch}", check=False).returncode == 0:
        output("blocked", tag, "Candidate branch already exists locally")
        return 0
    git("switch", "-c", branch)
    result = git("merge", "--no-ff", "--no-commit", new_commit, check=False)
    if result.returncode:
        git("merge", "--abort", check=False)
        output("conflict", tag, "Upstream merge conflict; QoE branch is unchanged")
        return 0
    MARKER.write_text(tag + "\n", encoding="utf-8")
    git("add", str(MARKER))
    git("commit", "-m", f"chore(qoe): sync Dispatcharr {tag}")
    output("updated", tag, f"candidate={branch} upstream_sha={new_commit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
