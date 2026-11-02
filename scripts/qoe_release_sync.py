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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="Published stable release tag from GitHub API")
    args = parser.parse_args()
    tag = args.tag
    if not TAG_RE.fullmatch(tag):
        output("blocked", tag, "Tag is not a stable semantic version")
        return 0
    if not MARKER.is_file():
        output("blocked", tag, "Bootstrap required: .qoe-upstream-tag is missing")
        return 0
    old_tag = MARKER.read_text(encoding="utf-8").strip()
    if not TAG_RE.fullmatch(old_tag):
        output("blocked", tag, "Invalid integrated-release marker")
        return 0
    if git("status", "--porcelain").stdout.strip():
        output("blocked", tag, "Checkout is not clean")
        return 0
    if tag == old_tag:
        output("unchanged", tag)
        return 0

    # Fetch only the two exact upstream tag refs. A release name is validated
    # above, and git refuses to replace an existing tag that changed remotely.
    for version in (old_tag, tag):
        result = git(
            "fetch", "--no-tags", "upstream",
            f"refs/tags/{version}:refs/tags/{version}", check=False,
        )
        if result.returncode:
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
