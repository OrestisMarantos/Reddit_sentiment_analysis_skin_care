import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

#------------
# -----------------------------
# Helpers
# -----------------------------
DISALLOWED_USERNAMES = {"Share", "Report", "Award", "Upvote", "Downvote"}
# Default directory containing .txt thread dumps. Change these constants if your folders differ.
DEFAULT_INPUT_DIR = Path(__file__).parent / "txt_threads"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "parsed_json"

def parse_txt_file_to_json(txt_path: str | Path, output_dir: str | Path) -> Path:
    txt_path = Path(txt_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = parse_reddit_thread_txt(txt_path)
    out_path = output_dir / f"{txt_path.stem}.parsed.json"
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def parse_folder_to_json(category_folder: str | Path, output_dir: str | Path) -> List[Path]:
    category_folder = Path(category_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_paths = []
    txt_files = sorted(category_folder.glob("*.txt"))

    for txt_file in txt_files:
        out_path = parse_txt_file_to_json(txt_file, output_dir)
        json_paths.append(out_path)

    return json_paths

def clean_lines(text: str) -> List[str]:
    lines = [ln.rstrip("\n").rstrip("\r").rstrip() for ln in text.splitlines()]
    # collapse repeated blanks
    out = []
    blank = False
    for ln in lines:
        if ln.strip() == "":
            if not blank:
                out.append("")
            blank = True
        else:
            out.append(ln)
            blank = False
    return out

def parse_int(s: str) -> Optional[int]:
    m = re.search(r"(-?\d+)", s)
    return int(m.group(1)) if m else None

def is_username_line(s: str) -> bool:
    s = s.strip()
    if not s or s in DISALLOWED_USERNAMES:
        return False
    return bool(re.match(r"^(u\/)?[A-Za-z0-9_\-]+$|^\[deleted\]$", s))

def looks_like_age_line(s: str) -> bool:
    # "5y ago", "2d ago", "1mo ago" etc.
    return bool(re.match(r"^\s*(edited\s+)?\d+\s*(y|yr|yrs|mo|d|h|m)\s+ago\s*$", s, re.I)) or \
           bool(re.match(r"^\s*(edited\s+)?\d+(y|mo|d|h|m)\s+ago\s*$", s, re.I))

def extract_age_and_edited(age_line: str) -> Tuple[Optional[str], bool]:
    s = age_line.strip().lower()
    edited = "edited" in s
    # normalize: "5y" from "5y ago", "5 y ago", etc.
    m = re.search(r"(\d+)\s*(y|yr|yrs|mo|d|h|m)", s)
    if not m:
        return None, edited
    num, unit = m.group(1), m.group(2)
    unit_norm = {
        "yr": "y", "yrs": "y"
    }.get(unit, unit)
    return f"{num}{unit_norm}", edited

# vote lines: your paste often has:
# Upvote
# 74
# Downvote
# (maybe number, maybe blank)
def find_vote_value(lines: List[str], idx: int) -> Tuple[Optional[int], int]:
    """
    If lines[idx] is 'Upvote' or 'Downvote', try to read a number from the next 1-2 lines.
    Returns (value_or_none, lines_consumed)
    """
    for look in range(1, 3):
        if idx + look < len(lines):
            val = parse_int(lines[idx + look])
            if val is not None:
                return val, look + 1
    return None, 1

UI_SINGLE_WORD = {"Award", "Share", "Report"}

# -----------------------------
# Comment start detector
# -----------------------------

def is_comment_start(lines: List[str], i: int) -> bool:
    """
    A comment usually looks like:
      author
      •
      5y ago
    or
      author
      •
      Edited 5y ago
    Sometimes "u/name avatar" appears before the author line.
    """
    if i < 0 or i >= len(lines):
        return False

    author = lines[i].strip()
    if not is_username_line(author):
        return False

    # scan next few lines for bullet + age
    # accept either:
    #  author, "•", "5y ago"
    #  author, "•", "Edited 5y ago"
    # or sometimes "•" and "5y ago" could be merged by paste, so check for age in next 4 lines too.
    bullet_found = False
    for j in range(i + 1, min(i + 6, len(lines))):
        s = lines[j].strip()
        if s == "•":
            bullet_found = True
            continue
        if bullet_found and looks_like_age_line(s):
            return True
        # fallback: if paste merges bullet/age weirdly
        if "•" in s and looks_like_age_line(s.replace("•", "").strip()):
            return True

    return False

def skip_avatar_lines(lines: List[str], i: int) -> int:
    # sometimes there is a line like "u/TheWaywardTrout avatar"
    if i < len(lines) and "avatar" in lines[i].lower():
        return i + 1
    return i

# -----------------------------
# Main parser
# -----------------------------

def parse_reddit_thread_txt(path: str) -> Dict[str, Any]:
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = clean_lines(raw)

    # thread title: first non-empty line that isn't "Discussion"
    title = None
    for ln in lines[:80]:
        if ln.strip() and ln.strip().lower() != "discussion":
            title = ln.strip()
            break

    go_to_comments_present = any("Go to comments" in ln for ln in lines[:250])

    # find first comment index
    first_comment_idx = None
    for i in range(len(lines)):
        if is_comment_start(lines, i):
            first_comment_idx = i
            break

    # post body between title and first comment (optional)
    post_body = ""
    if first_comment_idx is not None and title in lines:
        t_idx = lines.index(title)
        body_slice = lines[t_idx + 1:first_comment_idx]
        # remove obvious UI clutter
        cleaned = []
        for ln in body_slice:
            s = ln.strip()
            if not s:
                continue
            if s.lower() in {"upvote", "downvote", "discussion", "report", "share"}:
                continue
            if s.isdigit():
                continue
            cleaned.append(ln)
        post_body = "\n".join(cleaned).strip()

    comments: List[Dict[str, Any]] = []
    i = first_comment_idx or 0

    while i < len(lines):
        if not is_comment_start(lines, i):
            i += 1
            continue

        author = lines[i].strip()

        # parse metadata lines after author: look for bullet then age line
        j = i + 1
        j = skip_avatar_lines(lines, j)

        # find bullet
        while j < len(lines) and lines[j].strip() != "•":
            # sometimes there are blank lines
            if lines[j].strip() == "":
                j += 1
                continue
            # If something unexpected, break to avoid runaway
            j += 1

        if j >= len(lines):
            break

        # age line should be after bullet
        j += 1
        while j < len(lines) and lines[j].strip() == "":
            j += 1

        age = None
        edited = False
        if j < len(lines) and looks_like_age_line(lines[j].strip()):
            age, edited = extract_age_and_edited(lines[j])
            j += 1

        # now comment content block goes until next comment start
        block_start = j
        k = block_start
        while k < len(lines) and not is_comment_start(lines, k):
            k += 1
        block = lines[block_start:k]

        upvotes = None
        downvotes = None
        awards_present = False
        share_present = False
        report_present = False

        text_lines: List[str] = []
        b = 0
        while b < len(block):
            s = block[b].strip()

            if s == "":
                if text_lines and text_lines[-1] != "":
                    text_lines.append("")
                b += 1
                continue

            # Upvote / Downvote patterns like:
            # Upvote \n 74
            if s.lower() == "upvote":
                val, consumed = find_vote_value(block, b)
                upvotes = val
                b += consumed
                continue

            if s.lower() == "downvote":
                val, consumed = find_vote_value(block, b)
                downvotes = val
                b += consumed
                continue

            # flags
            if s in UI_SINGLE_WORD:
                if s == "Award":
                    awards_present = True
                elif s == "Share":
                    share_present = True
                elif s == "Report":
                    report_present = True
                b += 1
                continue

            # ignore pure UI lines
            if s.lower().startswith(("comments section", "search comments", "expand comment search", "sort by:", "best")):
                b += 1
                continue

            # treat everything else as comment text
            text_lines.append(block[b])
            b += 1

        text = "\n".join(text_lines).strip()

        comments.append({
            "author": author,
            "age": age,  # e.g. "5y"
            "edited": edited,
            "text": text,
            "upvotes": upvotes,
            "downvotes": downvotes,
            "awards_present": awards_present,
            "share_present": share_present,
            "report_present": report_present
        })

        i = k

    return {
        "thread": {
            "title": title,
            "post_body": post_body,
            "go_to_comments_present": go_to_comments_present,
            "source_file": str(path)
        },
        "comments": comments
    }


def __main__():
    """
    Parse every *.txt inside DEFAULT_INPUT_DIR and write *.parsed.json into DEFAULT_OUTPUT_DIR.
    No command-line args needed; adjust the constants at the top if your paths differ.
    """
    in_path = DEFAULT_INPUT_DIR
    out_dir = DEFAULT_OUTPUT_DIR

    if not in_path.exists():
        print(f"Input directory not found: {in_path.resolve()}")
        return
    if not in_path.is_dir():
        print(f"Configured path is not a directory: {in_path.resolve()}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(in_path.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in directory: {in_path.resolve()}")
        return

    for txt in txt_files:
        data = parse_reddit_thread_txt(txt)
        out_path = out_dir / f"{txt.stem}.parsed.json"
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Parsed {txt.name} -> {out_path.name} (comments: {len(data['comments'])})")


if __name__ == "__main__":
    __main__()
