#!/usr/bin/env python3
"""Daily HCI paper delivery system.

Fetches recent papers from CHI/CSCW/TOCHI/UIST/IUI via Semantic Scholar
and OpenAlex, scores by quality and relevance, generates AI insights via
Claude Haiku, and creates a GitHub Issue.
"""

import json
import math
import os
import re
import sys
import time
import datetime
import urllib.request
import urllib.error
import urllib.parse
from html.parser import HTMLParser

# ── Configuration ─────────────────────────────────────────────────────────────

TARGET_VENUES = ["CHI", "CSCW", "TOCHI", "UIST", "IUI"]
YEAR_RANGE = (2023, 2026)

KEYWORDS_COGNITION = [
    "attention", "memory", "mental model", "perception",
    "cognitive load", "cognition", "psychology", "working memory",
    "decision making", "metacognition", "sensemaking",
]
KEYWORDS_AI_HUMAN = [
    "LLM", "large language model", "AI", "chatbot", "agent",
    "generative", "GPT", "human-AI", "conversational agent",
    "foundation model", "language model", "artificial intelligence",
]
KEYWORDS_HCI_GENERAL = [
    "user study", "interaction design", "usability",
    "user interface", "accessibility", "design",
]

SENT_PAPERS_PATH = "scripts/daily-papers/sent_papers.json"
SIGCHI_AWARDS_URL = "https://programs.sigchi.org/chi/2025/awards/"

QUERY_ROTATIONS = [
    "cognitive attention memory user CHI CSCW",
    "AI language model human interaction CHI UIST",
    "human factors design usability CHI",
    "mental model decision making HCI CSCW",
    "conversational agent AI chatbot interaction CHI",
    "accessibility inclusive design CHI UIST",
    "perception embodied interaction user study",
]

# ── HTTP Utilities ────────────────────────────────────────────────────────────


def http_get(url, headers=None, retries=3):
    req = urllib.request.Request(url, headers=headers or {})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def http_post_json(url, payload, headers=None):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


# ── SIGCHI Honorable Mention Scraping ─────────────────────────────────────────


class _TitleParser(HTMLParser):
    """Extracts paper titles from the SIGCHI awards page."""

    def __init__(self):
        super().__init__()
        self.titles = set()
        self._in_honorable = False
        self._depth = 0
        self._capture = False
        self._buf = ""

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        cls = attrs_dict.get("class", "") or ""
        _id = attrs_dict.get("id", "") or ""
        if "honorable" in cls.lower() or "honorable" in _id.lower():
            self._in_honorable = True
        if self._in_honorable and tag in ("h3", "strong"):
            self._capture = True
            self._buf = ""

    def handle_endtag(self, tag):
        if self._capture and tag in ("h3", "strong"):
            text = self._buf.strip()
            if len(text) > 20:
                self.titles.add(text.lower())
            self._capture = False
            self._buf = ""

    def handle_data(self, data):
        if self._capture:
            self._buf += data


def fetch_honorable_mentions():
    try:
        html = http_get(SIGCHI_AWARDS_URL).decode("utf-8", errors="ignore")
        parser = _TitleParser()
        parser.feed(html)
        return parser.titles
    except Exception as exc:
        print(f"Warning: SIGCHI scraping failed: {exc}", file=sys.stderr)
        return set()


# ── Semantic Scholar API ──────────────────────────────────────────────────────

SS_BASE = "https://api.semanticscholar.org/graph/v1"
SS_FIELDS = (
    "paperId,title,authors,year,venue,citationCount,"
    "abstract,externalIds,openAccessPdf"
)


def _ss_search(query, limit=50):
    url = (
        f"{SS_BASE}/paper/search"
        f"?query={urllib.parse.quote(query)}"
        f"&fields={SS_FIELDS}"
        f"&limit={limit}"
    )
    try:
        raw = http_get(url, headers={"Accept": "application/json"})
        return json.loads(raw).get("data", [])
    except Exception as exc:
        print(f"Warning: Semantic Scholar search failed: {exc}", file=sys.stderr)
        return []


def fetch_ss_papers():
    day_idx = datetime.date.today().weekday()
    rotated = QUERY_ROTATIONS[day_idx % len(QUERY_ROTATIONS)]
    anchor = "CHI 2024 2025 user study honorable mention"

    results = []
    for q in [rotated, anchor]:
        results.extend(_ss_search(q))
        time.sleep(1.2)  # respect rate limit
    return results


# ── OpenAlex API ──────────────────────────────────────────────────────────────


def _reconstruct_abstract(inverted_index):
    if not inverted_index:
        return ""
    tokens = []
    for word, positions in inverted_index.items():
        for pos in positions:
            tokens.append((pos, word))
    tokens.sort()
    return " ".join(w for _, w in tokens)


def fetch_openalex_papers():
    url = (
        "https://api.openalex.org/works"
        "?filter=primary_location.source.display_name.search:TOCHI,"
        "publication_year:2023-2025,type:journal-article"
        "&sort=cited_by_count:desc"
        "&per_page=20"
        "&select=id,title,authorships,publication_year,cited_by_count,"
        "abstract_inverted_index,primary_location,open_access"
    )
    try:
        raw = http_get(url, headers={"Accept": "application/json"})
        works = json.loads(raw).get("results", [])
        papers = []
        for w in works:
            authors = [
                a["author"]["display_name"]
                for a in w.get("authorships", [])[:5]
            ]
            abstract = _reconstruct_abstract(w.get("abstract_inverted_index") or {})
            loc = w.get("primary_location") or {}
            doi_url = loc.get("landing_page_url", "") or w.get("id", "")
            papers.append({
                "id": "oa_" + w["id"].split("/")[-1],
                "title": w.get("title", ""),
                "authors": authors,
                "year": w.get("publication_year", 0),
                "venue": "TOCHI",
                "citation_count": w.get("cited_by_count", 0),
                "abstract": abstract,
                "url": doi_url,
                "pdf_url": (w.get("open_access") or {}).get("oa_url"),
                "is_honorable_mention": False,
                "score": 0.0,
                "topic_tags": [],
            })
        return papers
    except Exception as exc:
        print(f"Warning: OpenAlex failed: {exc}", file=sys.stderr)
        return []


# ── Normalization ─────────────────────────────────────────────────────────────


def _normalize_ss(raw, hm_titles):
    """Convert Semantic Scholar result to internal paper dict."""
    if not raw.get("abstract"):
        return None

    title = raw.get("title", "")
    venue = raw.get("venue", "")
    year = raw.get("year") or 0

    venue_upper = venue.upper()
    if not any(v in venue_upper for v in TARGET_VENUES):
        return None
    if not (YEAR_RANGE[0] <= year <= YEAR_RANGE[1]):
        return None

    pid = raw.get("paperId", "")
    ext = raw.get("externalIds") or {}
    doi = ext.get("DOI")
    url = f"https://doi.org/{doi}" if doi else f"https://www.semanticscholar.org/paper/{pid}"
    pdf_url = (raw.get("openAccessPdf") or {}).get("url")

    return {
        "id": pid,
        "title": title,
        "authors": [a.get("name", "") for a in raw.get("authors", [])[:5]],
        "year": year,
        "venue": venue,
        "citation_count": raw.get("citationCount") or 0,
        "abstract": raw.get("abstract", ""),
        "url": url,
        "pdf_url": pdf_url,
        "is_honorable_mention": title.lower().strip() in hm_titles,
        "score": 0.0,
        "topic_tags": [],
    }


# ── Scoring ───────────────────────────────────────────────────────────────────


def score_paper(paper):
    score = 0.0

    if paper["is_honorable_mention"]:
        score += 10

    venue_upper = paper["venue"].upper()
    if any(v in venue_upper for v in ["CHI", "UIST", "CSCW", "IUI"]):
        score += 5
    elif "TOCHI" in venue_upper:
        score += 4

    year = paper["year"]
    if year >= 2025:
        score += 4
    elif year == 2024:
        score += 3
    elif year == 2023:
        score += 1

    citations = paper.get("citation_count") or 0
    score += min(math.log1p(citations) * 0.5, 3)

    text = (paper["title"] + " " + paper["abstract"][:300]).lower()
    tags = []
    cog_hits = sum(1 for kw in KEYWORDS_COGNITION if kw.lower() in text)
    ai_hits = sum(1 for kw in KEYWORDS_AI_HUMAN if kw.lower() in text)
    hci_hits = sum(1 for kw in KEYWORDS_HCI_GENERAL if kw.lower() in text)

    if cog_hits > 0:
        score += min(cog_hits, 3)
        tags.append("cognition")
    if ai_hits > 0:
        score += min(ai_hits, 3)
        tags.append("ai-human")
    if hci_hits > 0:
        score += min(hci_hits, 2)
        tags.append("hci")

    paper["topic_tags"] = tags
    paper["score"] = score
    return paper


# ── Paper Selection ───────────────────────────────────────────────────────────


def select_two(candidates):
    ranked = sorted(candidates, key=lambda p: p["score"], reverse=True)
    cog_pool = [p for p in ranked if "cognition" in p["topic_tags"]]
    ai_pool = [p for p in ranked if "ai-human" in p["topic_tags"]]

    paper1 = cog_pool[0] if cog_pool else None
    paper2 = None
    for p in ai_pool:
        if paper1 is None or p["id"] != paper1["id"]:
            paper2 = p
            break

    if paper1 is None and paper2 is None:
        return ranked[:2]
    elif paper1 is None:
        paper1 = next((p for p in ranked if p["id"] != paper2["id"]), paper2)
    elif paper2 is None:
        paper2 = next((p for p in ranked if p["id"] != paper1["id"]), paper1)

    return [p for p in [paper1, paper2] if p is not None]


# ── Claude AI Insights ────────────────────────────────────────────────────────


def generate_insights(paper):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _keyword_insights(paper)

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        prompt = f"""당신은 HCI 및 인지과학 연구자입니다. 아래 논문을 분석하고 한국어로 인사이트를 제공하세요.

논문 제목: {paper['title']}
출판지: {paper['venue']} {paper['year']}
Abstract: {paper['abstract'][:1500]}

다음 형식의 JSON으로만 응답하세요:
{{
  "contributions": ["핵심기여1 (한 문장)", "핵심기여2 (한 문장)", "핵심기여3 (한 문장)"],
  "human_cognition": "인간 인지/심리 관점에서의 의미 (1-2문장)",
  "ai_human": "AI-인간 상호작용 관련성 (1-2문장)"
}}"""

        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception as exc:
        print(f"Warning: Claude API failed: {exc}", file=sys.stderr)

    return _keyword_insights(paper)


def _keyword_insights(paper):
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", paper["abstract"].strip())
    all_kws = KEYWORDS_COGNITION + KEYWORDS_AI_HUMAN + KEYWORDS_HCI_GENERAL
    scored = sorted(
        [(sum(1 for k in all_kws if k.lower() in s.lower()), s) for s in sentences],
        reverse=True,
    )
    contributions = [s for _, s in scored[:3] if s.strip()] or [paper["abstract"][:200]]
    text = (paper["title"] + " " + paper["abstract"]).lower()
    return {
        "contributions": contributions,
        "human_cognition": (
            "인간 인지 및 심리 과정을 탐구하는 연구입니다."
            if any(k in text for k in KEYWORDS_COGNITION)
            else "HCI 맥락에서 인간 행동 패턴을 분석합니다."
        ),
        "ai_human": (
            "AI와 인간의 상호작용 설계에 기여합니다."
            if any(k in text for k in KEYWORDS_AI_HUMAN)
            else "인간 중심 설계 원칙을 다룹니다."
        ),
    }


# ── Issue Formatting ──────────────────────────────────────────────────────────

_TAG_LABELS = {
    "cognition": "🧠 인간 인지/심리",
    "ai-human": "🤖 AI-인간 상호작용",
    "hci": "🖥️ HCI 설계",
}


def _first_sentences(text, n=3):
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    return " ".join(parts[:n])


def _format_section(idx, paper, insights):
    topic = " · ".join(
        _TAG_LABELS[t] for t in paper["topic_tags"] if t in _TAG_LABELS
    ) or "📄 HCI 연구"
    award = " 🏅 **Honorable Mention**" if paper["is_honorable_mention"] else ""
    authors = ", ".join(paper["authors"][:5])
    if len(paper["authors"]) > 5:
        authors += " et al."

    abstract_excerpt = _first_sentences(paper["abstract"])
    bullets = "\n".join(
        f"- {c}" for c in insights.get("contributions", [])[:3]
    ) or "- (요약 없음)"

    url = paper.get("url", "")
    title_link = f"[{paper['title']}]({url})" if url else paper["title"]

    return f"""## Paper {idx}: {topic}

### {title_link}

| 항목 | 내용 |
|------|------|
| **저자** | {authors} |
| **출판지** | {paper['venue']}, {paper['year']}{award} |
| **인용수** | {paper['citation_count']:,}회 |

**Abstract**
> {abstract_excerpt}

**🔍 핵심 기여** *(Claude 분석)*
{bullets}

**🧠 인간 인지/심리 관점**
{insights.get('human_cognition', '')}

**🤖 AI-인간 상호작용 관련성**
{insights.get('ai_human', '')}
"""


def build_issue_body(pairs, date_str):
    header = f"""# 📚 오늘의 HCI 논문 — {date_str}

> CHI · CSCW · TOCHI · UIST · IUI에서 자동 선별된 논문입니다.
> 관심 주제: 인간 인지 · AI-인간 상호작용 · HCI 설계

---
"""
    sections = [_format_section(i + 1, p, ins) for i, (p, ins) in enumerate(pairs)]
    score_line = " | ".join(
        f"Paper {i+1}: {p['score']:.1f}점" for i, (p, _) in enumerate(pairs)
    )
    footer = f"""
---
*Powered by [Semantic Scholar](https://www.semanticscholar.org/) + [OpenAlex](https://openalex.org/) + Claude Haiku*
*선별 점수: {score_line}*"""

    return header + "\n---\n\n".join(sections) + footer


# ── GitHub API ────────────────────────────────────────────────────────────────

GH_API = "https://api.github.com"


def _gh_headers():
    token = os.environ.get("GITHUB_TOKEN", "")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def ensure_label(repo, name, color, description):
    url = f"{GH_API}/repos/{repo}/labels"
    try:
        http_post_json(url, {"name": name, "color": color, "description": description}, _gh_headers())
    except urllib.error.HTTPError as exc:
        if exc.code != 422:
            raise


def create_issue(repo, title, body, labels):
    url = f"{GH_API}/repos/{repo}/issues"
    result = http_post_json(url, {"title": title, "body": body, "labels": labels}, _gh_headers())
    return result["number"], result["html_url"]


# ── State Management ──────────────────────────────────────────────────────────


def load_state():
    try:
        with open(SENT_PAPERS_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"sent_ids": [], "history": []}


def save_state(state):
    os.makedirs(os.path.dirname(SENT_PAPERS_PATH), exist_ok=True)
    with open(SENT_PAPERS_PATH, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    today = datetime.date.today()
    date_str = today.strftime("%Y년 %m월 %d일")
    date_iso = today.isoformat()
    repo = os.environ.get("GITHUB_REPOSITORY", "")

    if not repo:
        sys.exit("Error: GITHUB_REPOSITORY env var not set")

    print(f"=== Daily HCI Papers for {date_iso} ===")

    state = load_state()
    sent_set = set(state["sent_ids"])

    print("Fetching SIGCHI honorable mentions...")
    hm_titles = fetch_honorable_mentions()
    print(f"  Found {len(hm_titles)} honorable mention titles")

    print("Fetching from Semantic Scholar...")
    ss_raws = fetch_ss_papers()
    print(f"  Got {len(ss_raws)} raw results")

    print("Fetching from OpenAlex...")
    oa_papers = fetch_openalex_papers()
    print(f"  Got {len(oa_papers)} TOCHI papers")

    # Normalize, deduplicate, filter
    papers = []
    seen = set()

    for raw in ss_raws:
        p = _normalize_ss(raw, hm_titles)
        if p and p["id"] not in seen and p["id"] not in sent_set:
            papers.append(p)
            seen.add(p["id"])

    for p in oa_papers:
        if p["id"] not in seen and p["id"] not in sent_set:
            p["is_honorable_mention"] = p["title"].lower().strip() in hm_titles
            papers.append(p)
            seen.add(p["id"])

    print(f"Candidates after filtering: {len(papers)}")

    if not papers:
        sys.exit("Error: No qualifying papers found. Check API access.")

    papers = [score_paper(p) for p in papers]
    selected = select_two(papers)

    print(f"Selected {len(selected)} papers:")
    for p in selected:
        hm_tag = " [HM]" if p["is_honorable_mention"] else ""
        print(f"  [{p['score']:.1f}]{hm_tag} {p['title'][:70]}")

    print("Generating AI insights via Claude Haiku...")
    pairs = [(p, generate_insights(p)) for p in selected]

    body = build_issue_body(pairs, date_str)
    issue_title = f"Daily HCI Papers — {date_iso}"

    ensure_label(repo, "daily-papers", "0075ca", "Daily HCI paper recommendations")
    ensure_label(repo, date_iso, "e4e669", f"Papers from {date_iso}")

    print("Creating GitHub Issue...")
    issue_num, issue_url = create_issue(repo, issue_title, body, ["daily-papers", date_iso])
    print(f"Created issue #{issue_num}: {issue_url}")

    for p in selected:
        state["sent_ids"].append(p["id"])
    state["history"].append({
        "date": date_iso,
        "paper_ids": [p["id"] for p in selected],
        "issue_number": issue_num,
    })
    save_state(state)
    print("Done!")


if __name__ == "__main__":
    main()
