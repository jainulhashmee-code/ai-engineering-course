"""Archive every outcomeschool.com/blog page linked from README.md into lessons/<slug>.md (article HTML) + lessons/images/."""
import html, os, re, sys, time, urllib.parse, urllib.request

REPO = sys.argv[1]
BASE = "https://outcomeschool.com"
LESSONS = os.path.join(REPO, "lessons")
IMAGES = os.path.join(LESSONS, "images")
os.makedirs(IMAGES, exist_ok=True)


def get(url):
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def prose_div(page):
    # The article body is the first <div class="prose ..."> on the page; walk div depth to find its end.
    start = page.index('<div class="prose')
    depth = 0
    for m in re.compile(r"<(/?)div\b").finditer(page, start):
        depth += -1 if m.group(1) else 1
        if depth == 0:
            inner_start = page.index(">", start) + 1
            return page[inner_start : m.start()]
    raise ValueError("unterminated prose div")


def clean(body, slugs):
    body = re.sub(r"<noscript>.*?</noscript>", "", body, flags=re.S)
    body = re.sub(r'<img[^>]*src="data:[^"]*"[^>]*/?>', "", body)  # next/image placeholders

    def img(m):
        src = html.unescape(m.group(1))
        if src.startswith("/_next/image"):
            src = urllib.parse.parse_qs(urllib.parse.urlparse(src).query)["url"][0]
        if src.startswith("/"):
            name = os.path.basename(src)
            path = os.path.join(IMAGES, name)
            if not os.path.exists(path):
                open(path, "wb").write(get(BASE + src))
            src = "images/" + name
        alt = re.search(r'alt="([^"]*)"', m.group(0))
        return f'<img alt="{alt.group(1) if alt else ""}" src="{src}"/>'

    body = re.sub(r'<img[^>]*?\ssrc="([^"]+)"[^>]*/?>', img, body)
    body = re.sub(r"</?span\b[^>]*>", "", body)  # next/image wrappers

    def link(m):
        href = m.group(1)
        slug = href.split("#")[0].removeprefix(BASE).removeprefix("/blog/")
        if href.startswith(("/blog/", BASE + "/blog/")) and slug in slugs:
            return f'href="{slug}.md{"#" + href.split("#", 1)[1] if "#" in href else ""}"'
        return f'href="{BASE + href if href.startswith("/") else href}"'

    return re.sub(r'href="([^"]+)"', link, body)


def main():
    readme = open(os.path.join(REPO, "README.md")).read()
    # Matches both the original links and ones already rewritten to lessons/, so re-runs after an upstream merge work.
    slugs = sorted(set(re.findall(r"(?:https://outcomeschool\.com/blog/|\(lessons/)([\w-]+)", readme)))
    failed = []
    for n, slug in enumerate(slugs, 1):
        out = os.path.join(LESSONS, slug + ".md")
        if os.path.exists(out):
            continue
        try:
            page = get(f"{BASE}/blog/{slug}").decode()
            title = html.unescape(re.search(r"<title>(.*?)</title>", page, re.S).group(1).strip())
            body = clean(prose_div(page), slugs)
            open(out, "w").write(
                f"# {title}\n\n> Archived from <{BASE}/blog/{slug}> on {time.strftime('%Y-%m-%d')}.\n\n{body}\n"
            )
            print(f"{n}/{len(slugs)} ok {slug}", flush=True)
        except Exception as e:
            failed.append(slug)
            print(f"{n}/{len(slugs)} FAIL {slug}: {e}", flush=True)
        time.sleep(0.5)
    print("failed:", failed)
    # Point README at the local copy of every lesson that was archived; failed ones keep the live link.
    readme = re.sub(
        r"\(https://outcomeschool\.com/blog/([\w-]+)\)",
        lambda m: f"(lessons/{m.group(1)}.md)" if os.path.exists(os.path.join(LESSONS, m.group(1) + ".md")) else m.group(0),
        readme,
    )
    open(os.path.join(REPO, "README.md"), "w").write(readme)


if __name__ == "__main__":
    main()
