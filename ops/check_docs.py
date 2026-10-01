"""Fail a documentation build on broken local pages, images, or fragment links."""

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import sys


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        for name in ("href", "src"):
            if name in attrs:
                self.links.append(attrs[name])


def check(root):
    root = root.resolve()
    pages = {}
    for path in root.rglob("*.html"):
        parsed = Links()
        parsed.feed(path.read_text())
        pages[path] = parsed
    if not pages or not (root / "search.html").is_file():
        raise ValueError("documentation pages or search are missing")
    errors = []
    for path, page in pages.items():
        for link in page.links:
            url = urlsplit(link)
            if url.scheme or url.netloc or url.path.startswith("/"):
                continue
            target = (path.parent / unquote(url.path)).resolve() if url.path else path
            if target.is_dir():
                target /= "index.html"
            if not target.is_relative_to(root) or not target.is_file():
                errors.append(f"{path.name}: missing {link}")
            elif (
                url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids
            ):
                errors.append(f"{path.name}: missing fragment {link}")
    if errors:
        raise ValueError("\n".join(errors))
    print(f"Verified local links, fragments and assets across {len(pages)} documentation pages.")


if __name__ == "__main__":
    check(Path(sys.argv[1]))
