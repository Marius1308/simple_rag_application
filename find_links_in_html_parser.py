"""Includes the FindLinksInHTMLParser class"""
from html.parser import HTMLParser


class FindLinksInHTMLParser(HTMLParser):
    """This class is used to find all links in a HTML page and store them in a list"""

    def __init__(self, url_prefix: str):
        super().__init__()
        self.parsed_pages: list[str] = []
        self.url_prefix: str = url_prefix

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value is not None:
                    if value.startswith(self.url_prefix) and not value.endswith(
                        ".html"
                    ):
                        if value not in self.parsed_pages:
                            new_page = value
                            if value.endswith("/"):
                                new_page = value[:-1]
                            self.parsed_pages.append(
                                new_page.replace(self.url_prefix, "")
                            )

    def reset_pages(self):
        """Resets the list of parsed pages"""
        self.parsed_pages = []
