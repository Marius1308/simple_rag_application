from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import BSHTMLLoader
from pageLinks import pages
import requests
from data_supported_request.find_links_in_html_parser import FindLinksInHTMLParser


def cut_new_lines(page_content: str):
    while "\n\n" in page_content:
        page_content = page_content.replace("\n\n", "\n")
    while page_content[-1] == "\n":
        page_content = page_content[:-1]
    while page_content[0] == "\n":
        page_content = page_content[1:]
    return page_content


def add_new_lines(page_content: str, chunk_size: int):
    new_page_content = ""
    buffer = chunk_size / 10
    added = False
    for char_index, char in enumerate(page_content):
        suffix = ""
        if (
            char_index % chunk_size < buffer
            or char_index % chunk_size > chunk_size - buffer
        ) and char_index != 0:
            if not added and char in [".", "!", "?"]:
                suffix = "\n\n"
                added = True
        if not added and char_index % chunk_size == buffer:
            suffix = "\n\n"
            added = True
        if char_index % chunk_size == buffer + 1:
            added = False
        new_page_content += char + suffix
    return new_page_content


parsedPages = []


def add_pages(allPages: list[str], root: str):
    parser = FindLinksInHTMLParser("/docs/")
    paths: list[str] = []
    additional_pages: list[str] = []
    for p in allPages:
        response = requests.get(f"{root}/{p}", timeout=10)
        if response.status_code != 200:
            continue

        page_name = p.replace("/", "-")
        page_path = f"./langchainPages/html/{page_name}.html"

        response_text = response.content.decode("utf-8")
        text_file = open(page_path, "w", encoding="utf-8")
        text_file.write(response_text)
        text_file.close()
        paths.append(page_path)
        parser.reset_pages()
        parser.feed(response_text)
        additional_pages += parser.parsed_pages
    return paths, additional_pages


ROOT = "https://python.langchain.com/docs"
pages = [
    "get_started/introduction",
    "get_started/installation",
    "get_started/quickstart",
]
addedPages = 0
paths: list[str] = []

has_new_pages = True

while has_new_pages:
    has_new_pages = False
    newPaths, newPages = add_pages(pages[addedPages:], ROOT)
    addedPages = len(pages)
    print(addedPages)
    paths += newPaths
    for page in newPages:
        if page not in pages:
            pages.append(page)
            has_new_pages = True


for path in paths:
    loader = BSHTMLLoader(path)
    raw_documents = loader.load()
    for index, document in enumerate(raw_documents):
        raw_documents[index].page_content = add_new_lines(
            cut_new_lines(document.page_content), 1000
        )
    text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    Chroma.from_documents(  # type: ignore
        documents,
        OpenAIEmbeddings(),  # type: ignore
        persist_directory="./langchainPages/db/chroma_db",
    )
