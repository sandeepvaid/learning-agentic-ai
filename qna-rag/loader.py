import bs4
from langchain_community.document_loaders import WebBaseLoader


def get_webpage_data():
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))

    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()
    return docs
