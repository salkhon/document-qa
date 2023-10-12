import chromadb
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
import chainlit as cl
from chainlit.types import AskFileResponse
import tempfile
from dotenv import load_dotenv
import os
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

textsplitter = RecursiveCharacterTextSplitter()
embeddings = OpenAIEmbeddings()


def process_file(file: AskFileResponse):
    """Puts users uploaded file into the disk, and splits it using a text splitter.

    Args:
        file (AskFileResponse): Uploaded file

    Returns:
        list: List of splitted documents
    """
    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader

    with tempfile.NamedTemporaryFile() as f:
        f.write(file.content)
        loader = Loader(f.name)  # type:ignore
        documents = loader.load()

    docs = textsplitter.split_documents(documents)
    for i, doc in enumerate(docs):
        doc.metadata["source"] = f"source_{i}"

    return docs


def create_docsearch(file: AskFileResponse):
    """Creates a document search vectorstore from the uploaded file.

    Args:
        file (AskFileResponse): Uploaded file

    Returns:
        vectorstore: Document search vectorstore
    """
    docs = process_file(file)

    # save data in user session
    cl.user_session.set("docs", docs)  # shows doc to the client

    # create namespace (table) for file
    docsearch = Chroma.from_documents(docs, embeddings)
    return docsearch


@cl.on_chat_start
async def start():
    """Receive a file from the user, and create a document search vectorstore from it.
    Then create a chain with the vectorstore and a language model.
    """
    await cl.Message(
        "Welcome to Document based QnA! You can upload PDF or text files to query on them."
    ).send()

    # ask user to upload a file
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a file to start",
            accept=["text/plain", "application/pdf"],  # todo: gain insight on CSV files
            max_size_mb=20,
            timeout=180,
        ).send()

    # process file
    file = files[0]
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    docsearch = await cl.make_async(create_docsearch)(file)

    # chain the llm, retriever from vectorstore
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=ChatOpenAI(temperature=0, streaming=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(max_tokens_limit=4897),
    )

    # ready for qna
    msg.content = f"`{file.name}` processed. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: str):
    """Answer the question using the chain. If the answer is streamed, also stream the sources.

    Args:
        message (str): Question
    """
    # retrieve chain built on startup
    chain: RetrievalQAWithSourcesChain = cl.user_session.get("chain")  # type: ignore

    # as langchain streams llm output, this callback will be called
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True

    # run inference
    res = await chain.acall(message, callbacks=[cb])
    answer = res["answer"]

    # cite sources
    sources = res["sources"].strip()
    source_elems = []

    if sources:
        docs: list = cl.user_session.get("docs")  # type: ignore
        metadatas = [doc.metadata for doc in docs]
        all_sources = [m["source"] for m in metadatas]

        # find the elements corresponding to the sources
        for source in sources.split(","):
            source = source.strip().replace(".", "")

            # get idx of src
            try:
                idx = all_sources.index(source)
            except ValueError:
                continue

            txt = docs[idx].page_content
            source_elems.append(cl.Text(content=txt, name=source))

    # send the answer and sources
    if cb.has_streamed_final_answer and cb.final_stream is not None:
        cb.final_stream.elements = source_elems
        await cb.final_stream.update()
    else:
        await cl.Message(content=answer, elements=source_elems).send()
