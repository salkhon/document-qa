import chromadb


def main():
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="my_collection")
    collection.add(
        documents=["My name is Salman", "I am an undergrad"],
        metadatas=[
            {"source": "name"},
            {"source": "degree"},
        ],  # in case th model hallucinates, user can find out the source of the information
        ids=["id1", "id2"],
    )

    results = collection.query(
        query_texts=["What is my name?"],
        n_results=2,
    )
    print(results)


if __name__ == "__main__":
    main()
