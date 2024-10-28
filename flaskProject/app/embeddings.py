from .models.codebert_model import get_code_embedding
import chromadb
from chromadb.utils import embedding_functions

# ChromaDB 클라이언트 및 컬렉션 초기화
client = chromadb.Client()
collection = client.create_collection(name='code_embeddings')

def store_embeddings(code_snippets, ids):
    # 코드를 벡터화하여 ChromaDB에 저장하는 함수
    try:
        embeddings = [get_code_embedding(code) for code in code_snippets]
        collection.add(
            documents=code_snippets,
            metadatas=[{"id": id_} for id_ in ids],
            embeddings=embeddings,
            ids=ids
        )
        return True
    except Exception as e:
        print(f"Error storing embeddings: {e}")
        return False

def query_similar_code(code_snippet, n_results=5):
    # 입력 코드 스니펫에 대한 유사한 코드를 검색하는 함수
    try:
        query_embedding = get_code_embedding(code_snippet)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents']
        )
        related_codes = results['documents'][0]
        return related_codes
    except Exception as e:
        print(f"Error querying similar code: {e}")
        return []
