from langchain.chains import RetrievalQA
from langchain_openai import OpenAI  # 경로 수정
from langchain.schema import Document
from langchain.retrievers import MergerRetriever
from .embeddings import query_similar_code
from flask import current_app

def generate_code_review(code_snippet):
    try:
        # 유사한 코드 검색 결과에서 Document 리스트 생성
        related_codes = query_similar_code(code_snippet)
        documents = [Document(page_content=doc["text"]) for doc in related_codes]

        # OpenAI API 키 설정
        openai_api_key = current_app.config.get('SECRET_KEY')

        # LLM 초기화
        llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

        # MergerRetriever 사용
        retriever = MergerRetriever(retrievers=[], combine_documents_chain=[])

        # RetrievalQA 체인 생성
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        # 코드 리뷰 생성
        response = qa_chain.run(code_snippet)
        return response

    except Exception as e:
        print(f"Error generating code review: {e}")
        return None
