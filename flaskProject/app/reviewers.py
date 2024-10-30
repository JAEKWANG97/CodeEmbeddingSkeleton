# reviewers.py
import os
from .embeddings import query_similar_code  # vectordb 임포트
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from .embeddings import graphcodebert_embeddings
from langchain.vectorstores import Chroma


def generate_code_review(code_snippet, project_id):
    vectordb = Chroma(
        embedding_function=graphcodebert_embeddings,
        collection_name=f'code_embeddings_{project_id}'
    )
    try:
        # 유사한 코드 검색
        related_codes = query_similar_code(code_snippet, project_id)
        print("Related Codes:", related_codes)

        # 문서 생성
        if not related_codes:
            print("No related codes found. Generating review for the provided code snippet only.")
            documents = [Document(page_content=code_snippet)]
        else:
            # 입력 코드 스니펫을 포함하여 문서 리스트 생성
            documents = [Document(page_content=code_snippet)]
            documents.extend([Document(page_content=doc) for doc in related_codes if isinstance(doc, str)])

        print("Documents:", documents)

        # OpenAI API 키 설정
        openai_api_key = os.getenv('OPENAI_API_KEY')  # 환경 변수에서 API 키 가져오기


        llm = ChatOpenAI(
            model="gpt-4o-mini",  # 원하는 모델을 명시적으로 지정합니다
            temperature=0,
            openai_api_key=openai_api_key
        )

        # Retriever 생성 (Chroma의 as_retriever 사용)
        retriever = vectordb.as_retriever()

        # 커스텀 프롬프트 템플릿 정의
        prompt_template = """
        다음은 코드 스니펫과 관련된 정보입니다:

        {context}

        한국어로 주어진 코드에 대해 다음 사항을 포함하여 코드 리뷰를 작성해줘:
        1. **코드의 기능 설명**: 코드가 수행하는 주요 기능을 간단히 설명해주세요.
        2. **잘한 점**: 코드 작성에 있어서 효율적이거나 개선되지 않아도 좋은 부분에 대해 설명해주세요.
        3. **개선할 부분**: 코드 품질을 높이기 위해 수정하거나 보완할 필요가 있는 부분에 대해 설명해주세요. 성능, 보안, 코드 가독성, 확장성 등을 고려하여 구체적으로 작성해주세요.
        4. **참고 코드와의 비교**: 유사한 코드들이 있다면, 해당 코드와의 비교를 통해 공통점과 차이점을 간단히 설명하고 개선 방향을 제시해주세요.
        
        {question}
        """

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # context로 사용할 관련 코드 문서 내용 준비
        context = "\n\n".join([doc.page_content for doc in documents])

        # 최종 프롬프트 생성 및 출력
        formatted_prompt = PROMPT.format(context=context, question=code_snippet)
        print("Final Prompt to LLM:\n", formatted_prompt)

        # RetrievalQA 체인 생성 (커스텀 프롬프트 사용)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=False,
            chain_type_kwargs={"prompt": PROMPT}
        )



        # 코드 리뷰 생성
        response = qa_chain.invoke(code_snippet)
        return response

    except Exception as e:
        print(f"Error generating code review: {e}")
        return None
