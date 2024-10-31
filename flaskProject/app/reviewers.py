# reviewers.py
import os
from pathlib import Path
from app.chunking.GetCode import GitLabCodeChunker
from app.embeddings import CodeEmbeddingProcessor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
# import logging
# logging.basicConfig(level=logging.DEBUG)
from tqdm import tqdm


def getCodeReview(url, token, projectId, branch, commits):
    # 0. DB 초기화
    vectorDB = CodeEmbeddingProcessor()

    # 1. git Clone
    chunker = GitLabCodeChunker(
        gitlab_url=url,
        gitlab_token=token,
        project_id=projectId,
        local_path='./cloneRepo/' + projectId,
        branch=branch
    )
    try:
        # 2. 파일별 임베딩
        project_path = chunker.clone_project()
        if not project_path:
            return ''

        # 3. 리뷰 할 코드들 메서드 Chunking
        file_chunks = []
        for root, _, files in os.walk(project_path):
            for file in files:
                file_path = Path(root) / file

                # 불필요한 파일/폴더 제외
                if ('.git' in str(file_path)
                        or any(part.startswith('.') for part in file_path.parts)
                        or any(part.startswith('node_modules') for part in file_path.parts)):
                    continue

                # 파일 언어 확인
                language = chunker.get_file_language(str(file_path))
                if not language:
                    continue

                with open(file_path, 'r', encoding='utf-8') as f:
                    chunks = chunker.chunk_file(str(file_path), language)
                    file_chunks.extend(chunks)

        print("청크화 끝")
        vectorDB.store_embeddings(file_chunks)
        print("임베딩 끝")

        # 4. commits 에서 코드 분리해 Chunk
        file_extensions = {
            'python': ['.py'],
            'java': ['.java'],
            'javascript': ['.js', '.jsx'],
            'c': ['.c', '.h'],
            'cpp': ['.cpp', '.hpp']
        }

        review_queries = [] # path, 코드, 참고할 코드
        for commit in commits:
            language = get_language_from_extension(commit['fileName'])

            if (language == ''):
                continue
            code_chunks = chunker.chunk_code(commit['code'], language)
            for code_chunk in code_chunks:
                similar_codes = vectorDB.query_similar_code((code_chunk))
                review_queries.append((commit['fileName'], code_chunk, similar_codes))

        # 5. 메서드 별 관련 코드 가져와 리트리버 생성, 질의
        openai_api_key = os.getenv('OPENAI_API_KEY')  # 환경 변수에서 API 키 가져오기

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=openai_api_key
        )
        # 6. LLM 에 질의해 결과 반환
        result = get_code_review(review_queries, llm)

        # 7. 삭제
        chunker.cleanup_project_directory()


        return result


    except Exception as e:
        print(f"오류 발생: {e}")
        return ''


def get_language_from_extension(file_name: str) -> str:
    extension = file_name.split('.')[-1].lower()  # 확장자 추출
    # 확장자별 언어 매핑
    language_map = {
        'py': 'python',
        'java': 'java',
        'js': 'javascript',
        'jsx': 'javascript',
        'ts': 'javascript',
        'tsx': 'javascript',
        'c': 'c',
        'cpp': 'cpp',
        'h': 'c',
        'hpp': 'cpp'
    }

    return language_map.get(extension, '')

def get_code_review(review_queries, llm):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="review_result"
    )

    # print("리뷰할 쿼리", review_queries)
    review_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 프로젝트 코드 리뷰어입니다. 
        주어진 코드만 리뷰하고, 참고용 코드는 리뷰하지 않습니다.
        참고용 코드는 오직 리뷰 대상 코드의 개선 방향을 제안하는 데만 활용합니다."""),
        ("human", """
        아래 메서드에 대해서만 코드 리뷰를 진행해주세요.
        참고용 코드는 리뷰하지 말고, 단순히 개선 방향 제안시 참고만 해주세요.

        리뷰 대상 파일 경로: {file_path}

        ===리뷰 대상 메서드===
        ```
        {code_chunk}
        ```

        ===참고용 코드 (리뷰하지 말 것)===
        아래 코드는 GraphCodeBERT가 찾은 유사 코드로, 리뷰 대상이 아닙니다.
        개선 방향 제안시 참고용으로만 사용하세요.
        ```
        {similar_codes}
        ```

        다음 항목에 대해 리뷰 대상 메서드만 검토해 주세요:
        1. 기능 설명: 리뷰 대상 메서드의 목적과 수행 기능
        2. 개선 사항: 유사 코드를 참고하여 성능 최적화 및 개선할 부분 요약
        3. 수정 필요 항목: 즉시 수정이 필요한 버그나 논리적 오류가 있다면 제안
        """)
    ])

    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "이전에 진행한 코드 리뷰들을 종합하여 최종 리포트를 작성합니다."),
        ("human", """
        지금까지 진행한 코드 리뷰 내용을 종합하여 Merge Request 코멘트를 작성해주세요.
        리뷰 대상 메서드들에 대한 내용만 포함하고, 참고용 코드에 대한 내용은 제외해주세요.

        아래 형식으로 작성해주세요:
        - **파일 경로 및 메서드명**: [경로] - `[메서드명]`
            - **기능 설명**: 해당 메서드의 기능 설명
            - **주요 개선 사항**:
                1. [우선순위가 높은 개선사항]
                2. [그 다음 개선사항]
                ...
            - **즉시 수정이 필요한 주요 이슈**:
                - [발견된 이슈]
                ...

        모든 리뷰 내용을 종합하여 일관성 있게 작성해주세요.
        """)
    ])

    # 체인 구성
    review_chain = (
            review_prompt
            | llm
            | StrOutputParser()
    )

    summary_chain = (
            summary_prompt
            | llm
            | StrOutputParser()
    )

    try:
        for file_path, code_chunk, similar_codes in review_queries:
            # 리뷰 수행 및 결과 저장
            review_result = review_chain.invoke({
                "file_path": file_path,
                "code_chunk": code_chunk,
                "similar_codes": similar_codes,
                "chat_history": memory.load_memory_variables({})["chat_history"]
            })

            # 메모리에 리뷰 결과 저장
            memory.save_context(
                {"input": f"Review for {file_path}"},
                {"review_result": review_result}
            )
            print(f"개별 리뷰 완료: {review_result}")

        # 최종 요약 생성 - 이전 모든 리뷰 내용 참조
        final_review = summary_chain.invoke({
            "chat_history": memory.load_memory_variables({})["chat_history"]
        })

        return final_review

    except Exception as e:
        print(f"리뷰 중 오류 발생: {e}")
        return str(e)

    # 최종 요약 생성 - memory에서 이전 대화 내용을 자동으로 참조
    final_review = summary_chain.invoke({})
    print("최종 리뷰", final_review)
    return final_review