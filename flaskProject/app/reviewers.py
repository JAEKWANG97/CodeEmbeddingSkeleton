# reviewers.py
import os
from pathlib import Path
from app.chunking.GetCode import GitLabCodeChunker
from app.embeddings import CodeEmbeddingProcessor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain  # 추가
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

        # vectorDB.store_embeddings(file_chunks)
# === Clone, Chunking, Embedding Logic
#===============================================================================
# === diff 기반 Chunking, Embedding, Code Review
        # 4. commits 에서 코드 분리해 Chunk
        file_extensions = {
            'python': ['.py'],
            'java': ['.java'],
            'javascript': ['.js', '.jsx'],
            'c': ['.c', '.h'],
            'cpp': ['.cpp', '.hpp']
        }

        review_queries = [] # path, diff (전문), 참고할 코드 (메서드)
        for commit in commits:
            language = get_language_from_extension(commit['new_path'])

            if (language == ''):
                continue
            # ++, -- 별로 파싱 하는 로직 필요
            removed_lines, added_lines = parse_git_diff(commit['diff'])
            similar_codes = []
            code_chunks = []
            # 청크화
            for removed_line in removed_lines:
                code_chunks.append(chunker.chunk_code(removed_line, language))
            # -- 에 메서드와 유사한 메서드 추출
            for code_chunk in code_chunks:
                similar_codes.append(vectorDB.query_similar_code((code_chunk)))

            review_queries.append((commit['new_path'], commit['diff'], similar_codes))
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
        # output_key="review_result"
    )

    system_message = """당신은 다양한 프로그래밍 언어에 대한 전문적인 코드 리뷰어입니다.
    각 언어와 프레임워크의 특성을 고려하여 자세하게 리뷰를 진행합니다:

    - Java/Spring Boot: 계층 구조, 어노테이션, 의존성 주입
    - Python: PEP 8, 타입 힌트, 모듈화
    - JavaScript/React: 컴포넌트 구조, 상태 관리, 최신 ES 문법
    - C/C++: 메모리 관리, 포인터 사용, 최적화

    코드의 언어와 문맥을 파악하여 적절한 기준으로 리뷰해주세요."""

    human_message = """아래 git diff에 대해서만 코드 리뷰를 진행해주세요.
    파일 경로와 확장자를 보고 해당 언어에 맞는 리뷰 기준을 적용해주세요.
    중요한 부분, 핵심 로직은 강조해서 설명해주세요

    리뷰 대상 파일 경로: {file_path}

    ===리뷰 대상 git diff===
    ```
    {code_chunk}
    ```

    ===참고용 코드 (리뷰하지 말 것)===
    ```
    {similar_codes}
    ```

    다음 항목에 대해 리뷰 대상 코드만 검토해 주세요:
    1. 기능 설명: 코드의 목적과 수행 기능, 구현 방법을 상세히
    2. 개선 사항: 해당 언어/프레임워크의 관점에서 개선할 부분 [ 코드 추가 ]
    3. 수정 필요 항목: 버그나 오류 가능성이 있는 부분"""

    review_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_message)
    ])

    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "이전에 진행한 MR의 코드 리뷰들을 종합하여 최종 리포트를 작성합니다."),
        ("human", """지금까지 진행한 MR 내부 수정사항의 코드 리뷰 내용을 종합하여 수정 파일 별 Merge Request 코멘트를 작성해주세요.
            이떄 마크다운과 이모티콘을 사용해 이쁘게 꾸며 주세요
            아래 형식으로 작성해주세요:
            
                ### E.D.I.T.H. 코드리뷰 
                
                - 수정 파일
                    경로/파일명
                - 기능 설명 
                    해당 메서드의 기능 설명, 구현 방법, 사용된 기술 을 자세히 설명 
                - 주요 개선 사항
                    1. [우선순위가 높은 개선사항]
                    2. [그 다음 개선사항]
                    ...
                - 즉시 수정이 필요한 주요 이슈:
                    - [코드, 발견된 이슈]
                    ...
                - 컨벤션:
                    - [참고할 코드가 있으면 컨벤션이 잘 지켜졌는지]

            모든 리뷰 내용을 종합하여 일관성 있게 작성해주세요.""")
    ])



    # 체인 구성
    review_chain = (
            review_prompt
            | llm
            | StrOutputParser()
    )

    # summary_chain을 LLMChain으로 생성
    summary_chain = LLMChain(
        llm=llm,
        prompt=summary_prompt,
        memory=memory,  # 여기서 메모리 연결
        output_parser=StrOutputParser()
    )

    try:

        for file_path, code_chunk, similar_codes in review_queries:

            # 입력을 정확히 전달하도록 확인
            try:
                review_result = review_chain.invoke({
                    "file_path": file_path,
                    "code_chunk": code_chunk,
                    "similar_codes": similar_codes
                })

                # 메모리에 리뷰 결과 저장
                memory.save_context(
                    {"input": f"Review for {file_path}"},
                    {"output": review_result}
                )

            except Exception as e:
                print(f"개별 리뷰 중 오류 발생: {e}")
                continue  # 오류 발생 시 다음 항목으로 건너뜀

        response = summary_chain.invoke({"input": "Generate final review"})
        final_review = response.get('text', '') if isinstance(response, dict) else str(response)

        print(final_review)
        return final_review

    except Exception as e:
        print(f"리뷰 중 오류 발생: {e}")
        return str(e)

def parse_git_diff(diff_string):
    # diff 헤더(@@ -0,0 +1,30 @@) 이후부터 파싱
    lines = diff_string.split('\n')
    start_idx = 0

    # diff 헤더 찾기
    for i, line in enumerate(lines):
        if line.startswith('@@'):
            start_idx = i + 1
            break

    removed_lines = []
    added_lines = []

    # 실제 코드 변경사항 파싱
    for line in lines[start_idx:]:
        # 빈 줄 무시
        if not line:
            continue

        # 삭제된 라인('-'로 시작)
        if line.startswith('-'):
            removed_lines.append(line[1:])  # '-' 제외하고 저장
        # 추가된 라인('+'로 시작)
        elif line.startswith('+'):
            added_lines.append(line[1:])  # '+' 제외하고 저장

    return removed_lines, added_lines