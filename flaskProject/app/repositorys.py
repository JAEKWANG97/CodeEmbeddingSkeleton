from app.chunking.GetCode import GitLabCodeChunker

def registratinRepository(url, token, projectId):
    chunker = GitLabCodeChunker(
       gitlab_url=url,
       gitlab_token=token,
       project_id=projectId,
       local_path='./cloneRepo/' + projectId
    )
    try:
        chunker.process_project()
        return True
    except Exception as e:
        # 기타 예상치 못한 오류
        return False