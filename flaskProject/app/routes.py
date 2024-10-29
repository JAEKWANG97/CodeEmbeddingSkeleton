from flask import Blueprint, request, jsonify
# from . import embeddings, reviewers
from . import repositorys

# Blueprint 생성
routes_bp = Blueprint('routes', __name__)

@routes_bp.route('/', methods=['GET'])
def index():
    return 'hello Flask!'

@routes_bp.route('/flask/repository', methods=['POST'])
def regeRepo():
    data = request.get_json()
    url = data['url']
    token = data['token']
    projectId = data['projectId']

    # 청크화
    result = repositorys.registratinRepository(url, token, projectId)
    if (result):
        return jsonify({'status':'success', 'message': f'코드 임베딩 완료'}), 200
    # 기타 예상치 못한 오류
    return jsonify({'status': 'Internal server error', 'message': f'서버 오류가 발생했습니다'}), 500

# @routes_bp.route('/api/embed_repo', methods=['POST'])
# def embed_repo():
#     data = request.get_json()
#     code_snippets = data.get('code_snippets', [])
#     ids = data.get('ids', [])
#
#     if not code_snippets or not ids:
#         return jsonify({'status': 'fail', 'message': '코드 스니펫과 ID를 제공해야 합니다.'}), 400
#
#     result = embeddings.store_embeddings(code_snippets, ids)
#     if result:
#         return jsonify({'status': 'success', 'message': '코드 임베딩이 저장되었습니다.'})
#     else:
#         return jsonify({'status': 'fail', 'message': '임베딩 저장 중 오류가 발생했습니다.'}), 500
#
# @routes_bp.route('/api/code_review', methods=['POST'])
# def code_review():
#     data = request.get_json()
#     code_snippet = data.get('code_snippet', '')
#
#     if not code_snippet:
#         return jsonify({'status': 'fail', 'message': '코드 스니펫을 제공해야 합니다.'}), 400
#
#     review = reviewers.generate_code_review(code_snippet)
#     if review:
#         return jsonify({'status': 'success', 'review': review})
#     else:
#         return jsonify({'status': 'fail', 'message': '코드 리뷰 생성 중 오류가 발생했습니다.'}), 500
