from flask import Blueprint, request, jsonify
from . import reviewers

# Blueprint 생성
routes_bp = Blueprint('routes', __name__)

# @routes_bp.route('/', methods=['GET'])
# def index():
#     return 'hello Flask!'
#
# @routes_bp.route('/flask/repository', methods=['POST'])
# def registerRepo():
#     data = request.get_json()
#     url = data['url']
#     token = data['token']
#     projectId = data['projectId']
#
#     # 청크화
#     result = repositorys.registratinRepository(url, token, projectId)
#     if (result):
#         return jsonify({'status':'success', 'message': f'코드 임베딩 완료'}), 200
#     # 기타 예상치 못한 오류
#     return jsonify({'status': 'Internal server error', 'message': f'서버 오류가 발생했습니다'}), 500

@routes_bp.route('/flask/code-review', methods=['POST'])
def code_review():
    data = request.get_json()
    url = data.get('url')
    token = data.get('token')
    projectId = data.get('projectId')
    branch = data.get('branch')
    commits = data.get('commits')

    review = reviewers.getCodeReview(url, token, projectId, branch, commits)
    if review:
        return jsonify({'status': 'success', 'review': review})
    else:
        return jsonify({'status': 'fail', 'message': '코드 리뷰 생성 중 오류가 발생했습니다.'}), 500
