import os
import csv
from datetime import datetime
from uuid import uuid4
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, abort
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from evaluation import get_evaluator

# Flask 애플리케이션 초기화
app = Flask(__name__)

# 파일 업로드 폴더 설정
app.config['UPLOAD_FOLDER'] = 'uploads'

# 업로드 파일 크기 제한 설정 (기본값 25MB, 환경 변수로 변경 가능)
max_mb = int(os.getenv('MAX_CONTENT_LENGTH_MB', '25'))
app.config['MAX_CONTENT_LENGTH'] = max_mb * 1024 * 1024

# 허용되는 파일 확장자 설정 (환경 변수로 변경 가능)
ALLOWED_EXTENSIONS = set(
    os.getenv('ALLOWED_EXTENSIONS', 'png,jpg,jpeg,webp,bmp').lower().split(',')
)

# 업로드 폴더가 존재하지 않으면 생성
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    """메인 페이지를 렌더링합니다."""
    return render_template('index.html')

@app.route('/test')
def test():
    """테스트 페이지를 렌더링합니다."""
    return render_template('test.html')

@app.route('/healthz')
def healthz():
    """헬스 체크 엔드포인트입니다."""
    return {"status": "ok"}

@app.route('/evaluate', methods=['POST'])
def evaluate():
    """이미지 평가 요청을 처리하는 메인 로직입니다."""
    try:
        print("=== 평가 요청 시작 ===")
        print(f"요청 방법: {request.method}")
        print(f"Content-Type: {request.content_type}")
        print(f"Form keys: {list(request.form.keys())}")
        print(f"Files keys: {list(request.files.keys())}")
        
        # 요청에 이미지 파일이 없으면 메인 페이지로 리디렉션
        if 'images' not in request.files:
            print("ERROR: 이미지 파일이 요청에 없음")
            return redirect(url_for('index'))

        # 폼 데이터에서 주제, 프롬프트, 기준 이미지 이름 가져오기
        topic = (request.form.get('topic') or '').strip()
        prompt = (request.form.get('prompt') or '').strip()
        ref_image_name = request.form.get('ref_image')
        
        # 업로드된 파일 목록 가져오기
        files = request.files.getlist('images')
        
        def allowed_file(filename: str) -> bool:
            """파일 확장자가 허용되는지 확인합니다."""
            return ('.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)

        image_paths = {}  # 원본 파일명 -> 저장된 실제 경로
        image_urls = {}   # 원본 파일명 -> 웹에서 접근 가능한 URL

        # 모든 업로드된 파일을 처리
        for file in files:
            if not file or not file.filename:
                continue
            if not allowed_file(file.filename):
                return f"Error: Only image files are allowed ({', '.join(sorted(ALLOWED_EXTENSIONS))})"
            
            # 파일명 처리: 한글 등 특수문자가 포함된 파일명 처리
            original_filename = file.filename
            file_extension = original_filename.rsplit('.', 1)[1] if '.' in original_filename else ''
            
            # secure_filename이 파일명을 너무 많이 제거하는 경우를 대비해 대체 방법 사용
            secure_name = secure_filename(original_filename)
            if not secure_name or secure_name == file_extension:
                # secure_filename이 파일명을 너무 많이 제거한 경우, UUID 기반 이름 사용
                secure_name = f"image_{len(image_paths) + 1}.{file_extension}"
            
            unique_name = f"{uuid4().hex}_{secure_name}"
            path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
            file.save(path)
            
            # 원본 파일명을 키로 사용하여 경로와 URL 저장 (매핑 불필요)
            image_paths[original_filename] = path
            image_urls[original_filename] = url_for('serve_upload', filename=unique_name)

        # 기준 이미지 결정
        if not ref_image_name:
            if len(image_paths) == 1:
                # 이미지가 하나만 업로드된 경우, 해당 이미지를 기준으로 삼음
                ref_image_name = next(iter(image_paths.keys()))
            else:
                return "Error: 여러 이미지를 업로드할 경우 기준 이미지를 선택해야 합니다."

        ref_image_path = image_paths.get(ref_image_name)
        ref_image_url = image_urls.get(ref_image_name)
        
        if not ref_image_path:
            return f"Error: 기준 이미지를 찾을 수 없습니다. 요청된 이미지: '{ref_image_name}', 사용 가능한 이미지: {list(image_paths.keys())}"

        # 평가 모델 로드
        evaluator = get_evaluator()
        results = []
        
        # 각 이미지에 대해 평가 수행
        for filename, path in image_paths.items():
            # 이미지가 하나일 경우, 자기 자신을 기준으로 평가 (LPIPS 점수는 0이 됨)
            eval_ref_path = ref_image_path
            if len(image_paths) == 1:
                eval_ref_path = path

            # 평가 실행
            eval_result = evaluator.evaluate(path, eval_ref_path, prompt, topic)
            eval_result['filename'] = filename
            eval_result['url'] = image_urls[filename]  # 결과 표시에 사용할 URL 추가
            
            # 기준 이미지인 경우 LPIPS 점수를 제외 처리
            if filename == ref_image_name and len(image_paths) > 1:
                eval_result['lpips_score'] = None
                eval_result['lpips_norm'] = None
                eval_result['is_reference'] = True
                # LPIPS를 제외한 점수로 재계산
                clip_norm = eval_result['clip_norm']
                blip2_norm = eval_result['blip2_norm']
                eval_result['final_score'] = (clip_norm * 0.3 + blip2_norm * 0.7) * 100  # 가중치 재조정
            else:
                eval_result['is_reference'] = False
            
            results.append(eval_result)

        # 평가 결과를 CSV 파일로 저장
        csv_filename = f"results_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid4().hex[:8]}.csv"
        csv_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_filename)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'filename', 'clip_score', 'clip_norm', 'blip2_similarity', 'blip2_norm', 'lpips_score', 'lpips_norm', 'final_score', 'is_reference'
            ])
            for r in results:
                lpips_score = r['lpips_score'] if r['lpips_score'] is not None else '-'
                lpips_norm = r['lpips_norm'] if r['lpips_norm'] is not None else '-'
                writer.writerow([
                    r['filename'], r['clip_score'], r['clip_norm'], r['blip2_similarity'], r['blip2_norm'], lpips_score, lpips_norm, r['final_score'], r['is_reference']
                ])

        # 결과 페이지 렌더링
        return render_template('results.html', 
                             results=results, 
                             ref_image_name=ref_image_name, 
                             ref_image_url=ref_image_url, 
                             csv_filename=csv_filename)
    
    except Exception as e:
        # 예외 발생 시 에러 로깅 및 사용자에게 친화적인 메시지 반환
        import traceback
        print(f"평가 처리 중 오류 발생: {str(e)}")
        print(f"스택 트레이스: {traceback.format_exc()}")
        return f"Error: 평가 처리 중 문제가 발생했습니다. {str(e)}", 500

@app.route('/uploads/<path:filename>')
def serve_upload(filename: str):
    """업로드된 파일을 웹에서 접근할 수 있도록 제공합니다."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/download/<path:filename>')
def download_file(filename: str):
    """생성된 CSV 파일을 다운로드할 수 있도록 제공합니다."""
    uploads_dir = app.config['UPLOAD_FOLDER']
    file_path = os.path.join(uploads_dir, filename)
    # 파일이 존재하지 않으면 404 에러 반환
    if not os.path.isfile(file_path):
        abort(404)
    return send_from_directory(uploads_dir, filename, as_attachment=True)

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    """파일 크기 제한을 초과했을 때 에러를 처리합니다."""
    return "Error: 업로드된 파일이 너무 큽니다.", 413

@app.errorhandler(500)
def handle_internal_error(e):
    """내부 서버 오류를 처리합니다."""
    return "Error: 서버 내부 오류가 발생했습니다.", 500

if __name__ == '__main__':
    # 개발용 서버 실행
    app.run(debug=True, host='0.0.0.0', port=8081)