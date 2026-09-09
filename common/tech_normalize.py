"""
tech_normalize.py — 기술 스택 이름 정규화. 저장소 전체의 단일 출처.

---------------------------------------------------------------------------
어디서 왔나
---------------------------------------------------------------------------
`임시/test2.ipynb` 에서 데이터 정리할 때 만든 MASTER_TECH_DICT 와,
`01-tech-stack-wordcloud/app.py` 의 EXCLUDE_TECH 를 합쳐 모듈로 뺐다.
추출 로직(extract_techs)도 같은 노트북의 extract_from_complex_string 을 옮긴 것이다.

노트북 안에 두면 01·03·04 가 각자 복사해 쓰게 되고, 실제로 이미
`임시/streamlit.py`·`streamlit_year.py`(TECH_NAME_MAP, 약 50개)·
`01/app.py` 에 조금씩 다른 축소판이 흩어져 있었다. 여기를 정본으로 삼는다.

⚠️ test.ipynb 판(280개)에는 `'고': 'Go'` 가 있는데 test2.ipynb 판(279개)에는
   없다. '고'는 광고·고객·보고·사고에 걸리는 위험한 별칭이라 빠진 것으로 보고
   **test2 판을 정본으로 채택**했다.

---------------------------------------------------------------------------
왜 03(이력서 추천)에 결정적인가
---------------------------------------------------------------------------
이력서는 한국어로 쓴다.

    이력서 : "파이썬과 장고로 결제 API를 개발했습니다"
    공고   : final_techs = ['Python', 'Django']

사전 없이 매칭하면 교집합이 0이다. 임베딩을 붙여도 이 문제는 안 풀린다.
이 사전의 **한글 별칭 91개**가 그 간극을 메운다.

데이터셋에 실제로 등장하는 기술 156종을 이 사전이 전부 커버한다(100%).

---------------------------------------------------------------------------
쓰는 법
---------------------------------------------------------------------------
    from tech_normalize import extract_techs, canonical, EXCLUDE_TECH

    extract_techs("파이썬과 장고로 결제 API 개발")     -> {'Python', 'Django'}
    canonical("스프링부트")                            -> 'Spring Boot'
    extract_techs(resume_text, exclude_stopwords=True) -> 불용어 제거 버전
"""

import difflib
import re


# ---------------------------------------------------------------------------
# 1. 별칭 -> 표준명  (임시/test2.ipynb 원문 그대로)
# ---------------------------------------------------------------------------
MASTER_TECH_DICT = {
    # ==========================================
    # 🔹 프로그래밍 언어 (Languages)
    # ==========================================
    'python': 'Python', '파이썬': 'Python', 
    'java': 'Java', '자바': 'Java', 
    'c': 'C', 'c++': 'C++', 'c#': 'C#', 
    'javascript': 'JavaScript', 'js': 'JavaScript', '자바스크립트': 'JavaScript', 
    'typescript': 'TypeScript', 'ts': 'TypeScript', '타입스크립트': 'TypeScript', 
    'kotlin': 'Kotlin', '코틀린': 'Kotlin', 
    'swift': 'Swift', '스위프트': 'Swift', 
    'php': 'PHP', 
    'go': 'Go', 'golang': 'Go',
    'ruby': 'Ruby', '루비': 'Ruby', 
    'rust': 'Rust', '러스트': 'Rust', 
    'dart': 'Dart', '다트': 'Dart',
    'scala': 'Scala', '스칼라': 'Scala',    # 추가됨
    'r': 'R', 'perl': 'Perl', 'lua': 'Lua', # 추가됨
    'shell': 'Shell', 'bash': 'Bash', 'powershell': 'PowerShell', # 추가됨
    
    # ==========================================
    # 🔹 백엔드 & 프레임워크 (Backend)
    # ==========================================
    'spring': 'Spring', '스프링': 'Spring', 'spring framework': 'Spring',
    'springboot': 'Spring Boot', '스프링부트': 'Spring Boot',
    'springmvc': 'Spring MVC', 'springwebflux': 'Spring WebFlux', # 추가됨
    'jpa': 'JPA', 'hibernate': 'Hibernate', 'mybatis': 'MyBatis', # 추가됨
    'django': 'Django', '장고': 'Django', 
    'fastapi': 'FastAPI', 
    'flask': 'Flask', '플라스크': 'Flask',
    'nodejs': 'Node.js', 'node': 'Node.js', '노드': 'Node.js',
    'express': 'Express', '익스프레스': 'Express', 
    'nestjs': 'NestJS', 'nest.js': 'NestJS', '네스트': 'NestJS',
    'rubyonrails': 'Ruby on Rails', 'rails': 'Ruby on Rails',
    'laravel': 'Laravel', '라라벨': 'Laravel', 
    '.net': '.NET', '닷넷': '.NET', 'asp.net': 'ASP.NET',
    'graphql': 'GraphQL', '그래프ql': 'GraphQL', # 추가됨 (통신)
    'grpc': 'gRPC', 'websocket': 'WebSocket', 'restapi': 'REST API', 'restfulapi': 'REST API', # 추가됨
    
    # ==========================================
    # 🔹 프론트엔드 & 모바일 (Frontend & Mobile)
    # ==========================================
    'react': 'React', 'reactjs': 'React', '리액트': 'React', 
    'vue': 'Vue.js', 'vuejs': 'Vue.js', 'vue.js': 'Vue.js', '뷰': 'Vue.js', 
    'nextjs': 'Next.js', 'next.js': 'Next.js', '넥스트': 'Next.js', 
    'nuxtjs': 'Nuxt.js', 'nuxt.js': 'Nuxt.js', '넉스트': 'Nuxt.js', # 추가됨
    'angular': 'Angular', '앵귤러': 'Angular', 
    'svelte': 'Svelte', '스벨트': 'Svelte',
    'html': 'HTML', 'html5': 'HTML', 
    'css': 'CSS', 'css3': 'CSS',
    # 상태관리 & UI 라이브러리 (현업 필수 추가됨)
    'redux': 'Redux', '리덕스': 'Redux', 'recoil': 'Recoil', 'zustand': 'Zustand', '주스탠드': 'Zustand',
    'reactquery': 'React Query', 'tanstackquery': 'React Query',
    'tailwindcss': 'Tailwind CSS', 'tailwind': 'Tailwind CSS', '테일윈드': 'Tailwind CSS',
    'bootstrap': 'Bootstrap', '부트스트랩': 'Bootstrap', 'styledcomponents': 'Styled-Components',
    'threejs': 'Three.js', 'three.js': 'Three.js', 'webgl': 'WebGL',
    # 모바일
    'reactnative': 'React Native', '리액트네이티브': 'React Native', 'rn': 'React Native',
    'flutter': 'Flutter', '플러터': 'Flutter', 
    'ios': 'iOS', 'android': 'Android', '안드로이드': 'Android',
    
    # ==========================================
    # 🔹 데이터베이스 & 데이터 파이프라인 (DB & Data)
    # ==========================================
    'sql': 'SQL', 'rdbms': 'RDBMS', 'nosql': 'NoSQL',
    'mysql': 'MySQL', 'postgresql': 'PostgreSQL', 'postgres': 'PostgreSQL',
    'oracle': 'Oracle', '오라클': 'Oracle', 'mssql': 'MSSQL', 
    'mariadb': 'MariaDB', '마리아디비': 'MariaDB',
    'mongodb': 'MongoDB', '몽고디비': 'MongoDB', 
    'redis': 'Redis', '레디스': 'Redis', 
    'elasticsearch': 'Elasticsearch', '엘라스틱서치': 'Elasticsearch', 
    'dynamodb': 'DynamoDB', '다이나모디비': 'DynamoDB',
    'cassandra': 'Cassandra', '카산드라': 'Cassandra', # 추가됨
    'sqlite': 'SQLite', 'firebase': 'Firebase', 'firestore': 'Firestore', '파이어베이스': 'Firebase', # 추가됨
    # 데이터 파이프라인 & 분산처리 (추가됨)
    'kafka': 'Kafka', '카프카': 'Kafka', 'rabbitmq': 'RabbitMQ', '래빗mq': 'RabbitMQ', 
    'hadoop': 'Hadoop', '하둡': 'Hadoop', 'spark': 'Spark', '스파크': 'Spark',
    'airflow': 'Airflow', '에어플로우': 'Airflow', 'snowflake': 'Snowflake',
    'microsoft sql server': 'MSSQL', 
    # ==========================================
    # 🔹 인프라 / 클라우드 / DevOps
    # ==========================================
    'aws': 'AWS', 'amazonwebservices': 'AWS', 
    'gcp': 'GCP', 'googlecloudplatform': 'GCP', 
    'azure': 'Azure', '애저': 'Azure',
    'docker': 'Docker', '도커': 'Docker', 
    'kubernetes': 'Kubernetes', 'k8s': 'Kubernetes', '쿠버네티스': 'Kubernetes',
    'jenkins': 'Jenkins', '젠킨스': 'Jenkins', 
    'githubactions': 'GitHub Actions', 'githubaction': 'GitHub Actions',
    'cicd': 'CI/CD', 
    'terraform': 'Terraform', '테라폼': 'Terraform', 
    'linux': 'Linux', '리눅스': 'Linux', 
    'ubuntu': 'Ubuntu', '우분투': 'Ubuntu', 'centos': 'CentOS', '센트오에스': 'CentOS',
    # 웹 서버 & 모니터링 & 컨테이너 (현업 필수 추가됨)
    'nginx': 'Nginx', '엔진엑스': 'Nginx', 'apache': 'Apache', '아파치': 'Apache',
    'prometheus': 'Prometheus', '프로메테우스': 'Prometheus', 'grafana': 'Grafana', '그라파나': 'Grafana',
    'datadog': 'Datadog', '데이터독': 'Datadog', 'elk': 'ELK Stack', 'elk스택': 'ELK Stack',
    'argocd': 'ArgoCD', '아르고cd': 'ArgoCD', 'serverless': 'Serverless',
    
    # ==========================================
    # 🔹 AI / ML / Data Science / Vision
    # ==========================================
    'ai': 'AI', '인공지능': 'AI', 
    'ml': 'ML', 'machinelearning': 'ML', '머신러닝': 'ML',
    'dl': 'Deep Learning', 'deeplearning': 'Deep Learning', '딥러닝': 'Deep Learning',
    'llm': 'LLM', '대규모언어모델': 'LLM', 
    'generativeai': 'Generative AI', '생성형ai': 'Generative AI',
    'computervision': 'Computer Vision', '컴퓨터비전': 'Computer Vision', 'cv': 'Computer Vision',
    'ocr': 'OCR', 'nlp': 'NLP', '자연어처리': 'NLP', # 추가됨
    'tensorflow': 'TensorFlow', '텐서플로우': 'TensorFlow', 
    'pytorch': 'PyTorch', '파이토치': 'PyTorch', 
    'keras': 'Keras', '케라스': 'Keras',
    'yolo': 'YOLO', '욜로': 'YOLO', 
    'mediapipe': 'MediaPipe', '미디어파이프': 'MediaPipe', 
    'opencv': 'OpenCV', '오픈cv': 'OpenCV',
    'pandas': 'Pandas', '판다스': 'Pandas', 
    'numpy': 'NumPy', '넘파이': 'NumPy', 
    'scikitlearn': 'Scikit-Learn', '사이킷런': 'Scikit-Learn', 
    'huggingface': 'Hugging Face', '허깅페이스': 'Hugging Face',
    # 최신 AI 생태계 & MLOps (추가됨)
    'langchain': 'LangChain', '랭체인': 'LangChain', 'llamaindex': 'LlamaIndex',
    'vectordb': 'Vector DB', '벡터db': 'Vector DB', 'milvus': 'Milvus', 'chromadb': 'ChromaDB', 'pinecone': 'Pinecone',
    'mlops': 'MLOps', 'mlflow': 'MLflow', 'kubeflow': 'Kubeflow',
    'xgboost': 'XGBoost', 'lightgbm': 'LightGBM', 'claude code': 'Claude Code',
    
    # ==========================================
    # 🔹 개발 도구 및 환경 (Tools & IDE)
    # ==========================================
    'git': 'Git', '깃': 'Git', 
    'github': 'GitHub', '깃허브': 'GitHub', 
    'gitlab': 'GitLab', '깃랩': 'GitLab', 'bitbucket': 'Bitbucket', # 추가됨
    'jupyter': 'Jupyter Notebook', 'jupyternotebook': 'Jupyter Notebook', '주피터노트북': 'Jupyter Notebook',
    'vscode': 'VS Code', 'visualstudiocode': 'VS Code', 'visualstudio': 'Visual Studio',
    'intellij': 'IntelliJ', '인텔리제이': 'IntelliJ', 'eclipse': 'Eclipse', '이클립스': 'Eclipse', # 추가됨
    'postman': 'Postman', '포스트맨': 'Postman', 'swagger': 'Swagger', '스웨거': 'Swagger', # 추가됨
    'figma': 'Figma', '피그마': 'Figma', # 프론트 협업 필수 추가됨
    'notion': 'Notion', 'jira': 'Jira', 'slack': 'Slack', 'ajax': 'Ajax', 'devops': 'DevOps', 'ai agent': 'AI Agent'
    ,# 🔹 패키지 매니저 및 빌드 툴 (추가)
    'pnpm': 'pnpm', 'npm': 'npm', 'yarn': 'Yarn', 'webpack': 'Webpack', '웹팩': 'Webpack', 'vite': 'Vite', '바이트': 'Vite'
}

# --- 공백/점 표기 보강 --------------------------------------------------------
# 원본 사전은 `springboot` 처럼 **공백을 제거한 형태**만 별칭으로 갖고 있다.
# 노트북에서는 조회 전에 `re.sub(r'[^a-z0-9+#.]', '', ...)` 로 공백을 지웠기
# 때문에 문제가 없었지만, extract_techs 는 원문을 그대로 훑기 때문에
# "Spring Boot" 가 짧은 별칭 `spring` 에 먼저 걸려 'Spring' 으로 뭉개진다.
#
# 데이터셋에는 Spring(2,505건)과 Spring Boot(126건)가 **둘 다** 있으므로
# 이 뭉개짐은 실제 매칭 손실이다. 표준명의 공백형을 별칭으로 되먹인다.
# (긴 별칭 우선 매칭이므로 'spring boot' 가 'spring' 을 이긴다)
MASTER_TECH_DICT.update({
    v.lower(): v for v in set(MASTER_TECH_DICT.values())
    if (' ' in v or '.' in v) and v.lower() not in MASTER_TECH_DICT
})

# ---------------------------------------------------------------------------
# 2. 불용어 — 기술명처럼 생겼지만 변별력이 없는 일반어 (01/app.py 출처)
#    '시스템'·'데이터'·'네트워크'는 아무 공고에나 붙어 매칭 신호가 되지 못한다.
# ---------------------------------------------------------------------------
EXCLUDE_TECH = {
    '소프트웨어개발', '솔루션', 'SI', '시스템', '네트워크', '서버', '정보보안', 'Sm', '데이터', 'erp',
    '문서작성', '클라이언트', '유지보수', '방화벽', 'Ms office', '기술지원', '영어', '검증', '모델링',
    '전략기획', '회로설계', '재고관리', '아키텍처', '매출관리', '인터페이스', 'GUI', 'PPT', 'PM', '회계', '고객관리',
    '핀테크', '모바일앱개발', '문서관리', '보안관제', 'HTTP', '반응형웹', '포토샵'
}

# ---------------------------------------------------------------------------
# 3. 추출
# ---------------------------------------------------------------------------
# 긴 별칭부터 매칭한다. 'spring boot' 를 'spring' 보다 먼저 봐야
# "Spring Boot" 가 "Spring" 으로 뭉개지지 않는다.
_SORTED_KEYS = sorted(MASTER_TECH_DICT.keys(), key=len, reverse=True)

# 단어 경계. 앞뒤가 영숫자면 매칭하지 않는다 — 'java' 가 'javascript' 안에서
# 잡히는 것을 막는다. 한글은 경계로 치지 않으므로 "파이썬으로"에서 "파이썬"이 잡힌다.
_BOUNDARY = r'(?<![a-z0-9]){}(?![a-z0-9])'

_fuzzy_cache = {}


def extract_techs(text, exclude_stopwords=False):
    """자유 텍스트에서 표준 기술명 집합을 뽑는다.

    이력서 본문, 공고 본문, 스킬 문자열 어디에나 쓸 수 있다.
    매칭된 부분은 텍스트에서 지워가며 진행해 중복·간섭을 막는다
    (예: "Spring Boot" 를 잡은 뒤 남은 문자열에서 "Boot" 를 또 찾지 않는다)."""
    t = str(text or '').lower().strip()
    found = set()
    for key in _SORTED_KEYS:
        if not t.strip():
            break
        pat = _BOUNDARY.format(re.escape(key))
        if re.search(pat, t):
            found.add(MASTER_TECH_DICT[key])
            t = re.sub(pat, ' ', t)
    if exclude_stopwords:
        found -= EXCLUDE_TECH
    return found


def canonical(skill, fuzzy=True):
    """스킬 문자열 하나를 표준명으로. 못 찾으면 None.

    extract_techs 가 실패했을 때의 폴백 경로다. 노트북의
    merge_dedup_and_fuzzy_skills 에 있던 특수 케이스와 80% 유사도 매칭을 옮겼다."""
    original = str(skill or '').strip()
    if not original:
        return None

    got = extract_techs(original)
    if len(got) == 1:
        return next(iter(got))

    norm = re.sub(r'[^a-z0-9+#.]', '', original.lower()).strip()
    if not norm:
        return None                                   # 순수 한글 직무명 등

    # 버전 표기 흡수: java17 / jdk11 / vue3 / .netframework
    if norm.startswith(('.net', 'netframework', 'asp.net')):
        return '.NET'
    for pre, canon in (('java', 'Java'), ('jdk', 'Java')):
        if norm.startswith(pre) and norm[len(pre):].isdigit():
            return canon
    if norm.startswith('vue') and norm[3:].replace('js', '').isdigit():
        return 'Vue.js'

    if norm in MASTER_TECH_DICT:
        return MASTER_TECH_DICT[norm]

    if fuzzy and len(norm) >= 3:                      # 오타 보정
        if norm in _fuzzy_cache:
            hit = _fuzzy_cache[norm]
            return MASTER_TECH_DICT[hit] if hit else None
        matches = difflib.get_close_matches(norm, MASTER_TECH_DICT.keys(), n=1, cutoff=0.8)
        _fuzzy_cache[norm] = matches[0] if matches else None
        return MASTER_TECH_DICT[matches[0]] if matches else None
    return None


def normalize_list(skills, exclude_stopwords=False):
    """스킬 리스트를 표준명 집합으로. 공고의 final_techs/hard_skills 정리용."""
    out = set()
    for s in skills or []:
        got = extract_techs(s)
        if got:
            out |= got
            continue
        c = canonical(s)
        if c:
            out.add(c)
    if exclude_stopwords:
        out -= EXCLUDE_TECH
    return out


ALL_TECHS = frozenset(MASTER_TECH_DICT.values())


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    print(f"별칭 {len(MASTER_TECH_DICT)}개 -> 표준명 {len(ALL_TECHS)}종 "
          f"(한글 별칭 {sum(1 for k in MASTER_TECH_DICT if re.search('[가-힣]', k))}개)")
    print(f"불용어 {len(EXCLUDE_TECH)}개\n")
    for s in ["파이썬과 장고로 결제 API를 개발했습니다",
              "Spring Boot, JPA 기반 백엔드 3년",
              "AWS EKS에서 쿠버네티스 운영 경험",
              "타입스크립트 + 리액트로 프론트 개발"]:
        print(f"  {s}\n    -> {sorted(extract_techs(s))}")
