# 메모리 반도체 소재 발굴 플랫폼

Materials Project DFT 데이터 기반의 메모리 반도체 소재(고유전율 high-k · 강유전체
FeRAM/FeFET · NAND 터널/블로킹 산화물 · 저항변화 RRAM) 스크리닝 + 물리 시뮬레이션 +
조합 탐색 + Claude AI 어시스턴트 대시보드 (Streamlit).

## 구성
- `app.py` — Streamlit 대시보드(사이드바 스크리닝, 8개 탭, 비밀번호 게이트)
- `data.py` — Materials Project 조회 + 파생물성 + 필터
- `physics.py` — 유도 물성/지표(κ·Eg, EOT, ALD 합성성, 발굴점수 등)
- `chatbot.py` — Claude tool-use AI 어시스턴트(자연어 → 스크리닝)

## 로컬 실행
```bash
# 의존성 (uv 또는 pip)
pip install -r requirements.txt        # Python 3.12 권장

# 키 설정: .streamlit/secrets.toml.example 를 복사해 값 채우기
cp .streamlit/secrets.toml.example .streamlit/secrets.toml

streamlit run app.py
```

## 필요한 키 (secrets)
| 키 | 용도 | 필수 | 발급 |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | 챗봇(Claude) | 필수 | console.anthropic.com |
| `APP_PASSWORD` | 접근 비밀번호 게이트 | 선택(공개 배포 시 권장) | 직접 지정 |
| `MP_API_KEY` | Materials Project 조회 | 선택 | data.py에 무료 기본 키 내장(미설정 시 사용) |

> ⚠️ `.streamlit/secrets.toml` 은 절대 커밋하지 마세요(.gitignore로 제외됨).
> MP는 무료 API라 기본 키가 코드에 내장돼 있습니다(공개 저장소에 노출되나 비용 위험 없음).

## Streamlit Community Cloud 배포
1. 이 폴더를 GitHub 저장소에 푸시(아래 "최초 푸시" 참고).
2. https://share.streamlit.io → **Create app** → 저장소/브랜치 선택, Main file = `app.py`.
3. **Advanced settings → Python version = 3.12** 선택.
4. **Settings → Secrets** 에 키를 TOML 형식으로 입력(MP는 내장 기본 키라 생략 가능):
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   APP_PASSWORD = "강한-비밀번호"
   ```
5. **Deploy** — 빌드 후 공개 URL 발급. 비밀번호를 아는 사람만 입장.

### 최초 푸시
```bash
git init
git add .
git commit -m "Initial deploy: memory-semiconductor screening dashboard"
git branch -M main
git remote add origin https://github.com/<USER>/<REPO>.git
git push -u origin main
```
