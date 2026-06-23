"""demo.py를 stlite(WebAssembly) 단일 HTML로 감싼다 → demo.html.

demo.html은 파이썬·설치 없이 브라우저에서 열기만 하면 동작한다(첫 실행 시
런타임을 CDN에서 받으므로 인터넷 1회 필요; 이후 클라이언트에서 실행).
실행: ./.venv/bin/python build_html.py
"""
import json

STLITE = "1.8.1"
code = open("demo.py").read()
# JS 문자열 리터럴로 안전하게 임베드(이스케이프) + </script> 조기종료 방지
code_js = json.dumps(code).replace("</", "<\\/")

HTML = """<!doctype html>
<html lang="ko">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>메모리 반도체 소재 발굴 — 오프라인 데모</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@stlite/browser@__V__/build/stlite.css" />
<style>
  #boot { font-family: -apple-system, sans-serif; padding: 2.5rem; color:#1f4e79; line-height:1.6; }
  #boot b { font-size:1.1rem; }
</style>
</head>
<body>
<div id="root"><div id="boot"><b>⏳ 데모 로딩 중…</b><br>
첫 실행은 브라우저에 파이썬 런타임을 받느라 <b>20~40초</b> 걸립니다(인터넷 1회 필요).<br>
이후에는 파이썬·설치 없이 이 파일 하나로 동작합니다.</div></div>
<script type="module">
import { mount } from "https://cdn.jsdelivr.net/npm/@stlite/browser@__V__/build/stlite.js";
mount(
  {
    requirements: ["plotly", "pandas", "numpy"],
    entrypoint: "demo.py",
    files: { "demo.py": __CODE__ },
  },
  document.getElementById("root"),
);
</script>
</body>
</html>
"""

out = HTML.replace("__V__", STLITE).replace("__CODE__", code_js)
with open("demo.html", "w") as f:
    f.write(out)
print("demo.html 생성 완료:", len(out), "bytes")
