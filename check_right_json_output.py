# 논문 요약을 추출하면 markdown 파일로 주거나 깨지는데 이를 판별한다.
# 올바른 json을 포함한다면 json 파일로 변환하고 그렇지 않다면 에러를 출력하고 다음 내용을 처리한다.

from typing import Optional, Any, Dict
import os
import json


def turn_into_right_json(file_path: str) -> Optional[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    content = content.replace('```json', '').replace('```', '')
    try:
        results = json.loads(content)
    except:
        print(f"not a json format: {file_path}")
        return file_path
    keys = results.keys()
    for must_key in ['id', 'overall_summary', 'methods', 'result']:
        if must_key not in keys:
            print(f"{must_key} not found: {file_path}")
            return file_path
    if type(results['methods']) is list and len(results['methods']) > 0:
        file_name = file_path.rsplit('.', 2)[0]
        with open(f"{file_name}.json", 'w') as f:
            f.write(json.dumps(results))
        return None
    else:
        print(f"methods is not list or has zero length: {file_path}")
        return file_path


error_file_list = []
for file in [file for file in os.listdir('qweasd') if file.endswith('.md')]:
    error = turn_into_right_json(f'qweasd/{file}')
    if error is None:
        error_file_list.append(error)
