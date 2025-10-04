import os
import time
import clipboard
import pyautogui
from tqdm import tqdm

NEW_CONVERSTION = (136, 49)
ACTIVATE_INPUT = (399, 818)
COPY_BTN = (385, 705)
CHECK_RESPONSE_END = (1218, 858)
SCREEN_CENTER = (812, 444)
SUBMIT_BTN = (1302, 857)


def is_blackish(pixel):
    """필셀값이 rgb 100,100,100 이하인 경우 True 반환"""
    r, g, b = pixel[:3]
    # print(f"RGB({r}, {g}, {b})", end=' ')
    return r <= 100 and g <= 100 and b <= 100


def safe_move(x: int, y: int):
    """Clamp coordinates so the mouse never leaves the visible screen."""
    screen_w, screen_h = pyautogui.size()
    clamped_x = max(0, min(x, screen_w - 1))
    clamped_y = max(0, min(y, screen_h - 1))
    if (clamped_x, clamped_y) != (x, y):
        print(f"safe_move: ({x},{y}) → ({clamped_x},{clamped_y})")
    pyautogui.moveTo(clamped_x, clamped_y)


def click(x: int, y: int, *, clicks: int = 1, interval: float = 0.0, wait_before: float = 0.0, wait_after: float = 0.0):
    """Wrapper around pyautogui.click that never exits the screen."""
    if wait_before > 0:
        time.sleep(wait_before)
    safe_move(x, y)
    pyautogui.click(clicks=clicks, interval=interval)
    if wait_after > 0:
        time.sleep(wait_after)


def get_answer(input: str) -> str:
    clipboard.copy(input)
    time.sleep(0.1)

    # 새 대화
    click(*NEW_CONVERSTION, clicks=2, wait_after=0.5)

    # 대화창 포커스
    click(*ACTIVATE_INPUT, wait_after=0.5)

    # 질문하기
    pyautogui.keyDown('command')
    pyautogui.press('v')
    pyautogui.keyUp('command')
    click(*SUBMIT_BTN, wait_before=0.5, wait_after=0.5)

    # 답변 대기
    for i in range(80):
        time.sleep(5)
        pixel = pyautogui.screenshot(
            region=(CHECK_RESPONSE_END[0], CHECK_RESPONSE_END[1], 1, 1)).getpixel((0, 0))
        if is_blackish(pixel):
            break

    time.sleep(2)

    # 화면 중앙으로 이동해서 최하단으로 스크롤
    safe_move(*SCREEN_CENTER)
    for _ in range(35):
        pyautogui.scroll(-50)
        time.sleep(0.1)
    time.sleep(0.2)

    # 복사
    click(*COPY_BTN)
    time.sleep(0.1)

    # 결과 반환
    return clipboard.paste()


def get_paper_content(paper_path: str) -> str:
    if not os.path.exists(paper_path):
        raise Exception("Paper Not exists")

    with open(paper_path, 'r') as f:
        content = f.read()

    if content == '':
        raise Exception("Content not exists")

    return content


if __name__ == "__main__":
    print("3초 뒤에 프로세스를 시작합니다.")
    time.sleep(3)

    summary_results_path = 'summary_results'
    os.makedirs(summary_results_path, exist_ok=True)
    pmcid_list = [file for file in os.listdir(
        'papers') if file.endswith('.md')]

    summary_prompt_file_path = 'summary_prompt.txt'
    if not os.path.exists(summary_prompt_file_path):
        raise Exception(f'Prompt file not found: {summary_prompt_file_path}')
    with open(summary_prompt_file_path, 'r') as f:
        prompt = f.read()
    if prompt == '':
        raise Exception('Prompt is empty')

    for file in tqdm(pmcid_list, total=len(pmcid_list)):
        PMCID = file.split('.')[0]
        content = get_paper_content(os.path.join('papers', file))
        input = f"{prompt}\n{content}"

        results = get_answer(input)
        with open(os.path.join(summary_results_path, file), 'w') as f:
            f.write(results)
