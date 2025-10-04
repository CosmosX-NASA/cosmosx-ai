import pyautogui
import time

print("Ctrl+C를 눌러 종료하세요.")
try:
    while True:
        x, y = pyautogui.position()
        print(f"현재 좌표: ({x}, {y})")
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n좌표 출력 종료")
except Exception as e:
    print(f"오류 발생: {e}")
finally:
    print("프로그램 종료")
# 이 코드는 pyautogui를 사용하여 현재 마우스 커서의 좌표를 지속적으로 출력합니다.
# Ctrl+C를 눌러 종료할 수 있습니다.

"""
새 대화: (136, 49)
텍스트 입력 활성화: (399, 818)
복사: (381, 734)
답변 완료 감지: (1218, 858)
"""
