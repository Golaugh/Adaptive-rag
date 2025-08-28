# check_dotenv.py
import sys

print("当前 Python 路径列表：")
for p in sys.path:
    print("  ", p)

try:
    import dotenv
    print("\n✅ 成功导入 dotenv")
    print("dotenv 实际位置：", dotenv.__file__)
    print("dotenv 包里可用的属性：", dir(dotenv))
except Exception as e:
    print("\n❌ 导入 dotenv 失败：", e)
