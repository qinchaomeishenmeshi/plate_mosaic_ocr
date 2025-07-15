import os
import shutil
import time
import requests
import zipfile
from ultralytics import YOLO

# --- YOLOv8 模型下载 ---
def download_yolo_model():
    """下载并安放YOLOv8模型"""
    print("--- 正在检查YOLOv8模型 ---")
    model_dir = "models"
    model_path = os.path.join(model_dir, "yolov8n.pt")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(model_path):
        print(f"模型文件未找到: {model_path}")
        print("正在自动下载YOLOv8n模型...")
        try:
            # YOLO会自动下载模型到当前工作目录
            temp_model = YOLO("yolov8n.pt")
            # 将下载的模型移动到models目录
            if os.path.exists("yolov8n.pt"):
                shutil.move("yolov8n.pt", model_path)
                print(f"✅ 模型已下载并保存到: {model_path}")
        except Exception as e:
            print(f"❌ 自动下载YOLOv8模型失败: {e}")
            raise
    else:
        print("✅ YOLOv8模型已存在。")

# --- HyperLPR 模型下载 (带重试机制) ---
def download_hyperlpr_models_with_retry(retries=5, delay=3):
    """
    初始化HyperLPR以触发模型下载，包含针对CI环境中常见权限错误的重试逻辑。
    """
    print("\n--- 正在检查HyperLPR模型 ---")
    for i in range(retries):
        try:
            # 导入并初始化，这将触发内置的模型下载
            import hyperlpr3 as lpr3
            lpr3.LicensePlateCatcher()
            print("✅ HyperLPR模型已下载或已存在。")
            return True
        except PermissionError as e:
            print(f"⚠️ 第 {i + 1}/{retries} 次尝试失败，发生权限错误: {e}")
            if i < retries - 1:
                print(f"   将在 {delay} 秒后重试...")
                time.sleep(delay)
            else:
                print("❌ 重试次数已用完，HyperLPR模型初始化失败。")
                # 在CI环境中，即使失败也选择不抛出异常，以允许构建继续
                # 最终的EXE中将不包含HyperLPR功能
                return False
        except Exception as e:
            print(f"❌ HyperLPR初始化时发生未知错误: {e}")
            raise

if __name__ == "__main__":
    try:
        download_yolo_model()
        download_hyperlpr_models_with_retry()
        print("\n🎉 模型准备完成。")
    except Exception as e:
        print(f"\n❌ 模型准备过程中发生严重错误: {e}")
        # 抛出非零退出码，让CI/CD识别到失败
        exit(1) 