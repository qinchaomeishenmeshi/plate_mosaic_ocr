import os
import cv2
from ultralytics import YOLO
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import shutil
import numpy as np
import argparse
import sys

# 导入HyperLPR
try:
    import hyperlpr3 as lpr3
    catcher = lpr3.LicensePlateCatcher()
    HYPERLPR_AVAILABLE = True
    print("✅ HyperLPR 导入成功！")
except ImportError:
    catcher = None
    HYPERLPR_AVAILABLE = False
    print("⚠️ HyperLPR 未安装或导入失败，将使用基于车辆区域的检测方法")
    print("   请确保已正确安装: pip install hyperlpr3")

# --- 全局配置与模型加载 ---
MODEL_PATH = "models/yolov8n.pt"  # 使用通用YOLOv8模型检测汽车


def ensure_model_exists():
    """确保模型文件存在，如果不存在则自动下载"""
    if not os.path.exists("models"):
        os.makedirs("models")

    if not os.path.exists(MODEL_PATH):
        print(f"模型文件未找到: {MODEL_PATH}")
        print("正在自动下载YOLOv8n模型...")
        try:
            # YOLO会自动下载模型到当前目录
            temp_model = YOLO("yolov8n.pt")
            # 将下载的模型移动到models目录
            if os.path.exists("yolov8n.pt"):
                shutil.move("yolov8n.pt", MODEL_PATH)
                print(f"✅ 模型已下载并保存到: {MODEL_PATH}")
            return temp_model
        except Exception as e:
            print(f"❌ 自动下载模型失败: {e}")
            print("请手动下载yolov8n.pt模型文件到models/目录")
            return None
    else:
        return YOLO(MODEL_PATH)


# 全局加载模型，避免重复加载
try:
    print("正在加载YOLOv8模型...")
    model = ensure_model_exists()
    if model:
        print("✅ 模型加载成功！")
    else:
        print("❌ 模型加载失败")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    model = None

DEFAULT_MOSAIC_SCALE = 0.05  # 车牌打码更细致
DEFAULT_PADDING = 10
MAX_WORKERS = min(6, cpu_count())  # 限制最大并发进程数

# 车牌检测相关配置
CAR_CLASSES = [2, 3, 5, 7]  # COCO数据集中的汽车类别: car, motorcycle, bus, truck

# 检查HyperLPR可用性
if HYPERLPR_AVAILABLE:
    print("✅ 将使用HyperLPR进行车牌检测")
else:
    print("⚠️ HyperLPR不可用，将使用基于颜色和形状的方法检测车牌")
    print("建议安装HyperLPR以获得更好的检测效果：pip install hyperlpr3")

# --- 核心功能函数 ---


def detect_license_plates_with_hyperlpr(catcher_instance, frame):
    """
    使用HyperLPR的catcher实例检测整个画面中的车牌
    """
    plates = []
    if not catcher_instance:
        return plates

    try:
        results = catcher_instance(frame)
        for result in results:
            # 增加健壮性检查
            if isinstance(result, (list, tuple)) and len(result) >= 4:  # 确保结果至少有4个元素
                confidence = result[1]
                # 修正：根据日志，bbox在结果的第4个位置（索引为3）
                bbox = result[3]
                if confidence > 0.7:
                    # 核心修复：检查bbox是否是可迭代的4元素对象
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        plates.append((int(x1), int(y1), int(x2 - x1), int(int(y2 - y1))))
                    else:
                        # 如果bbox格式不正确，打印更详细的日志
                        print(f"⚠️ HyperLPR返回了异常的结果格式，bbox不正确。完整结果: {result}")
    except Exception as e:
        print(f"⚠️ HyperLPR检测失败: {e}")
    return plates


def detect_license_plates_in_car(catcher_instance, car_roi):
    """
    在汽车区域内检测车牌位置
    """
    plates = []
    # 方法1: 使用HyperLPR检测车牌
    if catcher_instance:
        try:
            results = catcher_instance(car_roi)
            for result in results:
                if isinstance(result, (list, tuple)) and len(result) >= 4: # 确保结果至少有4个元素
                    confidence = result[1]
                    # 修正：根据日志，bbox在结果的第4个位置（索引为3）
                    bbox = result[3]
                    if confidence > 0.6:
                        # 核心修复：检查bbox是否是可迭代的4元素对象
                        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            plates.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
                        else:
                            print(f"⚠️ HyperLPR在车辆区域检测中返回了异常的结果格式。完整结果: {result}")
        except Exception as e:
            print(f"⚠️ HyperLPR车辆区域检测失败: {e}")

    # 方法2: 基于颜色和形状的启发式检测（备用方案）
    if not plates:
        plates = detect_plates_by_color_and_shape(car_roi)
    return plates


def detect_plates_by_color_and_shape(car_roi):
    """
    基于颜色和形状特征检测车牌
    """
    plates = []
    h, w = car_roi.shape[:2]

    # 转换到HSV颜色空间
    hsv = cv2.cvtColor(car_roi, cv2.COLOR_BGR2HSV)

    # 定义白色和黄色车牌的HSV范围
    white_lower = np.array([0, 0, 180])
    white_upper = np.array([180, 30, 255])
    yellow_lower = np.array([15, 100, 100])
    yellow_upper = np.array([35, 255, 255])

    # 创建掩码
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    mask = cv2.bitwise_or(white_mask, yellow_mask)

    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:  # 过滤小区域
            continue

        x, y, w_rect, h_rect = cv2.boundingRect(contour)
        aspect_ratio = w_rect / h_rect

        # 车牌通常宽高比在2:1到5:1之间
        if 2.0 <= aspect_ratio <= 5.0 and area > 1000:
            plates.append((x, y, w_rect, h_rect))

    return plates


def estimate_plate_region(car_x1, car_y1, car_x2, car_y2):
    """
    根据车辆位置估算车牌可能的区域
    """
    car_w = car_x2 - car_x1
    car_h = car_y2 - car_y1

    # 车牌通常在车辆底部1/3区域
    plate_y1 = car_y1 + int(car_h * 0.6)
    plate_y2 = car_y2

    # 车牌宽度通常占车辆宽度的1/3到2/3
    plate_w = int(car_w * 0.5)
    plate_x1 = car_x1 + (car_w - plate_w) // 2
    plate_x2 = plate_x1 + plate_w

    return plate_x1, plate_y1, plate_x2, plate_y2


def mosaic_region(
    frame, x1, y1, x2, y2, mosaic_scale=DEFAULT_MOSAIC_SCALE, padding=DEFAULT_PADDING
):
    """
    对指定区域进行马赛克处理
    """
    h, w = frame.shape[:2]
    x1 = max(0, int(x1 - padding))
    y1 = max(0, int(y1 - padding))
    x2 = min(w, int(x2 + padding))
    y2 = min(h, int(y2 + padding))

    region_w, region_h = x2 - x1, y2 - y1
    roi = frame[y1:y2, x1:x2]

    if region_w <= 0 or region_h <= 0:
        return frame

    try:
        # 使用马赛克效果
        small = cv2.resize(
            roi,
            (0, 0),
            fx=mosaic_scale,
            fy=mosaic_scale,
            interpolation=cv2.INTER_LINEAR,
        )
        mosaic = cv2.resize(
            small, (region_w, region_h), interpolation=cv2.INTER_NEAREST
        )
        roi[:] = mosaic
    except Exception as e:
        print(f"⚠️ 打码失败: {e}")

    return frame


def process_video_file(
    processing_model,
    catcher_instance,
    hyperlpr_enabled: bool,
    video_path: str,
    output_path: str,
    mosaic_scale: float,
    conf: float,
    iou: float,
    detection_mode: str = "hyperlpr",
):
    """
    处理单个视频文件，进行车牌检测和打码。
    """
    if not processing_model:
        raise RuntimeError("YOLOv8模型未能成功加载，无法处理视频。")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {video_path}")

    # 根据文档，HyperLPR是端到端车牌识别框架，直接使用它应该是最高效的。
    # 'hyperlpr' 模式直接在整个帧上运行检测，这是推荐的方式。
    # 'precise' 模式先用YOLO检测车辆，再在车辆区域内检测车牌，步骤更复杂，作为备选。
    # 'estimate' 模式是当以上方法都不可用时的基本备用方案。

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with tqdm(
            total=total_frames,
            desc=f"正在处理 {os.path.basename(video_path)} (模式: {detection_mode})",
            unit="帧",
        ) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # --- 重构后的检测逻辑 ---

                # 模式1: 直接使用HyperLPR进行端到端检测 (推荐)
                if detection_mode == "hyperlpr":
                    if hyperlpr_enabled:
                        plates = detect_license_plates_with_hyperlpr(catcher_instance, frame)
                        for px, py, pw, ph in plates:
                            frame = mosaic_region(
                                frame, px, py, px + pw, py + ph, mosaic_scale, DEFAULT_PADDING
                            )
                    else:
                        # 如果选择了hyperlpr模式但库不可用，打印警告并且不处理该帧
                        print(f"⚠️ 'hyperlpr' 模式需要HyperLPR库，但它不可用。该帧将不被处理。")

                # 模式2 & 3: 基于YOLO的检测流程
                else:
                    results = processing_model.predict(frame, conf=conf, iou=iou, verbose=False)
                    boxes = results[0].boxes
                    if boxes is not None:
                        xyxy = boxes.xyxy.cpu().numpy()
                        cls = boxes.cls.cpu().numpy()
                        for i, (x1, y1, x2, y2) in enumerate(xyxy):
                            if int(cls[i]) in CAR_CLASSES:
                                # 模式2: 在YOLO检测到的车辆区域内进行精确检测
                                if detection_mode == "precise":
                                    # 即使hyperlpr不可用，此模式仍可回退到颜色形状检测
                                    if not hyperlpr_enabled:
                                        print(f"⚠️ HyperLPR不可用，'precise' 模式将使用备用的颜色形状检测。")
                                    car_roi = frame[int(y1):int(y2), int(x1):int(x2)]
                                    if car_roi.size > 0:
                                        plates = detect_license_plates_in_car(catcher_instance, car_roi)
                                        for px, py, pw, ph in plates:
                                            plate_x1 = int(x1) + px
                                            plate_y1 = int(y1) + py
                                            plate_x2 = plate_x1 + pw
                                            plate_y2 = plate_y1 + ph
                                            frame = mosaic_region(
                                                frame, plate_x1, plate_y1, plate_x2, plate_y2, mosaic_scale, DEFAULT_PADDING
                                            )
                                # 模式3: 根据车辆位置估算车牌区域
                                elif detection_mode == "estimate":
                                    plate_x1, plate_y1, plate_x2, plate_y2 = estimate_plate_region(x1, y1, x2, y2)
                                    frame = mosaic_region(
                                        frame, plate_x1, plate_y1, plate_x2, plate_y2, mosaic_scale, DEFAULT_PADDING
                                    )
                
                out.write(frame)
                pbar.update(1)

    finally:
        cap.release()
        if "out" in locals():
            out.release()


def init_worker():
    """
    多进程工作器初始化函数，用于加载模型。
    """
    global worker_model, worker_catcher
    print(f"初始化工作进程 {os.getpid()}...")
    worker_model = ensure_model_exists()
    if worker_model:
        print(f"✅ 工作进程 {os.getpid()} YOLOv8模型加载成功")
    else:
        print(f"❌ 工作进程 {os.getpid()} YOLOv8模型加载失败")

    # 为每个工作进程初始化HyperLPR
    try:
        import hyperlpr3 as lpr3
        worker_catcher = lpr3.LicensePlateCatcher()
        print(f"✅ 工作进程 {os.getpid()} HyperLPR加载成功")
    except ImportError:
        worker_catcher = None
        print(f"⚠️ 工作进程 {os.getpid()} HyperLPR不可用")


def process_video_multiprocess(args):
    """
    用于多进程处理的包装函数。
    """
    video_path, output_folder, mosaic_scale, conf, iou, detection_mode = args
    video_name = os.path.basename(video_path)
    output_path = os.path.join(output_folder, f"plate_mosaic_{video_name}")

    try:
        # 从全局变量获取模型
        if 'worker_model' not in globals() or not worker_model:
            raise RuntimeError(f"工作进程 {os.getpid()} 中的YOLOv8模型不可用。")

        # HyperLPR在工作进程中也应可用
        hyperlpr_in_worker = 'worker_catcher' in globals() and worker_catcher is not None

        process_video_file(
            worker_model, worker_catcher, hyperlpr_in_worker, video_path, output_path, mosaic_scale, conf, iou, detection_mode
        )
        return video_name, True, output_path
    except Exception as e:
        return video_name, False, str(e)


def batch_process(
    input_folder,
    output_folder,
    mosaic_scale=DEFAULT_MOSAIC_SCALE,
    conf=0.3,
    iou=0.3,
    detection_mode="hyperlpr",  # 命令行默认使用hyperlpr
):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
    ]
    if not video_files:
        print("指定目录中没有找到视频文件。")
        return

    args_list = [
        (vf, output_folder, mosaic_scale, conf, iou, detection_mode)
        for vf in video_files
    ]

    print(
        f"📦 共检测到 {len(video_files)} 个视频，使用 {MAX_WORKERS} 个进程开始处理..."
    )

    results = []
    with Pool(processes=MAX_WORKERS, initializer=init_worker) as pool:
        for res in tqdm(
            pool.imap_unordered(process_video_multiprocess, args_list),
            total=len(args_list),
            desc="总体进度",
        ):
            results.append(res)

    print("\n📋 处理结果：")
    for name, ok, msg in sorted(results):
        status = f"✅ 成功 → {msg}" if ok else f"❌ 失败 ({msg})"
        print(f"   - {name}: {status}")


def main():
    """主函数，处理命令行参数和交互式配置"""
    parser = argparse.ArgumentParser(
        description="视频车牌打码工具，支持命令行批量处理和交互式配置。",
        formatter_class=argparse.RawTextHelpFormatter  # 保持帮助信息格式
    )

    parser.add_argument(
        "-i", "--input", help="包含视频的输入文件夹。如果未提供，将进入交互模式。"
    )
    parser.add_argument(
        "-o", "--output", help="用于保存处理后视频的输出文件夹。如果未提供，将进入交互模式。"
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=float,
        default=DEFAULT_MOSAIC_SCALE,
        help=f"马赛克程度 (默认: {DEFAULT_MOSAIC_SCALE})。",
    )
    parser.add_argument(
        "--conf", type=float, default=0.3, help="置信度阈值 (默认: 0.3)。"
    )
    parser.add_argument(
        "--iou", type=float, default=0.3, help="IOU阈值 (默认: 0.3)。"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hyperlpr",
        choices=["hyperlpr", "precise", "estimate"],
        help="检测模式:\n"
             "  hyperlpr: (推荐)直接检测车牌，速度快效果好。\n"
             "  precise: 先检测车辆再检测车牌，更精确但稍慢。\n"
             "  estimate: 估算车牌位置，速度最快效果最差。\n"
             "(默认: hyperlpr)。",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="强制进入交互模式以配置所有参数。",
    )

    args = parser.parse_args()

    # 检查是否需要进入交互模式
    if args.interactive or not args.input or not args.output:
        print("🚀 进入交互式配置模式...")

        # 1. 获取输入文件夹
        if not args.input:
            while True:
                input_folder = input("➡️ 请输入包含视频的输入文件夹路径: ")
                if os.path.isdir(input_folder):
                    args.input = input_folder
                    break
                else:
                    print(f"❌ 错误: 路径 '{input_folder}' 不是一个有效的文件夹。请重试。")
        
        # 2. 获取输出文件夹
        if not args.output:
            output_folder = input("➡️ 请输入用于保存结果的输出文件夹路径 (默认: ./output): ")
            if not output_folder:
                output_folder = "output"
            args.output = output_folder

        # 3. 选择检测模式
        print("\n⚙️ 请选择车牌检测模式:")
        mode_choices = ["hyperlpr", "precise", "estimate"]
        mode_descriptions = [
            "hyperlpr (推荐): 直接使用HyperLPR进行端到端检测，速度快，效果好。",
            "precise: 先用YOLO检测车辆，再在车内检测车牌，更精确但稍慢。",
            "estimate: 在YOLO检测到的车辆上估算车牌位置，速度最快，效果最差。",
        ]
        for i, desc in enumerate(mode_descriptions):
            print(f"  {i+1}. {desc}")

        while True:
            choice = input(f"请选择模式 (1/2/3) [默认: {args.mode}]: ")
            if not choice:
                break # 使用默认值
            if choice in ["1", "2", "3"]:
                args.mode = mode_choices[int(choice) - 1]
                break
            else:
                print("❌ 无效输入，请输入 1, 2, 或 3。")

        # 4. 配置高级参数
        configure_advanced = input("\n🔧 是否需要配置高级参数 (如马赛克程度)? (y/N): ")
        if configure_advanced.lower() == 'y':
            # 马赛克程度
            while True:
                scale_str = input(f"  - 请输入马赛克程度 (0.01-1.0) [默认: {args.scale}]: ")
                if not scale_str: break
                try:
                    scale_val = float(scale_str)
                    if 0.01 <= scale_val <= 1.0:
                        args.scale = scale_val
                        break
                    else: print("❌ 无效范围。")
                except ValueError: print("❌ 无效输入。")

            # 置信度
            while True:
                conf_str = input(f"  - 请输入车辆检测置信度阈值 (0.1-0.9) [默认: {args.conf}]: ")
                if not conf_str: break
                try:
                    conf_val = float(conf_str)
                    if 0.1 <= conf_val <= 0.9:
                        args.conf = conf_val
                        break
                    else: print("❌ 无效范围。")
                except ValueError: print("❌ 无效输入。")
            
            # IOU
            while True:
                iou_str = input(f"  - 请输入IOU阈值 (0.1-0.9) [默认: {args.iou}]: ")
                if not iou_str: break
                try:
                    iou_val = float(iou_str)
                    if 0.1 <= iou_val <= 0.9:
                        args.iou = iou_val
                        break
                    else: print("❌ 无效范围。")
                except ValueError: print("❌ 无效输入。")

        # 最终确认
        print("\n--- ⚙️ 配置确认 ---")
        print(f"  🎬 输入文件夹: {os.path.abspath(args.input)}")
        print(f"  📂 输出文件夹: {os.path.abspath(args.output)}")
        print(f"  🔎 检测模式: {args.mode}")
        print(f"  🎨 马赛克程度: {args.scale}")
        print(f"  🎯 置信度阈值: {args.conf}")
        print(f"  📏 IOU阈值: {args.iou}")
        print("--------------------")
        
        confirm = input("确认以上配置并开始处理? (Y/n): ")
        if confirm.lower() == 'n':
            print("🔴 操作已取消。")
            sys.exit(0)

    # 检查模型是否加载成功
    if not model:
        print("❌ 错误：模型未能成功加载，无法继续执行")
        print("请确保以下条件满足：")
        print("1. 网络连接正常（首次运行需要下载模型）")
        print("2. 有足够的磁盘空间存储模型文件")
        print("3. 已安装ultralytics包：pip install ultralytics")
        sys.exit(1)

    # 开始批量处理
    batch_process(
        args.input, args.output, args.scale, args.conf, args.iou, args.mode
    )


if __name__ == "__main__":
    main()