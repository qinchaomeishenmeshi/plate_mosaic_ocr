# 视频车牌智能打码工具

[![Build Status](https://github.com/actions/workflows/build.yml/badge.svg)](https://github.com/actions/workflows/build.yml)

这是一个功能强大的视频处理工具，旨在自动检测视频中的车辆车牌并应用马赛克效果。它结合了 `YOLOv8` 车辆检测和 `HyperLPR` 车牌识别技术，提供了多种检测模式以适应不同场景的需求。

为了方便使用，本项目支持**命令行批量处理**和**交互式配置**两种模式，并且可以通过 GitHub Actions 实现自动化构建，直接生成可在 Windows 上运行的 `.exe` 可执行文件。

## ✨ 主要功能

- **多种检测模式**：
  - **`hyperlpr` (推荐)**: 直接使用 `HyperLPR` 进行端到端车牌检测，速度快，识别率高。
  - **`precise` (精确)**: 先由 `YOLOv8` 定位车辆，再在车辆区域内使用 `HyperLPR` 或备用算法精确检测车牌。
  - **`estimate` (估算)**: 在 `YOLOv8` 检测到的车辆上，根据常规位置估算车牌区域并打码，速度最快。
- **批量处理**：支持一次性处理指定文件夹内的所有视频文件（`.mp4`, `.mov`, `.avi`, `.mkv`）。
- **用户友好**：
  - **交互式配置**：无需记忆命令行参数，只需运行脚本即可进入引导式配置界面。
  - **命令行模式**：支持通过参数进行全自动批处理，方便集成到其他脚本中。
- **智能模型管理**：首次运行时会自动下载所需的 `YOLOv8` 模型文件。
- **自动化构建**：配置了 GitHub Actions，每次向 `main` 分支推送代码时，会自动打包生成 Windows `.exe` 文件。
- **高度可配置**：支持自定义马赛克程度、检测置信度、IOU 阈值等高级参数。

## ⚙️ 环境准备

1.  **克隆仓库**
    ```bash
    git clone https://github.com/<你的用户名>/<你的仓库名>.git
    cd <你的仓库名>
    ```

2.  **安装依赖**
    项目所需的所有依赖库都记录在 `requirements.txt` 文件中。通过以下命令一键安装：
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 使用说明

本项目提供了两种运行方式：

### 1. 交互式模式 (推荐)

这是最简单的方式，尤其适合初次使用的用户。直接运行脚本，程序会引导你完成所有设置。

```bash
python app.py
```
或者强制进入交互模式：
```bash
python app.py --interactive
```

程序会依次询问你：
- **输入文件夹路径**：包含待处理视频的文件夹。
- **输出文件夹路径**：用于保存已打码视频的文件夹。
- **检测模式**：从 `hyperlpr`, `precise`, `estimate` 中选择一种。
- **高级参数**（可选）：是否需要自定义马赛克程度、置信度等。

在开始处理前，会显示最终的配置供你确认。

### 2. 命令行模式

如果你熟悉命令行操作或需要将此工具集成到自动化流程中，可以使用命令行参数。

**基本用法：**
```bash
python app.py -i <输入文件夹> -o <输出文件夹>
```

**示例：**
```bash
# 使用 hyperlpr 模式处理 'videos' 文件夹中的视频，并保存到 'output' 文件夹
python app.py -i ./videos -o ./output --mode hyperlpr

# 使用 precise 模式，并设置较高的马赛克程度和较低的置信度
python app.py -i ./videos -o ./output --mode precise --scale 0.1 --conf 0.25
```

#### 所有可用参数

| 参数 | 简写 | 描述 | 默认值 |
|---|---|---|---|
| `--input` | `-i` | **(必需)** 包含视频的输入文件夹。 | |
| `--output` | `-o` | **(必需)** 保存处理后视频的输出文件夹。 | |
| `--mode` | | 检测模式: `hyperlpr`, `precise`, `estimate`。 | `hyperlpr` |
| `--scale` | `-s` | 马赛克程度 (0.01-1.0)。值越小，马赛克越模糊。| `0.05` |
| `--conf` | | YOLOv8 车辆检测的置信度阈值。 | `0.3` |
| `--iou` | | YOLOv8 车辆检测的 IOU 阈值。 | `0.3` |
| `--interactive` | | 强制进入交互模式。 | |

## 🏗️ 自动化构建

本项目已配置 GitHub Actions。当你将代码推送到 `main` 或 `master` 分支时，会自动触发一个构建流程，该流程将在 Windows 环境中打包生成一个独立的 `.exe` 可执行文件。

你可以在仓库主页的 **Actions** 标签页找到构建记录，并从最新的成功运行记录中下载 **Artifacts** (`plate-mosaic-ocr-windows-exe`)。

## 📂 项目结构

```
.
├── .github/
│   └── workflows/
│       └── build.yml      # GitHub Actions 自动化构建配置文件
├── models/
│   └── yolov8n.pt         # YOLOv8 模型文件 (自动下载)
├── .gitignore             # Git 忽略文件配置
├── app.py                 # 主应用程序脚本
├── README.md              # 本文档
└── requirements.txt       # Python 依赖库列表
```

## 📄 许可证

本项目采用 [MIT License](LICENSE) 授权。 