"""
本地Client - 连接服务器，轮询任务，执行路线1/2/3

使用方式：
    python3 client.py --server http://localhost:8000

特点：
    - 每5秒轮询 /api/jobs/next 获取新任务
    - 根据任务 route 调用对应路线脚本
    - 通过 WebSocket 上报实时进度到 /ws/{task_id}
    - 完成后上传结果到 /api/analysis/tasks/{task_id}/result
"""
import argparse
import asyncio
import importlib.util
import json
import os
import sys
import time
from urllib.parse import urlsplit, urlunsplit

import httpx

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
ROUTE1_DIR = os.path.join(PROJECT_DIR, "route1")
ROUTE2_DIR = os.path.join(PROJECT_DIR, "route2")
ROUTE3_DIR = os.path.join(PROJECT_DIR, "route3")
DEFAULT_SERVER_URL = os.environ.get("CV_WEB_SERVER_URL", "http://localhost:8000")
SERVER_URL = DEFAULT_SERVER_URL
POLL_INTERVAL = 5
LEVEL_NAMES = {0: "NORMAL", 1: "MILD", 2: "MODERATE", 3: "SEVERE"}


def normalize_server_url(url: str) -> str:
    url = (url or DEFAULT_SERVER_URL).strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    return url


def build_ws_url(task_id: str) -> str:
    parsed = urlsplit(SERVER_URL)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    base_path = parsed.path.rstrip("/")
    path = f"{base_path}/ws/{task_id}"
    return urlunsplit((scheme, parsed.netloc, path, "", ""))


def clear_route_src_modules():
    for name in list(sys.modules):
        if name == "src" or name.startswith("src."):
            del sys.modules[name]


def load_route_module(route_dir: str, filename: str, module_name: str):
    clear_route_src_modules()
    module_path = os.path.join(route_dir, filename)
    sys.path.insert(0, route_dir)
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if sys.path and sys.path[0] == route_dir:
            sys.path.pop(0)


def get_video_meta(video_path: str) -> dict:
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
    }


def get_vlm_options(config: dict, route: str) -> dict:
    provider = (config.get("vlm_provider") or "none").strip().lower()
    if route == "route2" and provider == "none":
        provider = "mock"

    model = (config.get("vlm_model") or "").strip()
    configured_ollama_model = (config.get("ollama_model") or "").strip()
    ollama_model = (model if provider == "ollama" and model else "") or configured_ollama_model or "qwen3-vl:8b"
    siliconflow_model = model if provider == "siliconflow" and model else "Qwen/Qwen3-VL-32B-Instruct"

    return {
        "provider": provider,
        "vlm_enabled": provider in {"mock", "ollama", "siliconflow"},
        "trigger": config.get("vlm_trigger") or "MODERATE",
        "api_key": config.get("vlm_api_key") or "",
        "workers": int(config.get("workers") or 4),
        "ollama_host": config.get("ollama_host") or "http://localhost:11434",
        "ollama_model": ollama_model,
        "siliconflow_model": siliconflow_model,
    }


def normalize_level_distribution(level_dist: dict) -> dict:
    normalized = {name: 0 for name in LEVEL_NAMES.values()}
    for key, value in (level_dist or {}).items():
        name = LEVEL_NAMES.get(key, LEVEL_NAMES.get(int(key), str(key)) if str(key).isdigit() else str(key))
        normalized[name] = normalized.get(name, 0) + int(value or 0)
    return normalized


def level_percentages(level_dist: dict) -> dict:
    total = max(sum(level_dist.values()), 1)
    return {key: round(value / total * 100, 1) for key, value in level_dist.items()}


async def ws_connect(ws_url: str, msg: dict):
    import websockets

    async with websockets.connect(ws_url) as ws:
        await ws.send(json.dumps(msg, ensure_ascii=False))
        try:
            await asyncio.wait_for(ws.recv(), timeout=3)
        except asyncio.TimeoutError:
            pass


async def report_progress(task_id: str, progress: float, message: str):
    try:
        await ws_connect(build_ws_url(task_id), {"type": "progress", "progress": progress, "message": message})
    except Exception as e:
        print(f"[WS] 上报进度失败: {e}")


async def report_complete(task_id: str, result: dict):
    try:
        await ws_connect(build_ws_url(task_id), {"type": "complete", "result": result})
    except Exception as e:
        print(f"[WS] 报告完成失败: {e}")


async def report_error(task_id: str, error: str):
    try:
        await ws_connect(build_ws_url(task_id), {"type": "error", "error": error})
    except Exception as e:
        print(f"[WS] 报告错误失败: {e}")


async def execute_rules_route(task_id: str, video_path: str, config: dict, route: str, route_dir: str) -> dict:
    print(f"[{route}] 启动分析: {video_path}")
    await report_progress(task_id, 2, "初始化模型...")

    import cv2

    module = load_route_module(route_dir, "main_rules.py", f"{route}_main_rules_client")
    meta = get_video_meta(video_path)
    total_frames = meta["total_frames"]
    fps = meta["fps"]
    vlm = get_vlm_options(config, route)
    model_path = os.path.join(route_dir, "yolo11n-pose.pt")

    detector_kwargs = {
        "model_path": model_path,
        "fps": fps,
        "vlm_enabled": vlm["vlm_enabled"],
        "vlm_provider": vlm["provider"] if vlm["vlm_enabled"] else "mock",
        "vlm_trigger_level": vlm["trigger"],
        "ollama_host": vlm["ollama_host"],
        "ollama_model": vlm["ollama_model"],
        "siliconflow_model": vlm["siliconflow_model"],
        "vlm_api_key": vlm["api_key"],
        "max_workers": vlm["workers"],
    }

    if route == "route3":
        detector_kwargs.update({
            "cnn_lstm_enabled": False,
            "cnn_lstm_model_path": os.path.join(route_dir, "checkpoints", "best_model.pth"),
        })

    detector = module.MeetingAttentionDetector(**detector_kwargs)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = detector.process_frame(frame)
            detector.submit_audits(results, frame)
            frame_count += 1

            if frame_count % 100 == 0 or frame_count == total_frames:
                progress = min(95.0, frame_count / max(total_frames, 1) * 95.0)
                await report_progress(task_id, progress, f"推理 {frame_count}/{total_frames} 帧")
                print(f"[{route}] 进度 {progress:.1f}%")
    finally:
        cap.release()
        detector.shutdown()

    stats = detector.stats
    level_dist = normalize_level_distribution(stats.level_distribution)
    result = {
        "route": route,
        "total_frames": total_frames,
        "processed_frames": frame_count,
        "total_detections": stats.total_detections,
        "fps": fps,
        "vlm_provider": vlm["provider"] if vlm["vlm_enabled"] else "none",
        "vlm_audits": stats.vlm_audits,
        "level_distribution": level_dist,
        "level_percentages": level_percentages(level_dist),
    }

    if getattr(detector, "audit_system", None):
        result["audit_summary"] = detector.audit_system.get_audit_summary()

    if route == "route3":
        result["cnn_lstm_enabled"] = False

    return result


async def execute_route1(task_id: str, video_path: str, config: dict) -> dict:
    return await execute_rules_route(task_id, video_path, config, "route1", ROUTE1_DIR)


async def execute_route2(task_id: str, video_path: str, config: dict) -> dict:
    print(f"[Route2] 启动真实VLM分析: {video_path}")
    await report_progress(task_id, 5, "初始化路线2 VLM分析...")

    module = load_route_module(ROUTE2_DIR, "main_vlm.py", "route2_main_vlm_client")
    meta = get_video_meta(video_path)
    vlm = get_vlm_options(config, "route2")
    model_path = os.path.join(ROUTE2_DIR, "yolo11n-pose.pt")

    def run_analysis():
        return module.analyze_video(
            video_path=video_path,
            provider=vlm["provider"],
            num_frames=int(config.get("num_frames") or 8),
            max_workers=vlm["workers"],
            model_path=model_path,
            ollama_host=vlm["ollama_host"],
            ollama_model=vlm["ollama_model"],
            siliconflow_model=vlm["siliconflow_model"],
            vlm_api_key=vlm["api_key"],
        )

    await report_progress(task_id, 15, "路线2采样、跟踪、裁剪与VLM分析中...")
    raw = await asyncio.to_thread(run_analysis)
    level_dist = normalize_level_distribution(raw.get("level_distribution", {}))

    result = {
        **raw,
        "route": "route2",
        "total_frames": meta["total_frames"],
        "total_detections": raw.get("num_persons", sum(level_dist.values())),
        "fps": meta["fps"],
        "sampled_frames": raw.get("num_frames"),
        "vlm_provider": vlm["provider"],
        "level_distribution": level_dist,
        "level_percentages": level_percentages(level_dist),
    }
    await report_progress(task_id, 95, "路线2分析完成，正在回传结果...")
    return result


async def execute_route3(task_id: str, video_path: str, config: dict) -> dict:
    return await execute_rules_route(task_id, video_path, config, "route3", ROUTE3_DIR)


async def upload_result(task_id: str, result: dict):
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{SERVER_URL}/api/analysis/tasks/{task_id}/result", json=result)
        resp.raise_for_status()


async def run_task(job: dict):
    task_id = job.get("task_id", "")
    route = job.get("route", "route1")
    video_path = job.get("video_path", "")

    print(f"\n[Client] 收到任务: {task_id} ({route})")
    print(f"[Client] 视频: {video_path}")

    if not os.path.exists(video_path):
        print(f"[Client] 视频不存在: {video_path}")
        await report_error(task_id, f"视频不存在: {video_path}")
        return

    try:
        if route == "route1":
            result = await execute_route1(task_id, video_path, job)
        elif route == "route2":
            result = await execute_route2(task_id, video_path, job)
        elif route == "route3":
            result = await execute_route3(task_id, video_path, job)
        else:
            raise ValueError(f"未知路线: {route}")

        await upload_result(task_id, result)
        await report_complete(task_id, result)
        print(f"[Client] 任务 {task_id} 完成，结果已上传")

    except Exception as e:
        print(f"[Client] 任务 {task_id} 失败: {e}")
        await report_error(task_id, str(e))


async def poll_loop():
    print(f"[Client] 连接服务器: {SERVER_URL}")
    print(f"[Client] WebSocket: {build_ws_url('{task_id}').replace('%7Btask_id%7D', '{task_id}')}")
    print(f"[Client] 轮询间隔: {POLL_INTERVAL}秒")
    print("[Client] 按 Ctrl+C 退出\n")

    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            try:
                resp = await client.get(f"{SERVER_URL}/api/jobs/next")
                resp.raise_for_status()
                data = resp.json()
                job = data.get("job")

                if job:
                    await run_task(job)
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] 暂无任务，等待...")

                await asyncio.sleep(POLL_INTERVAL)

            except httpx.ConnectError:
                print(f"[{time.strftime('%H:%M:%S')}] 无法连接服务器 {SERVER_URL}，重试...")
                await asyncio.sleep(POLL_INTERVAL)
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] 异常: {e}，重试...")
                await asyncio.sleep(POLL_INTERVAL)


def main():
    global SERVER_URL, POLL_INTERVAL

    parser = argparse.ArgumentParser(description="本地Client - 连接服务器执行分析任务")
    parser.add_argument("--server", type=str, default=DEFAULT_SERVER_URL, help="服务器地址，例如 http://localhost:8000")
    parser.add_argument("--poll-interval", type=int, default=POLL_INTERVAL, help="轮询间隔秒数")
    args = parser.parse_args()

    SERVER_URL = normalize_server_url(args.server)
    POLL_INTERVAL = max(1, args.poll_interval)

    try:
        asyncio.run(poll_loop())
    except KeyboardInterrupt:
        print("\n[Client] 退出")


if __name__ == "__main__":
    main()
