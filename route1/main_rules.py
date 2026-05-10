"""
路线1主入口：ByteTrack + RFAC + 规则引擎 + VLM审计（多线程并发）

完整流程：
1. ByteTrack跟踪 → 获取跨帧一致的track_id
2. 姿态估计 → 提取关键点
3. RFAC计算 → 四维度指标
4. 规则引擎 → 异常分级判定
5. VLM审计 → 存疑状态复核（多线程并发，可选）

使用方式：
    # 仅规则引擎
    python main_rules.py --video data/videos/meeting.mp4
    
    # 启用VLM审计（本地Ollama，4线程并发）
    python main_rules.py --video data/videos/meeting.mp4 --vlm --workers 4
    
    # 启用VLM审计（硅基流动云端，4线程并发）
    python main_rules.py --video data/videos/meeting.mp4 --vlm --vlm-provider siliconflow --workers 4
"""
import cv2
import sys
import os
import argparse
from dataclasses import dataclass
import numpy as np
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import threading

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.trackers import ByteTrackManager
from src.processors import PersonTracker, RFACCalculator
from src.rules import RuleEngine, RuleResult
from src.models import AttentionScorer
from src.utils.display_utils import DisplayHelper
from src.utils.video_utils import VideoHelper
from src.config import Settings, constants as const
from src.audit.audit_system import create_audit_system_v2


@dataclass
class ProcessingStats:
    """处理统计"""
    total_frames: int = 0
    total_detections: int = 0
    level_distribution: Dict[int, int] = None
    vlm_audits: int = 0
    
    def __post_init__(self):
        if self.level_distribution is None:
            self.level_distribution = {0: 0, 1: 0, 2: 0, 3: 0}
    
    def update(self, results: List[RuleResult]):
        self.total_frames += 1
        for r in results:
            self.total_detections += 1
            self.level_distribution[r.overall_level] += 1
    
    def increment_vlm_audits(self):
        self.vlm_audits += 1
    
    def print_report(self):
        print("\n" + "=" * 60)
        print("📊 规则引擎检测报告")
        print("=" * 60)
        print(f"总检测人次：{self.total_detections}")
        
        level_names = {0: "正常", 1: "轻微", 2: "中度", 3: "重度"}
        
        print("\n异常等级分布:")
        for level, count in sorted(self.level_distribution.items()):
            pct = count / self.total_detections * 100 if self.total_detections > 0 else 0
            print(f"  {level_names[level]}: {count} ({pct:.1f}%)")
        
        if self.vlm_audits > 0:
            print(f"\n🤖 VLM审计次数: {self.vlm_audits}")
        
        print("=" * 60)


class MeetingAttentionDetector:
    """会议注意力检测器（支持VLM审计多线程并发）"""
    
    def __init__(self, 
                 model_path: str = "yolo11n-pose.pt",
                 history_size: int = 10,
                 fps: float = 20.0,
                 vlm_enabled: bool = False,
                 vlm_provider: str = "ollama",
                 vlm_trigger_level: str = "MODERATE",
                 ollama_host: str = None,
                 ollama_model: str = None,
                 siliconflow_model: str = None,
                 vlm_api_key: str = None,
                 max_workers: int = 4):
        """
        初始化检测器
        
        Args:
            model_path: YOLO模型路径
            history_size: 历史帧数
            fps: 视频帧率
            vlm_enabled: 是否启用VLM审计
            vlm_provider: VLM提供者 (mock, ollama, siliconflow)
            vlm_trigger_level: 触发审计的等级
            ollama_host: Ollama服务地址
            ollama_model: Ollama模型名称
            siliconflow_model: 硅基流动模型名称
            vlm_api_key: VLM API Key
            max_workers: VLM审计最大并发线程数
        """
        self.tracker = ByteTrackManager(model_path=model_path)
        self.person_tracker = PersonTracker(history_size=history_size)
        self.rfac_calculator = RFACCalculator()
        self.rule_engine = RuleEngine()
        self.attention_scorer = AttentionScorer()
        
        self.dt = 1.0 / fps
        self.stats = ProcessingStats()
        
        # VLM审计系统
        self.vlm_enabled = vlm_enabled
        self.audit_system = None
        self.max_workers = max_workers
        self._executor = None
        self._lock = threading.Lock()
        self._pending_futures = []
        
        if vlm_enabled:
            self.audit_system = create_audit_system_v2(
                provider=vlm_provider,
                trigger_level=vlm_trigger_level,
                ollama_host=ollama_host,
                ollama_model=ollama_model,
                siliconflow_model=siliconflow_model,
                vlm_api_key=vlm_api_key
            )
            self._executor = ThreadPoolExecutor(max_workers=max_workers)
            print(f"[INFO] VLM审计并发线程: {max_workers}")
    
    def process_frame(self, frame) -> List[tuple]:
        """处理单帧（串行：track→score→RFAC→rule）"""
        results = []
        
        # 1. ByteTrack跟踪
        tracks = self.tracker.track_frame(frame)
        
        for track_id, keypoints in tracks:
            # 2. 计算当前评分
            current_score = self.attention_scorer.calculate_current_score(keypoints)
            
            # 3. 更新人员状态
            self.person_tracker.update_person(
                track_id, current_score, keypoints, self.dt
            )
            
            # 4. 获取人员状态
            person_state = self.person_tracker.get_or_create_person(track_id)
            
            # 5. RFAC计算
            rfac_score = self.rfac_calculator.calculate(person_state, self.dt)
            
            # 6. 规则引擎判定
            rule_result = self.rule_engine.evaluate_all(rfac_score)
            
            # 7. VLM审计帧缓冲（必须同步，给后续异步审计提供帧数据）
            if self.vlm_enabled and self.audit_system:
                self.audit_system.add_frame(frame, track_id)
            
            results.append((track_id, rule_result, rfac_score, keypoints))
        
        self.stats.update([r[1] for r in results])
        return results
    
    def submit_audits(self, results: List[tuple], frame: np.ndarray):
        """
        异步提交VLM审计任务（非阻塞）
        
        对所有达到触发阈值的人员，并发提交审计到线程池
        
        Args:
            results: process_frame()的返回结果
            frame: 当前帧（会被copy，避免主线程修改）
        """
        if not self.vlm_enabled or not self.audit_system or not self._executor:
            return
        
        frame_copy = frame.copy()
        
        for track_id, rule_result, rfac_score, keypoints in results:
            level_names = ["NORMAL", "MILD", "MODERATE", "SEVERE"]
            level = level_names[rule_result.overall_level]
            
            if self.audit_system.should_trigger_audit(
                rule_result.overall_value, level
            ):
                future = self._executor.submit(
                    self._audit_worker, track_id, rule_result.overall_value, level, frame_copy
                )
                with self._lock:
                    self._pending_futures.append(future)
    
    def _audit_worker(self, track_id: int, score: float, level: str, frame: np.ndarray):
        """
        VLM审计工作线程（在线程池中执行）
        
        Args:
            track_id: 人员ID
            score: 规则引擎评分
            level: 触发等级
            frame: 帧副本
        """
        try:
            self.audit_system.trigger_audit(
                person_id=track_id,
                score=score,
                level=level,
                frame=frame
            )
            self.stats.increment_vlm_audits()
        except Exception as e:
            print(f"❌ [VLM审计线程] P{track_id} 异常: {e}")
    
    def wait_audits_complete(self):
        """等待所有已提交的审计任务完成"""
        if not self._pending_futures:
            return
        
        with self._lock:
            futures = list(self._pending_futures)
            self._pending_futures.clear()
        
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"❌ [VLM审计] 任务异常: {e}")
    
    def shutdown(self):
        """关闭线程池，等待所有审计完成"""
        self.wait_audits_complete()
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
    
    def render_frame(self, frame, results: List[tuple]) -> np.ndarray:
        """渲染帧"""
        for track_id, rule_result, rfac_score, keypoints in results:
            level_colors = {
                0: (0, 255, 0),
                1: (0, 255, 255),
                2: (0, 165, 255),
                3: (0, 0, 255)
            }
            color = level_colors[rule_result.overall_level]
            
            for kp in keypoints:
                if kp[0] > 0 and kp[1] > 0:
                    cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, color, -1)
            
            nose = keypoints[0]
            if nose[0] > 0:
                level_names = ["正常", "轻微", "中度", "重度"]
                text = f"ID:{track_id} L{rule_result.overall_level}({level_names[rule_result.overall_level]})"
                cv2.putText(frame, text,
                           (int(nose[0]) - 60, int(nose[1]) - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                detail_y = int(nose[1]) - 40
                details = f"R{rule_result.rushing_level} F{rule_result.fatigue_level} A{rule_result.apathy_level}"
                cv2.putText(frame, details,
                           (int(nose[0]) - 40, detail_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.putText(frame, f"Frame: {self.stats.total_frames}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if self.vlm_enabled:
            cv2.putText(frame, f"VLM: ON ({self.stats.vlm_audits} audits, {self.max_workers} threads)",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame
    
    def print_audit_summary(self):
        """打印VLM审计汇总"""
        if self.vlm_enabled and self.audit_system:
            self.audit_system.print_summary()


def main(video_path: str, 
         model_path: str = "yolo11n-pose.pt",
         show_window: bool = True,
         save_output: bool = False,
         output_path: str = None,
         max_frames: int = None,
         vlm_enabled: bool = False,
         vlm_provider: str = "ollama",
         vlm_trigger_level: str = "MODERATE",
         ollama_host: str = None,
         ollama_model: str = None,
         siliconflow_model: str = None,
         vlm_api_key: str = None,
         max_workers: int = 4):
    """主程序"""
    print(f"[INFO] 初始化检测器...")
    print(f"[INFO] 模型: {model_path}")
    print(f"[INFO] 视频: {video_path}")
    print(f"[INFO] VLM审计: {'启用' if vlm_enabled else '禁用'}")
    if vlm_enabled:
        print(f"[INFO] VLM提供者: {vlm_provider}")
        print(f"[INFO] 触发等级: {vlm_trigger_level}")
        print(f"[INFO] 并发线程: {max_workers}")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[INFO] FPS: {fps}, 分辨率: {width}x{height}")
    
    detector = MeetingAttentionDetector(
        model_path=model_path,
        fps=fps,
        vlm_enabled=vlm_enabled,
        vlm_provider=vlm_provider,
        vlm_trigger_level=vlm_trigger_level,
        ollama_host=ollama_host,
        ollama_model=ollama_model,
        siliconflow_model=siliconflow_model,
        vlm_api_key=vlm_api_key,
        max_workers=max_workers
    )
    
    writer = None
    if save_output:
        if output_path is None:
            output_path = "output_rules.mp4"
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"[INFO] 输出将保存到: {output_path}")
    
    print("[INFO] 开始处理...")
    frame_count = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and frame_count >= max_frames:
                break
            
            # 串行：track → score → RFAC → rule
            results = detector.process_frame(frame)
            
            # 异步：VLM审计（非阻塞，提交到线程池）
            detector.submit_audits(results, frame)
            
            if show_window or save_output:
                frame = detector.render_frame(frame, results)
                
                if show_window:
                    cv2.imshow("Route1: Rules Engine + VLM", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("[INFO] 用户中断")
                        break
                
                if writer:
                    writer.write(frame)
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"[INFO] 已处理 {frame_count} 帧...")
    finally:
        # 确保所有审计完成 + 线程池关闭
        print("[INFO] 等待VLM审计完成...")
        detector.shutdown()
    
    cap.release()
    if writer:
        writer.release()
    if show_window:
        cv2.destroyAllWindows()
    
    detector.stats.print_report()
    detector.print_audit_summary()
    
    print(f"[INFO] 处理完成，共 {frame_count} 帧")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="会议注意力检测 - 规则引擎 + VLM审计（多线程并发）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 仅规则引擎
  python main_rules.py --video meeting.mp4
  
  # 本地Ollama (qwen3-vl:8b, 4线程并发)
  python main_rules.py --video meeting.mp4 --vlm --workers 4
  
  # 硅基流动云端 (Qwen3-VL-32B, 4线程并发)
  export SILICONFLOW_API_KEY=sk-xxx
  python main_rules.py --video meeting.mp4 --vlm --vlm-provider siliconflow --workers 4
"""
    )
    
    # 基本参数
    parser.add_argument("--video", type=str, 
                       default="data/videos/meeting_attention_video.mp4",
                       help="视频路径")
    parser.add_argument("--model", type=str,
                       default="yolo11n-pose.pt",
                       help="YOLO模型路径")
    parser.add_argument("--no-display", action="store_true",
                       help="不显示窗口")
    parser.add_argument("--save", action="store_true",
                       help="保存输出视频")
    parser.add_argument("--output", type=str,
                       help="输出视频路径")
    parser.add_argument("--frames", type=int,
                       help="最大处理帧数")
    
    # VLM审计参数
    parser.add_argument("--vlm", action="store_true",
                       help="启用VLM审计（存疑状态复核）")
    parser.add_argument("--vlm-provider", type=str, default="ollama",
                        choices=["mock", "ollama", "siliconflow"],
                       help="VLM提供者: mock(测试), ollama(本地), siliconflow(硅基流动)")
    parser.add_argument("--vlm-trigger", type=str, default="MODERATE",
                       choices=["MILD", "MODERATE", "SEVERE"],
                       help="触发VLM审计的等级")
    parser.add_argument("--vlm-api-key", type=str,
                       help="VLM API Key (也可通过环境变量设置)")
    parser.add_argument("--workers", type=int, default=4,
                       help="VLM审计并发线程数 (默认4)")
    
    # Ollama参数
    parser.add_argument("--ollama-host", type=str,
                       default="http://localhost:11434",
                       help="Ollama服务地址")
    parser.add_argument("--ollama-model", type=str, default="qwen3-vl:8b",
                       help="Ollama模型名称")
    
    # 硅基流动参数
    parser.add_argument("--siliconflow-model", type=str,
                       default="Qwen/Qwen3-VL-32B-Instruct",
                       help="硅基流动模型名称")
    
    args = parser.parse_args()
    
    # 处理API Key（命令行 > 环境变量）
    vlm_api_key = args.vlm_api_key
    if not vlm_api_key:
        if args.vlm_provider == "siliconflow":
            vlm_api_key = os.environ.get("SILICONFLOW_API_KEY", "")
    
    main(
        video_path=args.video,
        model_path=args.model,
        show_window=not args.no_display,
        save_output=args.save,
        output_path=args.output,
        max_frames=args.frames,
        vlm_enabled=args.vlm,
        vlm_provider=args.vlm_provider,
        vlm_trigger_level=args.vlm_trigger,
        ollama_host=args.ollama_host,
        ollama_model=args.ollama_model,
        siliconflow_model=args.siliconflow_model,
        vlm_api_key=vlm_api_key,
        max_workers=args.workers
    )
