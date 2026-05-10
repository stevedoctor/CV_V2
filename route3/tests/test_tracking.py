"""
测试ByteTrack跟踪ID一致性

验证内容：
1. 跟踪ID是否跨帧稳定
2. 人员遮挡/离开后ID是否恢复
3. 轨迹是否连续
"""
import cv2
import sys
import os
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trackers import ByteTrackManager


def test_tracking_consistency(video_path: str, 
                               max_frames: int = 300,
                               show_window: bool = True):
    """
    测试跟踪ID一致性
    
    Args:
        video_path: 视频路径
        max_frames: 最大测试帧数
        show_window: 是否显示窗口
    """
    print(f"[INFO] 测试视频: {video_path}")
    
    tracker = ByteTrackManager(model_path="yolo11n-pose.pt", verbose=False)
    
    # 统计数据
    id_history = defaultdict(list)  # {track_id: [frame_indices]}
    id_transitions = []             # ID切换记录
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0
    
    print(f"[INFO] FPS: {fps}")
    print("[INFO] 开始跟踪测试...")
    
    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 跟踪
        tracks = tracker.track_frame(frame)
        
        # 记录ID出现
        current_ids = set()
        for track_id, keypoints in tracks:
            current_ids.add(track_id)
            id_history[track_id].append(frame_idx)
        
        # 更新帧计数
        frame_idx += 1
        
        # 可视化（可选）
        if show_window:
            for track_id, kps in tracks:
                # 绘制鼻子位置（用于显示ID）
                nose = kps[0]
                if nose[0] > 0:
                    pos = (int(nose[0]), int(nose[1]))
                    # 使用颜色区分不同ID
                    color = ((track_id * 50) % 255, (track_id * 80) % 255, (track_id * 120) % 255)
                    cv2.putText(frame, f"ID:{track_id}", 
                               (pos[0] - 30, pos[1] - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.circle(frame, pos, 5, color, -1)
            
            # 显示帧号
            cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Tracks: {len(tracks)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("ByteTrack Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # 分析结果
    print("\n" + "=" * 60)
    print("📊 跟踪ID一致性报告")
    print("=" * 60)
    
    print(f"\n总帧数: {frame_idx}")
    print(f"检测到的唯一ID数: {len(id_history)}")
    
    print("\n📋 各ID轨迹详情:")
    print("-" * 60)
    
    for tid, frames in sorted(id_history.items()):
        appearance_rate = len(frames) / frame_idx * 100
        frame_range = f"[{min(frames)}, {max(frames)}]"
        gaps = _calculate_gaps(frames)
        
        status = "✅" if len(gaps) == 0 else "⚠️"
        
        print(f"ID {tid:3d}: 出现{len(frames):4d}帧 ({appearance_rate:5.1f}%) | "
              f"范围{frame_range:20s} | 间隙数:{len(gaps):2d} {status}")
        
        if gaps and len(gaps) <= 3:
            print(f"        间隙详情: {gaps}")
    
    # 统计摘要
    print("\n" + "-" * 60)
    print("📈 稳定性统计:")
    
    stable_ids = [tid for tid, frames in id_history.items() 
                  if len(_calculate_gaps(frames)) == 0]
    unstable_ids = [tid for tid in id_history.keys() if tid not in stable_ids]
    
    print(f"  ✅ 稳定跟踪ID: {len(stable_ids)} 个")
    print(f"  ⚠️ 有间隙ID: {len(unstable_ids)} 个")
    
    if unstable_ids:
        print(f"     {unstable_ids}")
    
    print("=" * 60)
    
    return id_history


def _calculate_gaps(frames: list) -> list:
    """计算轨迹间隙"""
    if len(frames) < 2:
        return []
    
    gaps = []
    sorted_frames = sorted(frames)
    
    for i in range(1, len(sorted_frames)):
        gap = sorted_frames[i] - sorted_frames[i-1]
        if gap > 1:  # 超过1帧的间隔视为间隙
            gaps.append((sorted_frames[i-1], sorted_frames[i], gap))
    
    return gaps


def test_track_video_generator(video_path: str, max_frames: int = 100):
    """
    测试视频流生成器
    """
    print("[INFO] 测试视频流生成器模式...")
    
    tracker = ByteTrackManager()
    frame_count = 0
    
    for frame, tracks in tracker.track_video(video_path):
        frame_count += 1
        print(f"Frame {frame_count}: {len(tracks)} tracks", end='\r')
        
        if frame_count >= max_frames:
            break
    
    print(f"\n[INFO] 完成 {frame_count} 帧测试")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试ByteTrack跟踪")
    parser.add_argument("--video", type=str, 
                       default="data/videos/meeting_attention_video.mp4",
                       help="视频路径")
    parser.add_argument("--frames", type=int, default=300,
                       help="最大测试帧数")
    parser.add_argument("--no-display", action="store_true",
                       help="不显示窗口")
    
    args = parser.parse_args()
    
    # 运行测试
    test_tracking_consistency(
        args.video, 
        max_frames=args.frames,
        show_window=not args.no_display
    )
