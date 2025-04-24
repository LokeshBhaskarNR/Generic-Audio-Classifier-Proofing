import subprocess
import os

def split_video_ffmpeg(input_path, output_dir, segment_duration=5):
    os.makedirs(output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, "segment_%03d.mp4")

    command = [
        "ffmpeg",
        "-i", input_path,
        "-c", "copy", 
        "-map", "0",
        "-segment_time", str(segment_duration),
        "-f", "segment",
        "-reset_timestamps", "1",
        output_template
    ]

    subprocess.run(command)
    print(f"Video split into {segment_duration}-second chunks at: {output_dir}")

split_video_ffmpeg("cars.mp4", "output_segments_ffmpeg")
