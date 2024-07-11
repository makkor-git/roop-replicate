from typing import List, Optional

source_path: Optional[str] = None
target_path: Optional[str] = None
output_path: Optional[str] = "output.gif"
headless: Optional[bool] = None
frame_processors: List[str] = []
keep_fps: Optional[bool] = True
keep_frames: Optional[bool] = None
skip_audio: Optional[bool] = None
many_faces: Optional[bool] = True
reference_face_position: Optional[int] = 0
reference_frame_number: Optional[int] = 0
similar_face_distance: Optional[float] = 0.85
temp_frame_format: Optional[str] = 'png'
temp_frame_quality: Optional[int] = 0
output_video_encoder: Optional[str] = 'libx264'
output_video_quality: Optional[int] = 35
max_memory: Optional[int] = 8
execution_providers: List[str] = ['DmlExecutionProvider']
execution_threads: Optional[int] = 1
log_level: str = 'error'
