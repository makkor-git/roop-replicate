import os
import sys
from typing import List
import platform
import signal
import shutil
import argparse
import onnxruntime
import tensorflow
import roop.globals
import roop.metadata
from roop.predictor import predict_image
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path, has_extension, create_gif, move_temp_gif



def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        return 8
    return 1


def limit_resources() -> None:
    # prevent tensorflow memory leak
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tensorflow.config.experimental.set_virtual_device_configuration(gpu, [
            tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)
        ])
    # limit memory usage
    if roop.globals.max_memory:
        memory = roop.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = roop.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    if not shutil.which('ffmpeg'):
        update_status('ffmpeg is not installed.')
        return False
    return True


def update_status(message: str, scope: str = 'ROOP.CORE') -> None:
    print(f'[{scope}] {message}')


def start() -> None:
    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
        if not frame_processor.pre_start():
            return

    # process image to image
    if has_image_extension(roop.globals.target_path):
        if predict_image(roop.globals.target_path):
            destroy()
        shutil.copy2(roop.globals.target_path, roop.globals.output_path)
        # process frame
        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            update_status('Progressing...', frame_processor.NAME)
            frame_processor.process_image(roop.globals.source_path, roop.globals.output_path, roop.globals.output_path)
            frame_processor.post_process()
        # validate image
        if is_image(roop.globals.target_path):
            update_status('Processing to image succeed!')
        else:
            update_status('Processing to image failed!')
        return

    # process image to videos
    update_status('Creating temporary resources...')
    create_temp(roop.globals.target_path)
    # converting gif to video
    #if has_extension(roop.globals.target_path, ['gif']):
    #    ffmpeg.create_gif_from_video(roop.globals.target_path, roop.globals.output_path)

    # extract frames
    if roop.globals.keep_fps:
        fps = detect_fps(roop.globals.target_path)
        update_status(f'Extracting frames with {fps} FPS...')
        extract_frames(roop.globals.target_path, fps)
    else:
        update_status('Extracting frames with 30000 FPS...')
        extract_frames(roop.globals.target_path)

    # process frame
    temp_frame_paths = get_temp_frame_paths(roop.globals.target_path)
    if temp_frame_paths:
        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            update_status('Progressing...', frame_processor.NAME)
            frame_processor.process_video(roop.globals.source_path, temp_frame_paths)
            frame_processor.post_process()
    else:
        update_status('Frames not found...')
        return

    # create video
    if roop.globals.keep_fps:
        fps = detect_fps(roop.globals.target_path)
        if has_extension(roop.globals.target_path, ['gif']):
            update_status(f'Creating gif with {fps} FPS...')
            create_gif(roop.globals.target_path, fps)
        else:
            update_status(f'Creating video with {fps} FPS...')
            create_video(roop.globals.target_path, fps)
    else:
        if has_extension(roop.globals.target_path, ['gif']):
            update_status('Creating gif with 30 FPS...')
            create_gif(roop.globals.target_path)
        else:
            update_status('Creating video with 30 FPS...')
            create_video(roop.globals.target_path)
        #if has_extension(roop.globals.target_path, ['gif']):
        #    ffmpeg.create_gif_from_video(roop.globals.target_path, roop.globals.output_path)

    if not has_extension(roop.globals.target_path, ['gif']):
        # handle audio
        if roop.globals.skip_audio:
            move_temp(roop.globals.target_path, roop.globals.output_path)
            update_status('Skipping audio...')
        else:
            if roop.globals.keep_fps:
                update_status('Restoring audio...')
            else:
                update_status('Restoring audio might cause issues as fps are not kept...')
            restore_audio(roop.globals.target_path, roop.globals.output_path)
    else:
        move_temp_gif(roop.globals.target_path, roop.globals.output_path);

    # clean temp
    update_status('Cleaning temporary resources...')
    clean_temp(roop.globals.target_path)

    # validate video
    if is_video(roop.globals.target_path):
        update_status('Processing to video succeed!')
        return roop.globals.output_path
    elif has_extension(roop.globals.target_path, ['gif']):
        update_status('Processing to gif succeed!')
        return roop.globals.output_path
    else:
        update_status('Processing to video failed!')


def destroy() -> None:
    if roop.globals.target_path:
        clean_temp(roop.globals.target_path)
    sys.exit()


def run_replicate(source_path, target_path):
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    roop.globals.source_path = source_path
    roop.globals.target_path = target_path
    roop.globals.headless = True
    roop.globals.keep_fps = True
    roop.globals.execution_providers = ['CPUExecutionProvider']
    roop.globals.execution_threads = 2
    roop.globals.reference_face_position = 0
    roop.globals.reference_frame_number = 0
    roop.globals.similar_face_distance = 0.85
    roop.globals.temp_frame_format = 'png'
    roop.globals.temp_frame_quality = 0
    roop.globals.output_video_encoder = 'libx264'
    roop.globals.output_video_quality = 35
    roop.globals.frame_processors = ['face_swapper']

    if not pre_check():
        return
    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
        if not frame_processor.pre_check():
            return

    if roop.globals.headless:
        output = start()
        return output

