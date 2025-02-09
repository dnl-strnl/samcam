from collections import deque
import cv2
import glob
import logging as log
import numpy as np
import os
import requests
from threading import Lock

class StreamBuffer:
    def __init__(
            self,
            video_source=1,
            buffer_size=1,
            frame_width=640,
            frame_height=480,
            frame_buffer=os.getcwd(),
        ):
        self.video_source = video_source
        self.buffer_size = buffer_size
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.temp_dir = frame_buffer
        self.lock = Lock()
        self.frame_buffer = []
        self.buffer_ready = False
        self.show_overlay = False
        self.frame_counter = 0
        self.current_frame = None
        self.next_mask = None

        if isinstance(video_source, str) and video_source.startswith("http://"):
            self.stream = self._ip_camera_generator(video_source)
        else:
            self.stream = cv2.VideoCapture(video_source)
            if not self.stream.isOpened():
                raise ValueError(f"failed to open: {video_source}")
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        self.clean_temp_dir()

    def _ip_camera_generator(self, url):
        response = requests.get(url, stream=True)
        bytes_data = bytes()
        for chunk in response.iter_content(chunk_size=1024):
            bytes_data += chunk
            a = bytes_data.find(b'\xff\xd8')
            b = bytes_data.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                frame = cv2.imdecode(
                    p.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                yield frame

    def get_resolution(self):
        return dict(width=self.frame_width, height=self.frame_height)

    def clean_temp_dir(self):
        if os.path.exists(self.temp_dir):
            files = glob.glob(f"{self.temp_dir}/*.jpg")
            for remove_file in files:
                os.remove(remove_file)
                log.debug(f"{remove_file=}")
        else:
            os.makedirs(self.temp_dir)
            log.debug(f"make: {self.temp_dir}")

    def manage_buffer(self, frame):
        frame_idx = self.frame_counter % self.buffer_size
        frame_path = os.path.join(self.temp_dir, f"{frame_idx:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        log.debug(f"wrote {self.frame_counter} to {frame_path}")
        if len(self.frame_buffer) < self.buffer_size:
            self.frame_buffer.append(frame)
            buffer_size = len(self.frame_buffer)
            log.debug(f"frame {len(self.frame_buffer):05d} added. {buffer_size=}")
        else:
            self.frame_buffer[frame_idx] = frame
            log.debug(f"wrote {frame_idx=}.")
        self.frame_counter += 1
        files = glob.glob(f"{self.temp_dir}/*.jpg")
        log.debug(f"buffer: {len(files)}/{self.buffer_size}")
        self.buffer_ready = len(self.frame_buffer) == self.buffer_size

    def get_frame(self):
        with self.lock:
            if isinstance(self.video_source, str) and self.video_source.startswith("http://"):
                try:
                    frame = next(self.stream)
                    self.current_frame = frame
                    self.manage_buffer(frame)
                    return frame
                except StopIteration:
                    return None
            else:
                if not self.stream.isOpened():
                    return None
                success, frame = self.stream.read()
                if not success:
                    return None
                self.current_frame = frame
                self.manage_buffer(frame)
                return frame

    def __del__(self):
        if not isinstance(self.video_source, str) or not self.video_source.startswith("http://"):
            self.stream.release()
        self.clean_temp_dir()

    def get_current_frame(self):
        with self.lock:
            return (self.current_frame.copy(), self.frame_counter) \
                if self.current_frame is not None \
                else (None, self.frame_counter)
