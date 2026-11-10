"""Lightweight main-thread RGB display using the installed MuJoCo GLFW stack."""
import time

import glfw
import numpy as np
from OpenGL import GL

from .execution import WindowClosed


class GlfwViewer:
    def __init__(self, width=640, height=480):
        if not glfw.init():
            raise RuntimeError("GLFW 初始化失败；可使用 display_backend='matplotlib'。")
        previous = glfw.get_current_context()
        glfw.default_window_hints()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        self.window = glfw.create_window(width, height, "Original pi0 simulation", None, None)
        if self.window is None:
            raise RuntimeError("无法创建 GLFW 显示窗口。")
        self.frames = 0
        self.draw_seconds = 0.0
        self.last_title_update = float("-inf")
        self.title_updates = 0
        self.last_stage_timings = {}
        self.width, self.height = width, height
        try:
            glfw.make_context_current(self.window)
            # Control pacing is handled by the simulation clock, not swap/vsync.
            glfw.swap_interval(0)
            self.texture = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, width, height, 0,
                            GL.GL_RGB, GL.GL_UNSIGNED_BYTE, None)
        except BaseException:
            glfw.destroy_window(self.window)
            self.window = None
            raise
        finally:
            glfw.make_context_current(previous)

    def poll(self):
        glfw.poll_events()
        if self.window is None or glfw.window_should_close(self.window):
            raise WindowClosed()
        if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(self.window, True)
            raise WindowClosed()

    def draw(self, image, title):
        self.poll()
        started = time.perf_counter()
        image = np.ascontiguousarray(image, dtype=np.uint8)
        if image.shape != (self.height, self.width, 3):
            raise ValueError("GLFW 显示帧尺寸与相机不匹配。")
        previous = glfw.get_current_context()
        stages = {}
        try:
            stage_start = time.perf_counter()
            glfw.make_context_current(self.window)
            stages["context_seconds"] = time.perf_counter() - stage_start
            width, height = glfw.get_framebuffer_size(self.window)
            if not width or not height:
                return
            scale = min(width / self.width, height / self.height)
            vw, vh = int(self.width * scale), int(self.height * scale)
            GL.glViewport(0, 0, width, height)
            GL.glClearColor(0, 0, 0, 1)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            GL.glViewport((width - vw) // 2, (height - vh) // 2, vw, vh)
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glLoadIdentity()
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glLoadIdentity()
            GL.glEnable(GL.GL_TEXTURE_2D)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
            stage_start = time.perf_counter()
            GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, self.width, self.height,
                               GL.GL_RGB, GL.GL_UNSIGNED_BYTE, image)
            stages["texture_upload_seconds"] = time.perf_counter() - stage_start
            GL.glColor3f(1, 1, 1)
            GL.glBegin(GL.GL_QUADS)
            for uv, xy in [((0, 1), (-1, -1)), ((1, 1), (1, -1)),
                           ((1, 0), (1, 1)), ((0, 0), (-1, 1))]:
                GL.glTexCoord2f(*uv)
                GL.glVertex2f(*xy)
            GL.glEnd()
            GL.glDisable(GL.GL_TEXTURE_2D)
            # Titles are status text, not animation. Avoid sending 50 window-
            # server title changes per second alongside the camera frames.
            now = time.perf_counter()
            stage_start = now
            if now - self.last_title_update >= 0.25:
                glfw.set_window_title(self.window, title)
                self.last_title_update = now
                self.title_updates += 1
            stages["title_seconds"] = time.perf_counter() - stage_start
            stage_start = time.perf_counter()
            glfw.swap_buffers(self.window)
            stages["swap_seconds"] = time.perf_counter() - stage_start
            self.frames += 1
        finally:
            glfw.make_context_current(previous)
            self.last_stage_timings = stages
            self.draw_seconds += time.perf_counter() - started

    def close(self):
        if self.window is None:
            return
        previous = glfw.get_current_context()
        try:
            glfw.make_context_current(self.window)
            GL.glDeleteTextures([self.texture])
        finally:
            glfw.make_context_current(None)
            glfw.destroy_window(self.window)
            self.window = None
            if previous:
                glfw.make_context_current(previous)
        # Never call glfw.terminate(): dm_control owns other live GLFW contexts.
