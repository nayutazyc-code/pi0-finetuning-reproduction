"""Display-only composition of caller-rendered front, side and wrist RGB frames.

This module never renders cameras, buffers earlier frames or creates a wrist
view from another image. The policy's separate ``cam_high`` image is excluded.
The caller supplies the three views from one control timestamp.
"""
from collections.abc import Mapping
import math
from numbers import Integral, Real

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size):
    for name in ('Menlo.ttc', 'Avenir.ttc', 'DejaVuSansMono.ttf'):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


class MultiViewComposer:
    """Fixed 1280x720 monitoring canvas; inputs and prior outputs stay untouched."""

    width = 1280
    height = 720
    _BACKGROUND = (17, 23, 28)
    _BORDER = (49, 63, 72)
    _MUTED = (141, 157, 167)
    _TEXT = (225, 232, 236)
    _CYAN = (86, 211, 210)
    _INPUT_SHAPES = {'front': (480, 640, 3), 'side': (240, 320, 3), 'wrist': (240, 320, 3)}
    _BOXES = {'front': (0, 0, 960, 720), 'side': (960, 0, 320, 240),
              'wrist': (960, 240, 320, 240), 'status': (960, 480, 320, 240)}

    def __init__(self):
        self._small_font = _font(12)
        self._label_font = _font(14)
        self._value_font = _font(18)
        self._template = Image.new('RGB', (self.width, self.height), self._BACKGROUND)
        draw = ImageDraw.Draw(self._template)
        draw.text((978, 495), 'LIVE / TASK', fill=self._CYAN, font=self._label_font)
        draw.line((978, 520, 1262, 520), fill=self._BORDER)
        self._rows = {'step': 534, 'time': 566, 'grasped': 598, 'inside_box': 630, 'is_success': 662}
        for key, label in (('step', 'TASK STEP'), ('time', 'SIM TIME'), ('grasped', 'GRASP'),
                           ('inside_box', 'IN BOX'), ('is_success', 'SUCCESS')):
            draw.text((978, self._rows[key]+2), label, fill=self._MUTED, font=self._small_font)
        draw.text((978, 701), 'DISPLAY ONLY', fill=self._MUTED, font=self._small_font)
        # Small static patches avoid a full-canvas alpha overlay each frame.
        self._patches = []
        for x, y, width, height in self._BOXES.values():
            self._patches.extend([
                (Image.new('RGB', (width, 1), self._BORDER), (x, y)),
                (Image.new('RGB', (width, 1), self._BORDER), (x, y+height-1)),
                (Image.new('RGB', (1, height), self._BORDER), (x, y)),
                (Image.new('RGB', (1, height), self._BORDER), (x+width-1, y))])
        for name in ('front', 'side', 'wrist'):
            x, y, _, _ = self._BOXES[name]
            self._patches.append((self._text_patch(name.upper(), self._CYAN, self._label_font), (x+12, y+12)))
        self._flag_patches = {
            True: self._text_patch('YES', self._CYAN, self._value_font, padding=0),
            False: self._text_patch('NO', self._TEXT, self._value_font, padding=0),
            None: self._text_patch('--', self._MUTED, self._value_font, padding=0)}
        self._status_template = self._template.crop((960, 480, 1280, 720))
        self._patches = [(np.array(patch, copy=True), position) for patch, position in self._patches]

    def _text_patch(self, text, color, font, padding=6):
        bounds = font.getbbox(text)
        patch = Image.new('RGB', (bounds[2]-bounds[0]+2*padding, bounds[3]-bounds[1]+2*padding), self._BACKGROUND)
        ImageDraw.Draw(patch).text((padding-bounds[0], padding-bounds[1]), text, fill=color, font=font)
        return patch

    @classmethod
    def describe(cls):
        """JSON-safe layout/provenance contract, without claiming camera capture."""
        return {'width': cls.width, 'height': cls.height, 'dtype': 'uint8', 'color_space': 'RGB',
                'display_only': True, 'policy_frame_included': False, 'excluded_policy_camera': 'cam_high',
                'input_shapes': {name: list(shape) for name, shape in cls._INPUT_SHAPES.items()},
                'panels_xywh': {name: list(box) for name, box in cls._BOXES.items()},
                'front_resize': '640x480 to 960x720, nearest-neighbor; repeated pixels may look blocky. Side and wrist keep original pixels.',
                'frame_source': 'Three caller-rendered camera frames at the same control timestamp; no synthesis or temporal buffering.',
                'status': {'step': 'Caller-provided control step',
                           'simulation_time_seconds': 'info value, otherwise step * 0.02',
                           'grasped': 'Current lifted pinch, not ever_grasped',
                           'inside_box': 'Current whole-object containment', 'is_success': 'Current simulator success'}}

    @classmethod
    def _image(cls, name, value):
        # uint8 is intrinsically finite; no extra full-frame finite scan needed.
        if not isinstance(value, np.ndarray) or value.dtype != np.uint8 or value.shape != cls._INPUT_SHAPES[name]:
            raise ValueError(f'{name} must be a uint8 RGB array of shape {cls._INPUT_SHAPES[name]}')
        return value

    @staticmethod
    def _flag(info, name):
        value = info.get(name)
        if value is not None and not isinstance(value, (bool, np.bool_)):
            raise ValueError(f'{name} must be a boolean when supplied')
        return bool(value) if value is not None else None

    def compose(self, front, side, wrist, *, step, info):
        """Compose one timestamp. Missing status flags display ``--``, never NO."""
        if isinstance(step, (bool, np.bool_)) or not isinstance(step, Integral) or step < 0:
            raise ValueError('step must be a non-negative integer control step')
        if not isinstance(info, Mapping):
            raise ValueError('info must be a mapping of simulator status fields')
        seconds = info.get('simulation_time_seconds', int(step)*.02)
        if isinstance(seconds, (bool, np.bool_)) or not isinstance(seconds, Real) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError('simulation_time_seconds must be finite and non-negative')
        flags = {name: self._flag(info, name) for name in ('grasped', 'inside_box', 'is_success')}
        images = {name: self._image(name, value) for name, value in [('front', front), ('side', side), ('wrist', wrist)]}
        # Compose camera pixels directly into the owned output array, keeping
        # Pillow's dynamic drawing confined to the small status panel.
        canvas = np.empty((self.height, self.width, 3), dtype=np.uint8)
        canvas[:, :960] = np.asarray(Image.fromarray(images['front']).resize((960, 720), Image.Resampling.NEAREST))
        canvas[:240, 960:] = images['side']
        canvas[240:480, 960:] = images['wrist']
        status = self._status_template.copy()
        draw = ImageDraw.Draw(status)
        text_step = str(step)
        if len(text_step) > 12:
            text_step = text_step[:9]+'...'
        text_time = f'{seconds:.2f}s' if seconds < 100000 else f'{seconds:.2e}s'
        for name, text in (('step', text_step), ('time', text_time)):
            bounds = self._value_font.getbbox(text)
            draw.text((302-(bounds[2]-bounds[0]), self._rows[name]-480-bounds[1]),
                      text, fill=self._TEXT, font=self._value_font)
        for name, value in flags.items():
            patch = self._flag_patches[value]
            status.paste(patch, (302-patch.width, self._rows[name]-480))
        canvas[480:, 960:] = np.asarray(status)
        for patch, (x, y) in self._patches:
            height, width = patch.shape[:2]
            canvas[y:y+height, x:x+width] = patch
        return canvas
