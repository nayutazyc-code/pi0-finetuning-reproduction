"""Local scene construction and preview; no policy connection or task controller."""

from dataclasses import asdict, dataclass
from datetime import datetime
import logging
import math
from pathlib import Path
import time
import tomllib
import uuid
import xml.etree.ElementTree as ET

from .config import PROJECT_ROOT
from .random_pose import validate_pose_randomization


@dataclass
class SceneConfig:
    banana_position: tuple[float, float, float] = (0.18, 0.48, 0.045)
    banana_yaw_degrees: float = 10.0
    banana_mass_kg: float = 0.12
    banana_friction: tuple[float, float, float] = (0.8, 0.005, 0.0001)
    box_position: tuple[float, float, float] = (0.13, 0.72, 0.0)
    camera: str = "top"
    settle_seconds: float = 0.6
    controller_profile: str = "original"
    table_rgba: tuple[float, float, float, float] | None = None
    light_scale: float = 1.0
    wrist_damping_scale: float = 1.0
    right_wrist_angle_kp_scale: float = 1.0
    randomize_banana_pose: bool = False
    banana_x_range: tuple[float, float] = (0.16, 0.20)
    banana_y_range: tuple[float, float] = (0.46, 0.50)
    banana_yaw_range_degrees: tuple[float, float] = (0.0, 20.0)


def _validate_wrist_damping(settings: SceneConfig) -> None:
    value = settings.wrist_damping_scale
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("wrist_damping_scale must be a finite positive number")
    if value != 1.0 and settings.controller_profile != "position_pd":
        raise ValueError("Non-default wrist_damping_scale requires controller_profile=position_pd")
    if not math.isfinite(2 * value):
        raise ValueError("wrist_damping_scale produces non-finite joint damping")


def _validate_right_wrist_kp(settings: SceneConfig) -> None:
    value = settings.right_wrist_angle_kp_scale
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("right_wrist_angle_kp_scale must be a finite positive number")
    if value != 1.0 and settings.controller_profile != "position_pd":
        raise ValueError("Non-default right_wrist_angle_kp_scale requires controller_profile=position_pd")
    if not math.isfinite(200 * value):
        raise ValueError("right_wrist_angle_kp_scale produces non-finite actuator gain")


def load_scene_config(path: Path) -> SceneConfig:
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    if set(data) != {"scene"}:
        raise ValueError("场景预览配置只接受 [scene]，请使用 configs/banana_box.toml。")
    settings = SceneConfig(**data["scene"])
    for name, bounds in validate_pose_randomization(settings).items():
        setattr(settings, name, bounds)
    for name in ("banana_position", "box_position", "banana_friction"):
        values = getattr(settings, name)
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError(f"{name} 必须包含 3 个数。")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in values):
            raise ValueError(f"{name} 必须是有限数值。")
        setattr(settings, name, tuple(float(value) for value in values))
    for name in ("banana_yaw_degrees", "banana_mass_kg", "settle_seconds"):
        value = getattr(settings, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} 必须是有限数值。")
    if settings.banana_mass_kg <= 0 or min(settings.banana_friction) < 0:
        raise ValueError("香蕉质量必须大于 0，摩擦系数不能小于 0。")
    if min(settings.banana_position[2], settings.box_position[2]) < 0:
        raise ValueError("物体初始 z 坐标不能小于桌面高度 0。")
    if settings.controller_profile not in {"original", "position_pd"}:
        raise ValueError("controller_profile 只能为 original 或 position_pd。")
    _validate_wrist_damping(settings)
    _validate_right_wrist_kp(settings)
    if settings.camera not in {"top", "angle"}:
        raise ValueError("camera 只能为 top 或 angle。")
    if not 0 <= settings.settle_seconds <= 10:
        raise ValueError("settle_seconds 必须在 0–10 秒之间。")
    if settings.table_rgba is not None:
        if len(settings.table_rgba) != 4 or any(not isinstance(v, (int,float)) or not 0 <= v <= 1 for v in settings.table_rgba):
            raise ValueError("table_rgba must have four components in [0,1]")
        settings.table_rgba = tuple(settings.table_rgba)
    if not 0.5 <= settings.light_scale <= 1.5:
        raise ValueError("light_scale must be in [0.5,1.5]")
    return settings


def _numbers(values) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _read_with_includes(path: Path, ancestors: tuple[Path, ...] = ()) -> ET.Element:
    """Inline trusted upstream XML and resolve its mesh paths in memory."""
    path = path.resolve()
    if path in ancestors:
        raise ValueError(f"XML include 形成循环：{path}")
    root = ET.parse(path).getroot()

    def expand(element: ET.Element) -> None:
        for child in list(element):
            if child.tag == "include":
                included = _read_with_includes(path.parent / child.attrib["file"], ancestors + (path,))
                position = list(element).index(child)
                element.remove(child)
                for offset, replacement in enumerate(list(included)):
                    element.insert(position + offset, replacement)
            else:
                if child.tag in {"mesh", "texture", "hfield", "skin"} and "file" in child.attrib:
                    child.set("file", str((path.parent / child.attrib["file"]).resolve()))
                expand(child)

    expand(root)
    return root


def _look_at_xyaxes(position, target) -> str:
    def normalize(values):
        norm = math.sqrt(sum(value * value for value in values))
        if norm < 1e-12:
            raise ValueError('Display camera has a degenerate viewing direction')
        return tuple(value / norm for value in values)

    def cross(left, right):
        return (left[1] * right[2] - left[2] * right[1],
                left[2] * right[0] - left[0] * right[2],
                left[0] * right[1] - left[1] * right[0])

    z = normalize(tuple(a - b for a, b in zip(position, target)))
    x = normalize(cross((0., 0., 1.), z))
    y = cross(z, x)
    return _numbers((*x, *y))


def _add_display_cameras(root: ET.Element) -> None:
    world = root.find('worldbody')
    gripper = root.find(".//body[@name='vx300s_right/gripper_link']")
    if world is None or gripper is None:
        raise ValueError('Cannot locate display-camera world/right gripper attachment')
    specifications = (
        (world, 'display_front', (0.13, -0.45, 0.62), (0.13, 0.59, 0.16), 58),
        (world, 'display_side', (0.82, 0.60, 0.36), (0.13, 0.60, 0.06), 58),
        (gripper, 'display_wrist_right', (-0.12, 0., 0.12), (0.14, 0., 0.), 65),
    )
    existing = {camera.get('name') for camera in root.iter('camera')}
    for parent, name, position, target, fovy in specifications:
        if name in existing:
            raise ValueError('Display camera name already exists: ' + name)
        ET.SubElement(parent, 'camera', name=name, mode='fixed', pos=_numbers(position),
                      xyaxes=_look_at_xyaxes(position, target), fovy=str(fovy))
        existing.add(name)


def build_scene_xml(settings: SceneConfig, *, display_cameras: bool = False) -> str:
    _validate_wrist_damping(settings)
    _validate_right_wrist_kp(settings)
    validate_pose_randomization(settings)
    if type(display_cameras) is not bool:
        raise ValueError('display_cameras must be a boolean')
    from gym_aloha.constants import ASSETS_DIR

    root = _read_with_includes(Path(ASSETS_DIR) / "bimanual_viperx_transfer_cube.xml")
    root.set("model", "aloha_banana_box")
    removed = False
    for world in root.findall("worldbody"):
        for body in list(world):
            if body.tag == "body" and body.get("name") == "box":
                world.remove(body)
                removed = True
    if not removed:
        raise ValueError("gym-aloha 场景结构与预期不同，无法定位原来的方块。")
    if settings.controller_profile == "position_pd":
        # A shared low-level controller for expert collection and policy evaluation.
        # Gravity compensation applies to robot links only, never to task objects.
        gains = {"waist": (600, 12), "shoulder": (1200, 25), "elbow": (800, 12),
                 "forearm_roll": (120, 2), "wrist_angle": (200, 2),
                 "wrist_rotate": (100, 1.5), "left_finger": (600, 20), "right_finger": (600, 20)}
        for body in root.iter("body"):
            if body.get("name", "").startswith(("vx300s_left", "vx300s_right")):
                body.set("gravcomp", "1")
        for joint in root.iter("joint"):
            if joint.get("name", "").startswith(("vx300s_left/", "vx300s_right/")):
                kind = joint.get("name").split("/")[-1]
                if kind in gains:
                    joint.set("frictionloss", "0.02")
                    damping = gains[kind][1]
                    if settings.wrist_damping_scale != 1.0 and kind in {"forearm_roll", "wrist_angle", "wrist_rotate"}:
                        damping *= settings.wrist_damping_scale
                    joint.set("damping", str(damping))
        for body in root.iter("body"):
            for geom in list(body.findall("geom")):
                name = geom.get("name", "")
                if name.endswith(("/10_left_gripper_finger", "/10_right_gripper_finger")):
                    # Explicit flat rubber pads replace the convex mesh contact hull.
                    # The original mesh remains visible; no object attachment is added.
                    geom.set("name", name + "_visual")
                    geom.set("contype", "0")
                    geom.set("conaffinity", "0")
                    side_y = -0.020 if name.endswith("10_left_gripper_finger") else 0.020
                    ET.SubElement(body, "geom", {
                        "name": name, "type": "box", "pos": f"0.060 {side_y} 0",
                        "size": "0.025 0.003 0.012", "rgba": "0.12 0.12 0.12 1",
                        "condim": "6", "friction": "1.5 0.01 0.005", "solref": "0.008 1",
                        "solimp": "0.95 0.99 0.001", "priority": "1",
                    })
        for geom in root.iter("geom"):
            if geom.get("name", "").endswith(("/10_left_gripper_finger", "/10_right_gripper_finger")):
                # Finite contact patch: resist rolling as well as sliding for curved objects.
                geom.set("condim", "6")
                geom.set("friction", "1.5 0.01 0.005")
        for actuator in root.iter("position"):
            joint_name = actuator.get("joint", "")
            kind = joint_name.split("/")[-1]
            if kind in gains:
                kp = gains[kind][0]
                if joint_name == "vx300s_right/wrist_angle" and settings.right_wrist_angle_kp_scale != 1.0:
                    kp *= settings.right_wrist_angle_kp_scale
                actuator.set("kp", str(kp))
    if settings.table_rgba is not None:
        table = next(g for g in root.iter("geom") if g.get("name") == "table")
        table.attrib.pop("material", None)
        table.set("rgba", _numbers(settings.table_rgba))
    if settings.light_scale != 1.0:
        for light in root.iter("light"):
            diffuse = [float(x) for x in light.get("diffuse", "0.7 0.7 0.7").split()]
            light.set("diffuse", _numbers(min(1.0, v * settings.light_scale) for v in diffuse))
    # The old keyframe encodes the old cube pose; initialize the new body explicitly.
    for keyframe in root.findall("keyframe"):
        root.remove(keyframe)
    option = ET.SubElement(root, "option")
    option.set("timestep", "0.002")
    if settings.controller_profile == "position_pd":
        # Reduce soft-friction creep during physical pinch grasps.
        for key, value in {"cone": "elliptic", "impratio": "10", "solver": "Newton",
                           "iterations": "100", "noslip_iterations": "20"}.items():
            option.set(key, value)

    objects = ET.parse(Path(__file__).parent / "assets/banana_box.xml").getroot()
    banana = objects.find("./worldbody/body[@name='banana']")
    box = objects.find("./worldbody/body[@name='open_box']")
    banana.set("pos", _numbers(settings.banana_position))
    half_yaw = math.radians(settings.banana_yaw_degrees) / 2
    banana.set("quat", _numbers((math.cos(half_yaw), 0, 0, math.sin(half_yaw))))
    inertial = banana.find("inertial")
    inertial.set("mass", str(settings.banana_mass_kg))
    # Uniformly scale the approximate inertia with mass, keeping geometry fixed.
    inertial.set("diaginertia", _numbers(
        value * settings.banana_mass_kg / 0.12 for value in (0.00004, 0.00028, 0.00030)
    ))
    for geom in banana.findall("geom"):
        if geom.get("name", "").startswith("banana_segment_"):
            geom.set("friction", _numbers(settings.banana_friction))
    box.set("pos", _numbers(settings.box_position))
    for section in list(objects):
        root.append(section)
    if display_cameras:
        _add_display_cameras(root)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def preview_scene(config_path: Path, *, camera: str | None = None, snapshot_only: bool = False) -> Path:
    settings = load_scene_config(config_path)
    if camera is not None:
        settings.camera = camera
    xml = build_scene_xml(settings)
    from dm_control import mujoco
    from gym_aloha.constants import START_ARM_POSE
    from PIL import Image
    from .recording import write_json

    physics = mujoco.Physics.from_xml_string(xml)
    figure = None
    plt = None
    try:
        if physics.model.nq != 23 or physics.model.nu != 16:
            raise ValueError("场景应包含 ALOHA 的 16 个物理关节位置和香蕉的 7 维自由位姿。")
        target = list(START_ARM_POSE)
        half_yaw = math.radians(settings.banana_yaw_degrees) / 2
        banana_pose = (*settings.banana_position, math.cos(half_yaw), 0, 0, math.sin(half_yaw))
        timestep = float(physics.model.opt.timestep)
        steps_per_frame = round(0.04 / timestep)  # 25 rendered frames / simulated second.

        def reset() -> None:
            with physics.reset_context():
                physics.data.qpos[:16] = target
                physics.named.data.qpos["banana_joint"] = banana_pose
                physics.data.qvel[:] = 0
                physics.data.ctrl[:] = target
            settle_steps = round(settings.settle_seconds / timestep)
            if settle_steps:
                physics.step(nstep=settle_steps)

        def render(camera_name: str):
            return physics.render(height=480, width=640, camera_id=camera_name)

        reset()
        frame = render(settings.camera)
        output = PROJECT_ROOT / "outputs" / (
            "scene-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        )
        output.mkdir(parents=True, exist_ok=False)
        (output / "scene.xml").write_text(xml + "\n", encoding="utf-8")
        write_json(output / "scene.json", {
            "kind": "local_scene_preview", "scene": asdict(settings),
            "source_config": str(config_path.resolve()), "policy_connected": False,
            "arm_control": "fixed initial joint targets; no task policy",
        })
        Image.fromarray(frame).save(output / "scene.png")
        logging.info("本地预览：机器人保持初始关节目标；未连接 π0。截图及场景：%s", output)
        if snapshot_only:
            return output

        import matplotlib.pyplot as plt
        plt.ion()
        # Avoid Matplotlib also opening its default Save dialog on our S key.
        plt.rcParams["keymap.save"] = []
        figure, axes = plt.subplots(figsize=(9, 7))
        axes.set_axis_off()
        artist = axes.imshow(frame)
        figure.text(0.5, 0.03, "1: top | 2: angle | Space: pause | R: reset | S: save PNG | close: exit", ha="center", fontsize=9)
        figure.subplots_adjust(bottom=0.08, top=0.92)
        state = {"camera": settings.camera, "paused": False, "reset": False, "save": False}

        def on_key(event) -> None:
            if event.key in {"1", "2"}:
                state["camera"] = {"1": "top", "2": "angle"}[event.key]
            elif event.key in {" ", "space"}:
                state["paused"] = not state["paused"]
            elif event.key in {"r", "R"}:
                state["reset"] = True
            elif event.key in {"s", "S"}:
                state["save"] = True

        figure.canvas.mpl_connect("key_press_event", on_key)
        plt.show(block=False)
        while plt.fignum_exists(figure.number):
            started = time.perf_counter()
            if state["reset"]:
                reset()
                state["reset"] = False
            elif not state["paused"]:
                physics.step(nstep=steps_per_frame)
            frame = render(state["camera"])
            artist.set_data(frame)
            status = "paused" if state["paused"] else "running"
            axes.set_title(f"ALOHA banana + open box | {state['camera']} | t={physics.data.time:.2f}s | {status}")
            if state["save"]:
                name = "view-" + datetime.now().strftime("%H%M%S-%f") + ".png"
                Image.fromarray(frame).save(output / name)
                logging.info("已保存：%s", output / name)
                state["save"] = False
            figure.canvas.draw_idle()
            plt.pause(max(0.001, 0.04 - (time.perf_counter() - started)))
        return output
    finally:
        if plt is not None and figure is not None:
            plt.close(figure)
        physics.free()
