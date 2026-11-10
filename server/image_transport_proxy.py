"""Optional lossless RGB transport sidecar; the original pi0 server stays intact."""
import argparse
import asyncio
import hashlib
import io
import logging
import time

import numpy as np
from PIL import Image
from openpi_client import msgpack_numpy
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed


def decode_images(observation):
    if "images_png" not in observation:
        return observation, None
    if "images" in observation:
        raise ValueError("Use either images or images_png, not both")
    images = {}
    encoded_bytes = raw_bytes = 0
    started = time.perf_counter()
    digests = observation.get("image_sha256", {})
    payloads = observation["images_png"]
    if set(payloads) != {"cam_high"} or set(digests) != set(payloads):
        raise ValueError("PNG sidecar expects one cam_high image and its digest")
    for name, payload in payloads.items():
        if not isinstance(payload, bytes) or len(payload) > 2 * 1024 * 1024:
            raise ValueError("Invalid PNG payload")
        with Image.open(io.BytesIO(payload)) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (640, 480):
                raise ValueError("PNG must be a raw 640x480 RGB camera image")
            array = np.ascontiguousarray(np.asarray(image).transpose(2, 0, 1))
        if hashlib.sha256(array.tobytes()).hexdigest() != digests[name]:
            raise ValueError("Decoded RGB digest mismatch")
        images[name] = array
        encoded_bytes += len(payload)
        raw_bytes += array.nbytes
    result = {key: value for key, value in observation.items() if key not in {"images_png", "image_sha256"}}
    result["images"] = images
    return result, {"codec": "png", "png_bytes": encoded_bytes, "raw_image_bytes": raw_bytes,
                    "decode_ms": (time.perf_counter() - started) * 1000, "pixel_digest_verified": True}


async def handler(socket, upstream_port):
    packer = msgpack_numpy.Packer()
    try:
        request = getattr(socket, "request", None)
        if request is not None and request.path == "/transport-probe":
            # Loopback diagnostic only: echo bounded synthetic bytes without
            # opening an upstream model connection or touching policy state.
            await socket.send(packer.pack({"schema": "pi0-transport-echo-v1", "reply_bytes": 8192}))
            async for raw in socket:
                if not isinstance(raw, bytes) or not 8192 <= len(raw) <= 1024*1024:
                    await socket.close(code=1008, reason="Bounded binary probe required")
                    return
                await socket.send(raw[:8192])
            return
        async with connect(f"ws://127.0.0.1:{upstream_port}", compression=None,
                           max_size=16*1024*1024, open_timeout=15, ping_interval=None) as upstream:
            raw = await upstream.recv()
            if isinstance(raw, str):
                raise RuntimeError(raw)
            metadata = msgpack_numpy.unpackb(raw)
            metadata["image_transport"] = ["raw", "png"]
            metadata["transport_proxy"] = {"kind": "lossless-png", "upstream_port": upstream_port,
                                           "timing_schema": "pi0-proxy-timing-v1"}
            await socket.send(packer.pack(metadata))
            async for raw in socket:
                if isinstance(raw, str):
                    raise ValueError("Binary MessagePack expected")
                message_started = time.perf_counter()
                observation = msgpack_numpy.unpackb(raw)
                unpacked = time.perf_counter()
                observation, transport = decode_images(observation)
                payload = packer.pack(observation)
                upstream_started = time.perf_counter()
                await upstream.send(payload)
                reply = await upstream.recv()
                upstream_finished = time.perf_counter()
                if isinstance(reply, str):
                    await socket.send(reply)
                    return
                result = msgpack_numpy.unpackb(reply)
                if transport is not None:
                    transport.update({"timing_schema": "pi0-proxy-timing-v1",
                        "request_unpack_ms": (unpacked-message_started)*1000,
                        "proxy_upstream_round_trip_ms": (upstream_finished-upstream_started)*1000,
                        "proxy_handler_ms_before_reply_pack": (time.perf_counter()-message_started)*1000,
                        "request_bytes": len(raw)})
                    result["transport_timing"] = transport
                await socket.send(packer.pack(result))
    except ConnectionClosed:
        return
    except Exception as exc:
        logging.exception("Image transport failed")
        try:
            await socket.send(f"Image transport failed: {type(exc).__name__}: {exc}")
            await socket.close(code=1011, reason="Image transport error")
        except ConnectionClosed:
            pass


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--upstream-port", type=int, default=8000)
    args = parser.parse_args()
    logging.info("Lossless PNG transport on %s:%d -> localhost:%d", args.host, args.port, args.upstream_port)
    async with serve(lambda socket: handler(socket, args.upstream_port), args.host, args.port,
                     compression=None, max_size=3*1024*1024, ping_interval=20, ping_timeout=120):
        await asyncio.Future()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
