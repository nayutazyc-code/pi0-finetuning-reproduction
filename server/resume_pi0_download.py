"""Resume the original pi0_aloha_sim checkpoint with parallel HTTP byte ranges.

Run on the Linux server after stopping serve_pi0.py or the earlier downloader.
Uses only the Python standard library; requires an existing .partial directory.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, wait
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import threading
import time
from urllib.parse import quote, urlencode
from urllib.request import ProxyHandler, Request, build_opener


BUCKET = "openpi-assets"
PREFIX = "checkpoints/pi0_aloha_sim/"
MIB = 1024 * 1024


def direct_open(address: str | Request, timeout: float):
    # The AutoDL academic proxy does not list Google Storage as a supported site.
    return build_opener(ProxyHandler({})).open(address, timeout=timeout)


def get_manifest(timeout: float) -> list[dict]:
    query = urlencode({
        "prefix": PREFIX,
        "maxResults": 1000,
        "fields": "items(name,size,md5Hash,generation),nextPageToken",
    })
    address = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?{query}"
    with direct_open(address, timeout) as response:
        manifest = json.load(response)
    if manifest.get("nextPageToken"):
        raise RuntimeError("官方清单出现分页，已停止，未修改权重。")
    items = []
    for item in manifest.get("items", []):
        name = item["name"]
        if name.endswith("/"):
            continue
        if not name.startswith(PREFIX):
            raise RuntimeError("官方清单包含检查点目录之外的文件。")
        relative = PurePosixPath(name[len(PREFIX):])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("文件清单中的相对路径不合法。")
        if not item.get("md5Hash") or not item.get("generation"):
            raise RuntimeError(f"缺少完整性校验信息：{name}")
        item["relative"] = relative.as_posix()
        item["size"] = int(item["size"])
        items.append(item)
    if not items:
        raise RuntimeError("没有取得官方文件清单，已停止。")
    return items


def verify_file(path: Path, item: dict) -> None:
    if not path.is_file() or path.stat().st_size != item["size"]:
        raise RuntimeError(f"文件缺失或大小不匹配：{item['relative']}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "md5").digest()
    if base64.b64encode(digest).decode("ascii") != item["md5Hash"]:
        raise RuntimeError(f"MD5 校验失败：{item['relative']}；已有文件已保留。")


def part_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def download_part(address, path, start, end, total, stop, timeout):
    length = end - start + 1
    for attempt in range(3):
        if stop.is_set():
            return
        existing = part_size(path)
        if existing > length:
            raise RuntimeError(f"分段文件大小异常，已保留：{path}")
        if existing == length:
            return
        offset = start + existing
        request = Request(address, headers={
            "Range": f"bytes={offset}-{end}",
            "Accept-Encoding": "identity",
        })
        try:
            with direct_open(request, timeout) as response:
                expected_range = f"bytes {offset}-{end}/{total}"
                if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                    raise RuntimeError("服务器没有正确返回所请求的字节范围，已停止该连接。")
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise RuntimeError("服务器返回了压缩内容，已停止以免错误拼接。")
                with path.open("ab") as stream:
                    read_chunk = getattr(response, "read1", response.read)
                    while offset <= end:
                        if stop.is_set():
                            return
                        chunk = read_chunk(min(256 * 1024, end - offset + 1))
                        if not chunk:
                            raise OSError("连接在分段下载完成前中断。")
                        stream.write(chunk)
                        offset += len(chunk)
            return
        except (OSError, http.client.HTTPException):
            if attempt == 2:
                raise
            if stop.wait(1 + attempt):
                return


def copy_exact(source, destination, count):
    while count:
        chunk = source.read(min(8 * MIB, count))
        if not chunk:
            raise RuntimeError("合并期间已有文件变短，已停止；原权重文件已保留。")
        destination.write(chunk)
        count -= len(chunk)


def complete_file(path: Path, item: dict, parts_root: Path, workers: int, timeout: float):
    before = path.stat() if path.exists() else None
    base = before.st_size if before else 0
    total = item["size"]
    if base > total:
        raise RuntimeError(f"已有文件比官方文件更大，已保留：{path}")
    if base == total:
        return
    remaining = total - base
    identifier = hashlib.sha256(item["name"].encode()).hexdigest()[:20]
    folder = parts_root / identifier / item["generation"]
    folder.mkdir(parents=True, exist_ok=True)
    count = min(workers, remaining)
    width = (remaining + count - 1) // count
    segments = []
    for start in range(base, total, width):
        end = min(start + width, total) - 1
        segments.append((folder / f"{start}-{end}.part", start, end))
    assembled = folder / "assembled.tmp"
    cached = sum(part_size(segment[0]) for segment in segments)
    needed = total + max(0, remaining - cached) + 128 * MIB
    reusable_scratch = part_size(assembled)
    if shutil.disk_usage(folder).free + reusable_scratch < needed:
        raise RuntimeError("临时合并空间不足；请预留约 3 GB，已有权重已保留。")
    address = (
        f"https://storage.googleapis.com/{BUCKET}/"
        + quote(item["name"], safe="/")
        + "?generation=" + item["generation"]
    )
    print(f"\n待补齐：{item['relative']}，约 {remaining / MIB:.1f} MiB，{len(segments)} 路连接", flush=True)
    stop = threading.Event()
    pool = ThreadPoolExecutor(max_workers=len(segments))
    futures = []
    previous = cached
    previous_time = time.monotonic()
    try:
        for segment, start, end in segments:
            futures.append(pool.submit(
                download_part, address, segment, start, end, total, stop, timeout,
            ))
        while True:
            done, pending = wait(futures, timeout=1)
            for future in done:
                future.result()
            current = sum(part_size(segment[0]) for segment in segments)
            now = time.monotonic()
            speed = max(0, current - previous) / max(now - previous_time, 0.001)
            print(
                f"\r本次剩余部分：{current / MIB:.1f}/{remaining / MIB:.1f} MiB；"
                f"最近总速度：{speed / 1024:.0f} KiB/s       ",
                end="", flush=True,
            )
            previous, previous_time = current, now
            if not pending:
                break
        for segment, start, end in segments:
            if part_size(segment) != end - start + 1:
                raise RuntimeError(f"分段尚未下载完整：{segment}")
    finally:
        stop.set()
        for future in futures:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        print(flush=True)
    if before:
        after = path.stat()
        if (after.st_size, after.st_mtime_ns, after.st_ino) != (
            before.st_size, before.st_mtime_ns, before.st_ino,
        ):
            raise RuntimeError("原权重文件仍在被其他程序写入，已停止合并；请先停止其他下载。")
    elif path.exists():
        raise RuntimeError("有其他程序创建了同一个权重文件，已停止合并。")
    print("分段完成，正在临时合并并校验；原文件会保留到校验通过。", flush=True)
    with assembled.open("wb") as destination:
        if base:
            with path.open("rb") as source:
                copy_exact(source, destination, base)
        for segment, start, end in segments:
            with segment.open("rb") as source:
                copy_exact(source, destination, end - start + 1)
        destination.flush()
        os.fsync(destination.fileno())
    verify_file(assembled, item)
    if before:
        after = path.stat()
        if (after.st_size, after.st_mtime_ns, after.st_ino) != (
            before.st_size, before.st_mtime_ns, before.st_ino,
        ):
            raise RuntimeError("合并期间原文件发生变化，已保留合并结果并停止。")
    elif path.exists():
        raise RuntimeError("合并期间原文件被其他程序创建，已停止替换。")
    path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(assembled, path)
    for segment, _, _ in segments:
        segment.unlink()
    print("该权重文件 MD5 校验通过。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="原版 π0 ALOHA 检查点：保留进度、分段续传、完整校验")
    parser.add_argument("--workers", type=int, default=4, help="每个未完成文件的并发连接数，默认 4，最多 8")
    parser.add_argument("--timeout", type=float, default=30, help="每次网络读写的超时秒数")
    parser.add_argument("--cache-dir", type=Path, default=Path(
        os.environ.get("OPENPI_DATA_HOME", "/root/pi0-repro/cache/openpi")
    ))
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or args.timeout <= 0:
        parser.error("workers 需要在 1–8 之间，timeout 需要为正数。")
    final = args.cache_dir.expanduser().resolve() / BUCKET / "checkpoints/pi0_aloha_sim"
    partial = final.with_suffix(".partial")
    if not final.is_dir() and not partial.is_dir():
        raise RuntimeError("没有找到已有缓存或 .partial 目录；该工具不会从头下载整个模型。")
    with final.with_suffix(".lock").open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("原下载仍在运行。请先停止原下载，等命令提示符回来后再执行。") from None
        print("读取官方文件清单……", flush=True)
        items = get_manifest(args.timeout)
        root = final if final.is_dir() else partial
        if root == partial:
            for item in items:
                complete_file(
                    partial / item["relative"], item,
                    final.parent / ".pi0-download-segments",
                    args.workers, args.timeout,
                )
        print(f"\n开始校验全部 {len(items)} 个文件；校验期间没有网络下载进度是正常的。", flush=True)
        for item in items:
            verify_file(root / item["relative"], item)
            print(f"校验通过：{item['relative']}", flush=True)
        if root == partial:
            if final.exists():
                raise RuntimeError("正式缓存目录意外出现，已停止，未覆盖任何目录。")
            partial.rename(final)
        print(f"\n全部下载及校验完成：{final}\n可以重新启动 π0 服务。", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已停止；已有权重和分段文件均保留。可用相同命令继续。", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"\n停止：{error}\n已有权重和分段文件均保留。", file=sys.stderr)
        sys.exit(1)
