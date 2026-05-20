# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import argparse
import logging
import os
import socket
import torch

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


def main(args) -> None:
    """Build the policy wrapper and start the websocket server.

    The wrapper now owns un-normalization + chunk_size discovery so that all
    eval clients (LIBERO / SimplerEnv / etc.) just need to forward `examples`
    and consume already-unnormalized actions from the response.
    """
    if args.idle_timeout > 0 and not env_flag("STARVLA_ALLOW_IDLE_TIMEOUT"):
        logging.warning(
            "Ignoring positive idle_timeout=%s so evaluation is not interrupted by inactivity. "
            "Set STARVLA_ALLOW_IDLE_TIMEOUT=1 to opt in.",
            args.idle_timeout,
        )
        args.idle_timeout = -1

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but unavailable; falling back to CPU policy server.")
        device = "cpu"

    wrapper = PolicyServerWrapper(
        ckpt_path=args.ckpt_path,
        device=device,
        use_bf16=args.use_bf16,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # start websocket server; wrapper.metadata is sent at handshake.
    server = WebsocketPolicyServer(
        policy=wrapper,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=wrapper.metadata,
        max_concurrency=args.max_concurrency,
    )
    logging.info("server running ... metadata=%s", wrapper.metadata)
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--device", type=str, default=os.getenv("STARVLA_POLICY_DEVICE", "cuda"), choices=["cuda", "cpu"])
    parser.add_argument("--idle_timeout", type=int, default=-1, help="Idle timeout in seconds, -1 means never close")
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=int(os.getenv("STARVLA_SERVER_INFERENCE_CONCURRENCY", "1")),
        help="Maximum concurrent inference requests served by one policy model copy.",
    )
    parser.add_argument("--debugpy", action="store_true", help="explicitly wait for a debugpy attach; disabled by default")
    return parser


def env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def start_debugpy_once():
    """start debugpy once"""
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10095))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10095 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    if args.debugpy or env_flag("STARVLA_ENABLE_DEBUGPY"):
        print("🔍 DEBUGPY is enabled")
        start_debugpy_once()
    main(args)
