"""`triforge` command line.

    triforge models                         list registered specialists and whether they are installed
    triforge route "integrate x^2 dx"       show the router's recommendation
    triforge generate -m auto "prompt"      one-shot generation with stats
    triforge chat [-m tricode]              interactive session
    triforge serve [--port 8765]            local HTTP API (see triforge.inference.api)
    triforge sysinfo                        hardware / memory snapshot

Chat commands: /model NAME|auto  /temp X  /max N  /topk N  /topp X  /stats  /load NAME  /unload NAME
               /models  /quit
"""

from __future__ import annotations

import argparse
import json
import sys

from .router import route


def _engine(args):
    from .engine import Engine
    return Engine(args.registry, device=args.device, backend=args.backend)


def cmd_chat(args):
    eng = _engine(args)
    settings = {"model": args.model, "temperature": args.temperature, "max_new_tokens": args.max_new_tokens,
                "top_k": args.top_k, "top_p": args.top_p}
    last = {}
    print("TriForge chat. /quit to exit, /stats for the last generation's stats.")
    while True:
        try:
            line = input(f"[{settings['model']}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            cmd, *rest = line[1:].split(maxsplit=1)
            arg = rest[0] if rest else ""
            if cmd in ("quit", "exit"):
                break
            elif cmd == "model":
                settings["model"] = arg or "auto"
            elif cmd in ("temp", "max", "topk", "topp"):
                key = {"temp": "temperature", "max": "max_new_tokens", "topk": "top_k", "topp": "top_p"}[cmd]
                settings[key] = (int if cmd in ("max", "topk") else float)(arg)
            elif cmd == "stats":
                print(json.dumps(last, indent=2))
            elif cmd == "load":
                print(json.dumps(eng.load(arg), indent=2))
            elif cmd == "unload":
                eng.unload(arg)
            elif cmd == "models":
                print(json.dumps(eng.available(), indent=2))
            else:
                print(__doc__)
            continue
        try:
            text, last = eng.generate(line, model=settings["model"], max_new_tokens=settings["max_new_tokens"],
                                      temperature=settings["temperature"], top_k=settings["top_k"],
                                      top_p=settings["top_p"], stream=lambda s: print(s, end="", flush=True))
        except (KeyError, FileNotFoundError) as e:
            print(f"error: {e}")
            continue
        print(f"\n-- {last['model']} | {last['new_tokens']} tok | {last['decode_tokens_per_s']} tok/s")


def main(argv=None):
    p = argparse.ArgumentParser(prog="triforge")
    p.add_argument("--registry", default="configs/models.yaml")
    p.add_argument("--device", default=None)
    p.add_argument("--backend", default="reference", choices=["reference", "int8", "addonly", "triton"])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("models")
    sub.add_parser("sysinfo")
    r = sub.add_parser("route")
    r.add_argument("prompt")
    for name in ("generate", "chat"):
        g = sub.add_parser(name)
        if name == "generate":
            g.add_argument("prompt")
        g.add_argument("-m", "--model", default="auto")
        g.add_argument("--temperature", type=float, default=0.7)
        g.add_argument("--max-new-tokens", type=int, default=256)
        g.add_argument("--top-k", type=int, default=50)
        g.add_argument("--top-p", type=float, default=0.95)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    args = p.parse_args(argv)

    if args.cmd == "route":
        rt = route(args.prompt)
        print(json.dumps({"specialist": rt.specialist, "scores": rt.scores, "reason": rt.reason}, indent=2))
    elif args.cmd == "models":
        print(json.dumps(_engine(args).available(), indent=2))
    elif args.cmd == "sysinfo":
        from ..hardware import detect
        from .engine import system_stats
        print(json.dumps({**detect().__dict__, **system_stats()}, indent=2, default=str))
    elif args.cmd == "generate":
        text, stats = _engine(args).generate(args.prompt, model=args.model, max_new_tokens=args.max_new_tokens,
                                             temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
        print(text)
        print(json.dumps(stats, indent=2), file=sys.stderr)
    elif args.cmd == "chat":
        cmd_chat(args)
    elif args.cmd == "serve":
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(_engine(args)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
