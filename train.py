#!/usr/bin/env python3
"""Train an AI to play PyCrossy.

By default this opens TWO windows: the game (the AI actively playing) and a live analytics
dashboard. Use ``--headless`` for fast training with no windows (CSV/JSON/TensorBoard
metrics are always written).

Examples:
    python train.py --algo neat                 # live dual-window NEAT training
    python train.py --algo ppo --episodes 5000  # PPO
    python train.py --algo dqn --headless        # fast headless DQN
    python train.py --algo es --resume runs/es   # resume from a checkpoint
    python train.py --list                        # list available algorithms
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys

from ai import algorithms
from ai.trainer import Trainer, TrainConfig


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train an AI to play PyCrossy")
    p.add_argument("--algo", default="neat", help="algorithm: " + ", ".join(algorithms.available()))
    p.add_argument("--episodes", type=int, default=3000, help="target training episodes")
    p.add_argument("--logdir", default=None, help="output dir (default runs/<algo>)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--headless", action="store_true", help="no windows; fast training")
    p.add_argument("--parallel", type=int, default=0,
                   help="evaluate populations across N worker processes (evolutionary algos)")
    p.add_argument("--no-dashboard", action="store_true", help="game window only (no dashboard)")
    p.add_argument("--speed", type=float, default=3.0, help="live play speed multiplier (x real-time)")
    p.add_argument("--dashboard-hz", type=float, default=12.0,
                   help="dashboard graph refresh rate (5-20 recommended)")
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--resume", default=None, help="checkpoint dir/file to resume from")
    p.add_argument("--list", action="store_true", help="list algorithms and exit")
    p.add_argument("--no-tensorboard", action="store_true")
    p.add_argument("--eval-episodes", type=int, default=5,
                   help="episodes per candidate for CRN averaging / per eval")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VAL",
                   help="override an algorithm hyperparameter (repeatable), e.g. --set pop_size=100")
    p.add_argument("--config", default=None, help="JSON file of algorithm hyperparameters")
    return p.parse_args(argv)


def _coerce(v: str):
    """Coerce a CLI string to int → float → bool → tuple → str."""
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if "," in v:                                  # e.g. hidden=64,64
        return tuple(_coerce(x) for x in v.split(","))
    return v


def build_algo_cfg(args) -> dict:
    """Assemble ``algo_cfg`` from an optional --config JSON file and repeated --set overrides."""
    import json
    cfg: dict = {}
    if args.config:
        with open(args.config) as fh:
            cfg.update(json.load(fh))
    for item in args.overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VAL, got {item!r}")
        k, v = item.split("=", 1)
        cfg[k.strip()] = _coerce(v.strip())
    return cfg


def build_trainer(args) -> Trainer:
    logdir = args.logdir or os.path.join("runs", args.algo)
    cfg = TrainConfig(algo=args.algo, logdir=logdir, seed=args.seed,
                      target_episodes=args.episodes, eval_every=args.eval_every,
                      eval_episodes=args.eval_episodes,
                      checkpoint_every=args.checkpoint_every,
                      use_tensorboard=not args.no_tensorboard,
                      algo_cfg=build_algo_cfg(args))
    trainer = Trainer(cfg)
    if args.resume:
        path = args.resume if os.path.isfile(args.resume) else os.path.join(args.resume, "checkpoint.pkl")
        trainer.load_checkpoint(path)
        print(f"resumed from {path} at episode {trainer.episode}")
    return trainer


def run_headless(args) -> None:
    trainer = build_trainer(args)
    is_pop = getattr(trainer.algo, "supports_parallel", False)
    # Population algorithms (NEAT/ES/GA/CMA-ES) default to the parallel GENERATION path — it
    # is the only path with Common Random Numbers + held-out validation (see trainer). Pass
    # --parallel 1 to force the legacy sequential evaluator.
    if is_pop and args.parallel != 1:
        workers = args.parallel if args.parallel > 1 else max(1, (os.cpu_count() or 2) - 1)
        print(f"[headless|parallel x{workers}] {args.algo} -> {trainer.cfg.logdir} "
              f"(CRN K={trainer.cfg.eval_episodes}, validated champion)")

        def on_gen(metrics, snap):
            print(f"gen {metrics.get('generation', 0):4d} | best_eval {trainer.best_eval:6.2f} | "
                  f"peak {snap['best_score']:3d} | mean_fit {metrics.get('mean_fitness', 0):7.2f} | "
                  f"species {int(metrics.get('species', 0)):2d} | {snap['steps_per_sec']:.0f} steps/s")

        trainer.train_parallel(n_workers=workers, on_generation=on_gen)
        trainer.save_checkpoint()
        print(f"done. best validated mean score = {trainer.best_eval:.2f}. metrics in {trainer.cfg.logdir}")
        return
    print(f"[headless] training {args.algo} for {args.episodes} episodes -> {trainer.cfg.logdir}")
    last = [0]

    def on_episode(summary, snap):
        if summary["episode"] - last[0] >= 25:
            last[0] = summary["episode"]
            print(f"ep {summary['episode']:5d} | score {summary['score']:3d} | "
                  f"reward {summary['reward']:7.2f} | best {snap['best_score']:3d} | "
                  f"{snap['episodes_per_sec']:.1f} eps/s")

    trainer.train(args.episodes, on_episode=on_episode)
    print(f"done. metrics in {trainer.cfg.logdir} (metrics.csv / summary.json / tensorboard)")


def run_live(args) -> None:
    import time

    from pycrossy import config, gpu
    from ai.game_window import LivePlayWindow

    # Prefer the dedicated GPU + native (Wayland) video backend BEFORE spawning the dashboard
    # subprocess, so it inherits the same stable, flicker-free presentation settings.
    gpu.prefer_high_performance_gpu(config.PREFER_DEDICATED_GPU)
    trainer = build_trainer(args)
    # Pace the engine to `speed`× real-time; render at most ~120 FPS regardless of speed.
    tick_budget = 1.0 / (60.0 * max(0.1, args.speed))
    render_interval = 1.0 / 120.0
    # Population algorithms can evaluate the generation across worker processes while the
    # window showcases one genome. PPO/A2C/DQN aren't population-based, so --parallel is N/A.
    parallel = bool(args.parallel) and getattr(trainer.algo, "supports_parallel", False)

    # Create the worker pool NOW — from the main thread, before any GL window exists — so
    # workers fork cleanly once and are then reused every generation (never forked from
    # inside the render loop, which would be unsafe with a live GL context).
    pool = None
    if parallel:
        from ai import vec_env
        pool = vec_env.make_pool(args.parallel, trainer.cfg.max_steps)

    dash_proc = None
    q = None
    if not args.no_dashboard:
        ctx = mp.get_context("spawn")
        q = ctx.Queue(maxsize=8)
        from ai.dashboard import run_dashboard
        dash_proc = ctx.Process(target=run_dashboard,
                                args=(q, {"logdir": trainer.cfg.logdir, "refresh_hz": args.dashboard_hz}),
                                daemon=True)
        dash_proc.start()

    window = LivePlayWindow(title=f"PyCrossy — AI ({args.algo})")
    hud = {"algo": args.algo, "episode": 0, "score": 0, "best_score": 0, "generation": None,
           "workers": args.parallel if parallel else 0}
    env = trainer.env
    last_render = [0.0]
    last_tick = [time.monotonic()]

    def render_tick():
        # Pump events every tick (keeps the window responsive — no "not responding"),
        # render at most ~120 FPS (cheap GPU), and pace the engine to speed× real-time.
        now = time.monotonic()
        if now - last_render[0] >= render_interval:
            hud["score"] = env.max_z
            window.render(env.scene, hud)
            last_render[0] = now
        if not window.pump():
            raise KeyboardInterrupt
        delay = tick_budget - (time.monotonic() - last_tick[0])
        if delay > 0:
            time.sleep(delay)
        last_tick[0] = time.monotonic()

    def push_snapshot():
        if q is not None:
            try:
                q.put_nowait(trainer.logger.snapshot())
            except Exception:
                pass

    try:
        if parallel:
            _run_live_parallel(args, trainer, window, hud, env, render_tick, push_snapshot,
                               last_render, render_interval, pool)
        else:
            print(f"[live] training {args.algo}; two windows open. Esc to stop.")
            for _ in range(args.episodes):
                hud["episode"] = trainer.episode + 1
                trainer.train_episode(render_tick=render_tick)
                prog = trainer.algo.progress
                hud["best_score"] = trainer.best_score
                hud["generation"] = prog.get("generation")
                push_snapshot()
                if not window.pump():       # stay responsive during the inter-episode gap
                    break
                if not window.running:
                    break
    except KeyboardInterrupt:
        print("\nstopping…")
    finally:
        trainer.save_checkpoint()
        trainer.save_best()
        trainer.logger.close()
        if q is not None:
            try:
                q.put_nowait("STOP")
            except Exception:
                pass
        if dash_proc is not None:
            dash_proc.join(timeout=2)
        if pool is not None:
            pool.terminate()
            pool.join()
        window.close()
        print(f"saved checkpoint + best model in {trainer.cfg.logdir}")


def _run_live_parallel(args, trainer, window, hud, env, render_tick, push_snapshot,
                       last_render, render_interval, pool) -> None:
    """Live training for population algorithms: showcase the current best in the window while
    the whole generation is evaluated across ``args.parallel`` workers in the background."""
    import threading
    import time

    pop = max(1, len(trainer.algo.population_payloads()))
    gens = max(1, args.episodes // pop)
    print(f"[live|parallel x{args.parallel}] {args.algo}: {args.parallel} workers evaluate "
          f"each generation of {pop}; the window showcases the current best. Esc to stop.")

    for _ in range(gens):
        if not window.running:
            break
        hud["generation"] = trainer.algo.progress.get("generation")
        hud["episode"] = trainer.episode
        # 1) Showcase the current best policy, rendered live (raises KeyboardInterrupt on Esc).
        trainer.showcase_episode(render_tick=render_tick)
        if not window.pump():
            break
        # 2) Evaluate the whole generation across the worker pool in a background thread; keep
        #    the window responsive (showing the last frame) until it finishes.
        err = []

        def work():
            try:
                trainer.train_generation(n_workers=args.parallel, pool=pool,
                                         on_generation=lambda m, s: push_snapshot())
            except Exception as exc:                  # surface to the main thread
                err.append(exc)

        th = threading.Thread(target=work, daemon=True)
        th.start()
        while th.is_alive():
            now = time.monotonic()
            if now - last_render[0] >= render_interval:
                hud["score"] = env.max_z
                window.render(env.scene, hud)
                last_render[0] = now
            if not window.pump():
                window.running = False
                break
            time.sleep(0.02)
        th.join()                                     # let the generation finish cleanly
        if err:
            raise err[0]
        hud["best_score"] = trainer.best_score


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.list:
        print("Available algorithms:", ", ".join(algorithms.available()))
        return
    if args.algo.lower() not in algorithms.available():
        print(f"Unknown algorithm '{args.algo}'. Available: {', '.join(algorithms.available())}")
        sys.exit(1)

    from ai.base import _REGISTRY
    cls = _REGISTRY[args.algo.lower()]
    if args.parallel and not getattr(cls, "supports_parallel", False):
        print(f"note: --parallel only applies to population algorithms (NEAT/ES/GA/CMA-ES); "
              f"ignored for '{args.algo}'.")
    if getattr(cls, "uses_planning", False):
        print(f"note: '{args.algo}' is a PLANNING agent — it searches each move live and does NOT\n"
              f"      learn, so this only SHOWCASES it playing (training metrics won't improve;\n"
              f"      to just watch it, use Auto-Play in the app). Tune strength vs speed with\n"
              f"      --set max_depth=N (2-8, default 6); --parallel does not apply, and --speed\n"
              f"      paces the sim but each move still costs its search time.")

    if args.headless:
        run_headless(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
