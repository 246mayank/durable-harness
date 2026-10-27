"""Run the simulated platform as a real HTTP server.

    python -m platform_sim --port 8700 --db sim.db --p-lost-ack 0.1 --seed 7
"""

from __future__ import annotations

import argparse

import uvicorn

from platform_sim.app import create_app
from platform_sim.faults import FaultConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8700)
    parser.add_argument("--db", default=":memory:", help="SQLite path (default: in-memory)")
    parser.add_argument("--p-429", type=float, default=0.0)
    parser.add_argument("--p-5xx", type=float, default=0.0)
    parser.add_argument("--p-lost-ack", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    faults = FaultConfig(args.p_429, args.p_5xx, args.p_lost_ack, args.seed)
    uvicorn.run(create_app(args.db, faults), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
