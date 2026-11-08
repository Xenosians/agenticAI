"""Draw the animated request-path diagram shown in the README.

One request travels the real path through Janus: the browser, Phoenix, the
Lua gateway, routing, a specialist, the trusted guard, approval, the MCP tool
and back. Each stage lights up in turn with a one-line note on what it
enforces. The output is a single self-contained SVG with CSS animation only
(no script), so GitHub renders and plays it inline.

Usage: python scripts/render_request_flow.py [--out docs/assets/request-flow.svg]
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

WIDTH, HEIGHT = 960, 420
NODE_W, NODE_H = 150, 64
CYCLE_SECONDS = 18.0
INK, PAPER, MUTED, LINE = "#f2f2f2", "#0b0b0b", "#b0b0b0", "#6e6e6e"


@dataclass(frozen=True)
class Stage:
    key: str
    title: str
    subtitle: str
    note: str
    x: int
    y: int


# Positions form a loop: out along the top row, down, back along the bottom row.
STAGES = (
    Stage("browser", "Browser", "Nim / Karax", "You ask in chat; the browser keeps no secrets or tokens.", 40, 70),
    Stage("phoenix", "Phoenix", "sessions, jobs", "Phoenix checks your session and CSRF, saves a durable job.", 225, 70),
    Stage("gateway", "Lua gateway", "every outbound link", "Every model call, network link and tool call crosses Lua first.", 410, 70),
    Stage("router", "Hub routing", "semantic intent", "The hub names the specialist and the values you gave.", 595, 70),
    Stage("specialist", "Specialist", "one proposal", "A specialist proposes one tool call: a proposal, never authority.", 780, 70),
    Stage("guard", "Trusted guard", "policy, grounding", "Trusted code checks the call against policy and your exact words.", 780, 280),
    Stage("approval", "Approval", "you decide", "Anything that changes data waits for your approval of exact values.", 595, 280),
    Stage("tool", "MCP tool", "real integration", "The tool runs once against the real system and reads the result back.", 410, 280),
    Stage("result", "Result", "verified outcome", "Phoenix records the verified outcome; unknown stays unknown.", 225, 280),
    Stage("answer", "Answer", "in your chat", "You see the answer and the evidence behind it, in your chat.", 40, 280),
)


def centre(stage: Stage) -> tuple[float, float]:
    return stage.x + NODE_W / 2, stage.y + NODE_H / 2


def dock(stage: Stage) -> tuple[float, float]:
    """Where the token rests on a box: the edge facing the middle band, so its
    moves run through the gaps and never across a label."""
    cx, _ = centre(stage)
    return cx, (stage.y + NODE_H if stage.y < HEIGHT / 2 else stage.y)


def edge_points(a: Stage, b: Stage) -> tuple[float, float, float, float]:
    """Start and end points on the facing sides of two nodes."""
    (ax, ay), (bx, by) = centre(a), centre(b)
    if abs(ax - bx) > abs(ay - by):
        direction = 1 if bx > ax else -1
        return ax + direction * NODE_W / 2, ay, bx - direction * NODE_W / 2, by
    direction = 1 if by > ay else -1
    return ax, ay + direction * NODE_H / 2, bx, by - direction * NODE_H / 2


def keyframes(name: str, index: int, count: int, on: str, off: str, prop: str) -> str:
    """A property that is `on` only during this stage's slice of the cycle."""
    slice_pct = 100 / count
    start, end = index * slice_pct, (index + 1) * slice_pct
    stops = [(0.0, off)]
    if start > 0:
        stops.append((max(start - 0.01, 0), off))
    stops += [(start, on), (max(end - 0.5, start), on), (min(end, 100), off), (100.0, off)]
    body = " ".join(f"{pct:.2f}% {{ {prop}: {value}; }}" for pct, value in stops)
    return f"@keyframes {name} {{ {body} }}"


def render() -> str:
    count = len(STAGES)
    styles = [
        f"text {{ font-family: ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif; fill: {INK}; }}",
        ".title { font-size: 15px; font-weight: 600; }",
        f".subtitle {{ font-size: 12px; fill: {MUTED}; }}",
        ".note { font-size: 15px; opacity: 0; }",
        f".edge {{ stroke: {LINE}; stroke-width: 1.5; fill: none; marker-end: url(#arrow); }}",
        f".box {{ fill: {PAPER}; stroke: {LINE}; stroke-width: 1.5; }}",
        f".lit {{ fill: none; stroke: {INK}; stroke-width: 2.5; opacity: 0; }}",
        f".token {{ fill: {INK}; }}",
    ]
    nodes, notes = [], []
    for index, stage in enumerate(STAGES):
        styles.append(keyframes(f"lit-{stage.key}", index, count, "1", "0", "opacity"))
        styles.append(f".lit-{stage.key} {{ animation: lit-{stage.key} {CYCLE_SECONDS}s linear infinite; }}")
        styles.append(f".note-{stage.key} {{ animation: lit-{stage.key} {CYCLE_SECONDS}s linear infinite; }}")
        cx, _ = centre(stage)
        nodes.append(
            f'<g><rect class="box" x="{stage.x}" y="{stage.y}" width="{NODE_W}" height="{NODE_H}" rx="10"/>'
            f'<rect class="lit lit-{stage.key}" x="{stage.x - 3}" y="{stage.y - 3}" width="{NODE_W + 6}" '
            f'height="{NODE_H + 6}" rx="12"/>'
            f'<text class="title" x="{cx}" y="{stage.y + 28}" text-anchor="middle">{escape(stage.title)}</text>'
            f'<text class="subtitle" x="{cx}" y="{stage.y + 47}" text-anchor="middle">{escape(stage.subtitle)}</text>'
            "</g>"
        )
        notes.append(
            f'<text class="note note-{stage.key}" x="{WIDTH / 2}" y="210" text-anchor="middle">'
            f"{index + 1}. {escape(stage.note)}</text>"
        )

    edges, path = [], []
    for a, b in zip(STAGES, STAGES[1:]):
        x1, y1, x2, y2 = edge_points(a, b)
        edges.append(f'<line class="edge" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"/>')
    stops = [dock(stage) for stage in STAGES]
    for x, y in stops:
        path.append(f"{'M' if not path else 'L'} {x} {y}")

    # The token rests on a box while it is lit, then moves to the next box in
    # the last quarter of that slice, so the dot and the highlight stay in step.
    # Distances along the path decide where each box sits as a fraction of it.
    lengths = [0.0]
    for (ax, ay), (bx, by) in zip(stops, stops[1:]):
        lengths.append(lengths[-1] + abs(bx - ax) + abs(by - ay))
    points = [length / lengths[-1] for length in lengths]
    times, positions = [], []
    for index in range(count):
        start, end = index / count, (index + 1) / count
        times += [start, start + (end - start) * 0.75]
        positions += [points[index], points[index]]
    times.append(1.0)
    positions.append(points[-1])
    token = (
        f'<circle class="token" r="5"><animateMotion dur="{CYCLE_SECONDS}s" repeatCount="indefinite" '
        f'calcMode="linear" keyTimes="{";".join(f"{t:.4f}" for t in times)}" '
        f'keyPoints="{";".join(f"{p:.4f}" for p in positions)}" path="{" ".join(path)}"/></circle>'
    )

    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" '
        f'height="{HEIGHT}" role="img" aria-labelledby="flow-title flow-desc">',
        '<title id="flow-title">How one request travels through Janus</title>',
        '<desc id="flow-desc">'
        + escape(" ".join(f"{i + 1}. {s.title}: {s.note}" for i, s in enumerate(STAGES)))
        + "</desc>",
        "<defs>",
        f'<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{LINE}"/></marker>',
        "</defs>",
        "<style>" + "\n".join(styles) + "</style>",
        f'<rect width="{WIDTH}" height="{HEIGHT}" rx="16" fill="{PAPER}"/>',
        f'<text class="subtitle" x="40" y="40">How one request travels through Janus</text>',
        *edges,
        *nodes,
        *notes,
        token,
        "</svg>",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the animated request-path diagram shown in the README.")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "docs/assets/request-flow.svg")
    out = parser.parse_args().out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(), encoding="utf-8", newline="\n")
    print(out)


if __name__ == "__main__":
    main()
