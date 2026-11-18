"""Label Studio local cho tập test (phase 08 bước 4–5): tạo project, nạp nhãn đã soát, đếm duyệt, xuất nhãn.

Server chạy ngoài repo (dữ liệu ``D:\\DevCache\\label-studio``), chỉ nghe 127.0.0.1; lệnh khởi động ở
``data/real/labels/README.md``. Tài khoản + token đọc từ file ``credentials.json`` trong thư mục dữ liệu đó (biến
``XLA_LS_CREDENTIALS`` ghi đè), không bao giờ nằm trong repo.

Dùng::

    python -m xla.detect.labelstudio setup --json    # tạo project xla-test-v1 + kho local files + nạp khung test
                                                     #   kèm nhãn agent đã soát làm predictions (idempotent)
    python -m xla.detect.labelstudio status --json   # số khung đã / chưa Submit
    python -m xla.detect.labelstudio export --json   # annotation đã Submit → data/real/labels/test/*.txt + manifest;
                                                     #   exit 1 khi còn khung chưa duyệt (không ghi gì)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any

from xla.detect.devval import LABELS_REL, load_manifest, save_manifest
from xla.detect.prelabel import H, W, to_ls_result
from xla.detect.review import AGENT_TEST_REL, REVIEWED_JSON_REL, write_labels
from xla.paths import repo_root

PROJECT_TITLE = "xla-test-v1"
CONFIG_REL = Path("configs/labelstudio.xml")
DEFAULT_URL = "http://127.0.0.1:8080"
DEFAULT_DATA_DIR = Path(r"D:\DevCache\label-studio")
DOC_ROOT_REL = Path("data/real")  # LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = <repo>/data/real
FRAMES_SUB = "frames"
MATCH_IOU = 0.5
SAME_IOU = 0.95  # cặp hộp IoU ≥ ngưỡng này coi như chủ dự án không sửa


def credentials() -> dict[str, str]:
    p = Path(os.environ.get("XLA_LS_CREDENTIALS", DEFAULT_DATA_DIR / "credentials.json"))
    if not p.exists():
        raise SystemExit(f"thiếu {p} (tạo lúc dựng server, xem data/real/labels/README.md)")
    return json.loads(p.read_text(encoding="utf-8"))


class Client:
    def __init__(self, url: str | None = None, token: str | None = None) -> None:
        import requests

        c = credentials() if token is None else {}
        self.url = (url or os.environ.get("XLA_LS_URL") or c.get("url") or DEFAULT_URL).rstrip("/")
        self.s = requests.Session()
        self.s.headers["Authorization"] = f"Token {token or c['token']}"

    def get(self, path: str, **kw: Any) -> Any:
        r = self.s.get(self.url + path, timeout=60, **kw)
        r.raise_for_status()
        return r.json()

    def post(self, path: str, payload: Any) -> Any:
        r = self.s.post(self.url + path, json=payload, timeout=120)
        if r.status_code >= 400:
            raise SystemExit(f"POST {path} → {r.status_code}: {r.text[:500]}")
        return r.json() if r.content else None

    def project(self) -> dict[str, Any] | None:
        res = self.get("/api/projects", params={"title": PROJECT_TITLE, "page_size": 100})
        items = res["results"] if isinstance(res, dict) else res
        return next((p for p in items if p["title"] == PROJECT_TITLE), None)

    def tasks(self, pid: int) -> list[dict[str, Any]]:
        out, page = [], 1
        while True:
            r = self.s.get(f"{self.url}/api/tasks", params={"project": pid, "page": page, "page_size": 100,
                                                            "fields": "all"}, timeout=120)
            if r.status_code == 404:
                break
            r.raise_for_status()
            body = r.json()
            items = body["tasks"] if isinstance(body, dict) else body
            if not items:
                break
            out += items
            if len(items) < 100:
                break
            page += 1
        return out


def task_frame(t: dict[str, Any]) -> str:
    return t["data"]["frame"]


def submitted(t: dict[str, Any]) -> dict[str, Any] | None:
    """Annotation Submit mới nhất (bỏ annotation bị huỷ/skip); None khi chưa duyệt."""
    anns = [a for a in t.get("annotations", []) if not a.get("was_cancelled")]
    if not anns:
        return None
    return max(anns, key=lambda a: (a.get("updated_at") or a.get("created_at") or "", a["id"]))


def ann_boxes(a: dict[str, Any]) -> tuple[list[list[float]], bool]:
    boxes, hard = [], False
    for r in a.get("result", []):
        if r.get("type") == "rectanglelabels":
            v = r["value"]
            x1, y1 = v["x"] / 100 * W, v["y"] / 100 * H
            x2, y2 = x1 + v["width"] / 100 * W, y1 + v["height"] / 100 * H
            boxes.append([max(0.0, x1), max(0.0, y1), min(float(W), x2), min(float(H), y2)])
        elif r.get("type") == "choices" and "hard" in r["value"].get("choices", []):
            hard = True
    return boxes, hard


def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def diff_boxes(agent: list[list[float]], owner: list[list[float]]) -> dict[str, int]:
    """Đếm sửa của chủ dự án so với nhãn agent: ghép tham lam theo IoU ≥ 0,5."""
    pairs = sorted(((_iou(a, o), i, j) for i, a in enumerate(agent) for j, o in enumerate(owner)), reverse=True)
    ua, uo, modified = set(), set(), 0
    for v, i, j in pairs:
        if v < MATCH_IOU or i in ua or j in uo:
            continue
        ua.add(i)
        uo.add(j)
        if v < SAME_IOU:
            modified += 1
    return {"deleted": len(agent) - len(ua), "added": len(owner) - len(uo), "modified": modified}


def load_reviewed(root: Path) -> dict[str, Any]:
    return json.loads((root / REVIEWED_JSON_REL).read_text(encoding="utf-8"))["frames"]


def cmd_setup(args: argparse.Namespace) -> int:
    root = repo_root()
    c = Client()
    reviewed = load_reviewed(root)
    m = load_manifest(root)
    p = c.project()
    created = False
    if p is None:
        p = c.post("/api/projects", {"title": PROJECT_TITLE,
                                     "description": "Nhãn người tập test XLA (phase 08). Sửa hộp nếu cần rồi Submit.",
                                     "label_config": (root / CONFIG_REL).read_text(encoding="utf-8"),
                                     "show_collab_predictions": True, "maximum_annotations": 1})
        created = True
    pid = p["id"]
    frames_dir = str((root / DOC_ROOT_REL / FRAMES_SUB).resolve())
    storages = c.get("/api/storages/localfiles", params={"project": pid})
    if not any(Path(s["path"]).resolve() == Path(frames_dir) for s in storages):
        c.post("/api/storages/localfiles", {"project": pid, "title": "frames", "path": frames_dir,
                                            "use_blob_urls": True, "regex_filter": r".*\.jpg$"})
    have = {task_frame(t) for t in c.tasks(pid)}
    todo = [f for f in m["test_frames"] if f not in have]
    tasks = []
    for f in todo:
        if f not in reviewed:
            raise SystemExit(f"{f}: chưa có nhãn agent soát ({REVIEWED_JSON_REL})")
        boxes = [{"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]} for b in reviewed[f]["boxes"]]
        result = to_ls_result(boxes)
        if reviewed[f]["hard"]:
            result.append({"id": "hard", "from_name": "flags", "to_name": "image", "type": "choices",
                           "value": {"choices": ["hard"]}})
        tasks.append({"data": {"image": f"/data/local-files/?d={FRAMES_SUB}/{f}", "frame": f},
                      "predictions": [{"model_version": "agent-reviewed-v1", "score": 1.0, "result": result}]})
    for i in range(0, len(tasks), 100):
        c.post(f"/api/projects/{pid}/import", tasks[i:i + 100])
    n = len(c.tasks(pid))
    out = {"project_id": pid, "created": created, "imported": len(tasks), "tasks": n,
           "url": f"{c.url.replace('127.0.0.1', 'localhost')}/projects/{pid}/data"}
    print(json.dumps(out, ensure_ascii=False) if args.json else out)
    return 0 if n == len(m["test_frames"]) else 1


def _collect(c: Client) -> tuple[int, dict[str, dict[str, Any] | None]]:
    p = c.project()
    if p is None:
        raise SystemExit(f"chưa có project {PROJECT_TITLE}: chạy `setup` trước")
    return p["id"], {task_frame(t): submitted(t) for t in c.tasks(p["id"])}


def cmd_status(args: argparse.Namespace) -> int:
    root = repo_root()
    m = load_manifest(root)
    pid, by_frame = _collect(Client())
    missing = sorted(set(m["test_frames"]) - set(by_frame))
    done = sorted(f for f in m["test_frames"] if by_frame.get(f))
    out = {"project_id": pid, "test_frames": len(m["test_frames"]), "submitted": len(done),
           "pending": len(m["test_frames"]) - len(done), "missing_tasks": len(missing)}
    print(json.dumps(out, ensure_ascii=False) if args.json else out)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    root = repo_root()
    m = load_manifest(root)
    pid, by_frame = _collect(Client())
    pending = [f for f in m["test_frames"] if not by_frame.get(f)]
    if pending:
        out = {"project_id": pid, "pending": len(pending), "first_pending": pending[:10], "written": 0}
        print(json.dumps(out, ensure_ascii=False) if args.json else out)
        print(f"LỖI: còn {len(pending)} khung test chưa Submit", file=sys.stderr)
        return 1
    reviewed = load_reviewed(root)
    # hard_frames = đúng các khung chủ dự án tick hard (agent đã tick sẵn trong predictions; bỏ tick = bỏ cờ).
    # Không gộp cờ của agent (kể cả dev-val): cờ agent giữ trong results/labels/agent_reviewed.json để tham khảo.
    # Owner quyết 18/11/2026 sau khi hỏi chủ dự án: chủ dự án cố ý bỏ phần lớn cờ của agent.
    hard: set[str] = set()
    labels, edits = {}, {"deleted": 0, "added": 0, "modified": 0}
    for f in m["test_frames"]:
        boxes, is_hard = ann_boxes(by_frame[f])
        boxes = [b for b in boxes if b[2] - b[0] > 0 and b[3] - b[1] > 0]
        labels[f] = boxes
        if is_hard:
            hard.add(f)
        d = diff_boxes(reviewed[f]["boxes"] if f in reviewed else [], boxes)
        for k in edits:
            edits[k] += d[k]
    # file cũ của lần xuất trước (nếu có) bị ghi đè; khung không người = file rỗng
    write_labels(root / LABELS_REL / "test", labels)
    agent_total = sum(len(v["boxes"]) for f, v in reviewed.items() if v["split"] == "test")
    m["approved_by_owner"] = True
    m["approved_date"] = date.today().isoformat()
    m["hard_frames"] = sorted(hard)
    m["n_boxes"]["test"] = sum(len(v) for v in labels.values())
    m["owner_edits"] = sum(edits.values())
    save_manifest(m, root)
    out = {"project_id": pid, "written": len(labels), "n_boxes_test": m["n_boxes"]["test"], "owner_edits": edits,
           "owner_edits_total": m["owner_edits"], "agent_boxes_test": agent_total,
           "owner_edit_ratio": round(m["owner_edits"] / agent_total, 4) if agent_total else None,
           "agent_reviewed_dir": str(AGENT_TEST_REL)}
    print(json.dumps(out, ensure_ascii=False) if args.json else out)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m xla.detect.labelstudio", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, h in (("setup", "tạo project + nạp khung test"), ("status", "số khung đã/chưa Submit"),
                    ("export", "xuất nhãn đã duyệt → YOLO txt + manifest")):
        p = sub.add_parser(name, help=h)
        p.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    return {"setup": cmd_setup, "status": cmd_status, "export": cmd_export}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
