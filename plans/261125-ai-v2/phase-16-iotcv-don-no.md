# Phase 16 — IOT-CV: xoá `ml/scripts/` sau grep tham chiếu (lane iotcv-debt)

Lane: iotcv-debt
Depends-on: 01
Effort: S
Repo: IOT-CV, worktree `D:\IOTCV-wt\iotcv-debt`, nhánh `lane/iotcv-debt` từ `feat/ai-v2` IOT-CV (phase 01 đã tạo)
Owns: `ml/scripts/**`, `ml/README.md` (nếu có), `docs/bao-cao-tong-quan-du-an.md` (chỉ dòng 383–385), `backend/tests/test_repo_hygiene.py`
Creates: `backend/tests/test_repo_hygiene.py`
AX: n/a because phase deletes files and adds one test
Accept: `pytest backend/tests/test_repo_hygiene.py`

## Mục tiêu

Trả nợ đã duyệt trong thiết kế (brainstorm §3 "Nợ nhỏ"): xoá 6 khung sườn `ml/scripts/*.py` (`augment`, `convert_visdrone_person`,
`evaluate`, `extract_frames`, `split_dataset`, `train`; mỗi file thân chỉ `NotImplementedError`) sau khi grep tham chiếu.

## Đầu vào

Kết quả rà 25/11: tham chiếu ngoài `ml/` chỉ có trong tài liệu — `docs/archive/plan-nhap-92-giai-doan.md`,
`docs/archive/team-cu/P4-vision-ml.md`, `docs/bao-cao-tong-quan-du-an.md:383,385`, `plans/PROGRESS.md`, `plans/ai/ai-phase-0{1..4}-*.md`,
`plans/reports/260921-research-ai-vision-pipeline.md`; zero trong code, CI, config (`.github/workflows`, `infra`, `pyproject.toml`,
`backend`, `frontend`, `scripts`). `.github/workflows/iotcv-deploy.yml:11` lọc `ml/**` (deploy chạy khi merge, không lỗi).

## Bước

1. Grep lại (`grep -rnE "ml/scripts|augment\.py|convert_visdrone_person|evaluate\.py|extract_frames|split_dataset|scripts/train\.py"`
   loại `ml, node_modules, .git, .venv, .agents, .claude, __pycache__, dist, logs, .ruff_cache`); hit mới trong code → dừng, báo.
2. `git rm` sáu file (pathspec commit). Giữ `ml/configs/` (ngoài phạm vi đã duyệt).
3. Tài liệu lịch sử (`docs/archive/**`, `plans/ai/**`, `plans/reports/260921-*`, `plans/PROGRESS.md`) giữ nguyên như bản ghi.
   Tài liệu sống `docs/bao-cao-tong-quan-du-an.md:383-385`: sửa thành trỏ tới XLA (`D:\XLA` / Localhost218/XLA). `ml/README.md`
   (nếu có) ghi "đã chuyển sang XLA, xoá 25/11/2026".
4. `test_repo_hygiene.py`: `ml/scripts` không tồn tại; không file nào trong `backend/`, `frontend/src`, `scripts/`, `.github/`,
   `infra/`, `pyproject.toml` nhắc `ml/scripts`.
5. Đẩy `lane/iotcv-debt`, báo Owner (gộp ở phase 15).

## Tiêu chí xong

Accept xanh; `backend/tests` toàn bộ xanh.

## Risk

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Một script ngoài repo gọi file đã xoá | low | low | Thân file chỉ `NotImplementedError`, không thể đang được dùng; git giữ lịch sử |
| Merge kích hoạt deploy vì `paths: ml/**` | high | low | Deploy chạy cùng PR phase 15 vốn đã đổi backend; ghi trong mô tả PR |

Rollback: `git revert` commit xoá.
