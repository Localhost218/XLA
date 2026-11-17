import sys, importlib.util
spec = importlib.util.spec_from_file_location("scan", "jpeg_dqt_scan.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
LUMA = [16,11,10,16,24,40,51,61,12,12,14,19,26,58,60,55,14,13,16,24,40,57,69,56,14,17,22,29,51,87,80,62,18,22,37,56,68,109,103,77,24,35,55,64,81,104,113,92,49,64,78,87,103,121,120,101,72,92,95,98,112,100,103,99]
CHROMA = [17,18,24,47,99,99,99,99,18,21,26,66,99,99,99,99,24,26,56,99,99,99,99,99,47,66,99,99,99,99,99,99] + [99]*32
def ijg(tbl, q):
    s = 5000 // q if q < 50 else 200 - 2 * q
    return [min(255, max(1, (v * s + 50) // 100)) for v in tbl]
def ijg_scale(tbl, s):
    return [min(255, max(1, (v * s + 50) // 100)) for v in tbl]
fr = next(m.frames_from_avi(sys.argv[1]))
p = m.parse_jpeg(fr); L, C = p["dqt"][0], p["dqt"][1]
print("scale75 luma match:", L == ijg_scale(LUMA, 75), " chroma match:", C == ijg_scale(CHROMA, 75))
for q in (60, 61, 62, 63, 64, 81):
    dl = sum(a != b for a, b in zip(L, ijg(LUMA, q))); dc = sum(a != b for a, b in zip(C, ijg(CHROMA, q)))
    print(f"IJG q={q}: luma mismatches={dl}/64 chroma mismatches={dc}/64")
