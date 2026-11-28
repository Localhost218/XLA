"""Phục vụ đúng một file trên 127.0.0.1 cho trang Drive đọc (CORS + Private Network Access). Tự tắt sau 15 phút."""
import http.server, sys, threading
PATH, PORT = sys.argv[1], int(sys.argv[2])
ORIGIN = "https://drive.google.com"
class H(http.server.BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", ORIGIN)
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()
    def do_GET(self):
        if self.path != "/deck.pptx":
            self.send_response(404); self.end_headers(); return
        data = open(PATH, "rb").read()
        self.send_response(200); self._cors()
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data))); self.end_headers()
        self.wfile.write(data)
srv = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), H)
threading.Timer(900, srv.shutdown).start()
print("serving", flush=True)
srv.serve_forever()
