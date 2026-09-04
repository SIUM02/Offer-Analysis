import os
import sys
from http.server import BaseHTTPRequestHandler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from user_test import load_catalogue, reference_rates      # noqa: E402
from web import route                                      # noqa: E402

CATALOGUE = load_catalogue()
RATES = reference_rates(CATALOGUE)
OPERATORS = sorted({offer["operator"] for offer in CATALOGUE})


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path, _, query = self.path.partition("?")
        status, kind, body = route(path, query, CATALOGUE, RATES, OPERATORS)
        self.send_response(status)
        if body:
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)
