import os
import sys
from http.server import BaseHTTPRequestHandler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from user_test import load_catalogue, reference_rates
from web import render_page

CATALOGUE = load_catalogue()
RATES = reference_rates(CATALOGUE)
OPERATORS = sorted({offer["operator"] for offer in CATALOGUE})


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path, _, query = self.path.partition("?")

        if path.endswith("/favicon.ico"):
            self.send_response(204)
            self.end_headers()
            return

        body = render_page(CATALOGUE, RATES, OPERATORS, query)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
