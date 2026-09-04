"""Vercel entry point: the same page, one invocation per request.

Vercel runs Python as serverless functions, so there is no server to start
and nothing to keep running -- it imports this file and calls `handler` for
each request. That suits the recommender: it holds no state between
requests, and the whole catalogue is 203 rows.

Everything below the HTTP layer is the code already in scripts/: the page
comes from web.render_page, the answers from user_test. Nothing about the
matching algorithm is duplicated here.

The catalogue is read once per cold start, not once per request, so a warm
invocation does no file I/O at all.

Deploying (from the project root, with the Vercel CLI):

    npx vercel        # preview
    npx vercel --prod # live

vercel.json rewrites every path here and tells the build to ship data/ and
scripts/ alongside this file -- without that, the function would start with
no catalogue to read.
"""

import os
import sys
from http.server import BaseHTTPRequestHandler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from user_test import load_catalogue, reference_rates      # noqa: E402
from web import render_page                                # noqa: E402

CATALOGUE = load_catalogue()
RATES = reference_rates(CATALOGUE)
OPERATORS = sorted({offer["operator"] for offer in CATALOGUE})


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path, _, query = self.path.partition("?")

        # Every path lands here because of the rewrite, so the page is served
        # whatever the URL says -- except the icon the browser asks for
        # unprompted, which would otherwise render the whole page again.
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
