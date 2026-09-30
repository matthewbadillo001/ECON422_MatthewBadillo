"""Local browser control tower. Run: python local_dashboard.py (no web dependencies)."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread, Lock
import argparse
import json
import secrets
import time
import webbrowser

ROOT = Path(__file__).resolve().parent
TOKEN = secrets.token_urlsafe(32)
LOCK = Lock()
STATE = dict(status='idle', progress=0, message='Your analysts are ready.', events=[])
REPORT = None
METHODS = ['Ridge', 'Lasso', 'Elastic Net', 'CART', 'Random Forest', 'Gradient Boosting']


def update(**values):
    with LOCK:
        STATE.update(values)


def work(config):
    global REPORT
    try:
        # Imports happen in the worker so the browser opens immediately.
        from ml_analyst import MLAnalyst, AnalysisConfig, make_demo_data, prepare_data
        from model_evaluation_analyst import ModelEvaluationAnalyst
        run = ROOT / 'outputs' / ('browser_' + time.strftime('%Y%m%d_%H%M%S') + '_' + secrets.token_hex(3))
        run.mkdir(parents=True)
        (run / 'user_configuration.json').write_text(json.dumps(config, indent=2))
        cfg = AnalysisConfig(methods=tuple(config['methods']), folds=config['folds'], seed=config['seed'])
        df = make_demo_data(n=config['rows'], seed=cfg.seed)
        df.to_csv(run / 'input_data.csv', index=False)
        Xe, Xt, ye, yt = prepare_data(df, [c for c in df if c != cfg.target], cfg)
        completed = 0

        def progress(role, stage, message):
            nonlocal completed
            if role == 'ML Analyst':
                if 'completed; CV MSE' in message:
                    completed += 1
                pct = 10 + round(55 * completed / len(config['methods']))
            elif 'Computing SHAP' in message:
                pct = 78
            elif 'drawing figures' in message:
                pct = 87
            elif 'Writing the report' in message:
                pct = 95
            else:
                pct = 70 if stage != 'Completed' else 98
            with LOCK:
                STATE.update(progress=pct, message=message, role=role)
                STATE['events'].append(dict(time=time.strftime('%H:%M:%S'), role=role, stage=stage, message=message))
        update(progress=10, message='Estimation and testing samples prepared.')
        path = MLAnalyst(cfg, progress=progress).run(Xe, ye, run / 'analysis')
        result = ModelEvaluationAnalyst(max_explain=config['explain_rows'], progress=progress).run(path, Xt, yt, run / 'evaluation')
        with LOCK:
            REPORT = result['report']
            STATE.update(status='done', progress=100, message='Your report is ready!', selected=result['selection']['selected_model'], metrics=result['evidence']['metrics'], output_path=str(run))
        (run / 'dashboard_events.json').write_text(json.dumps(STATE['events'], indent=2))
    except Exception as exc:
        update(status='error', message=f'{type(exc).__name__}: {exc}')


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, data, kind='application/json', status=200, download=False):
        data = data.encode() if isinstance(data, str) else data
        self.send_response(status)
        self.send_header('Content-Type', kind + '; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        if download:
            self.send_header('Content-Disposition', 'attachment; filename="Analyst_Report.html"')
        self.end_headers()
        self.wfile.write(data)

    def authorized(self):
        return self.headers.get('Host') == f'127.0.0.1:{self.server.server_port}'

    def do_GET(self):
        if not self.authorized():
            return self.send('{}', status=403)
        if self.path == '/':
            return self.send((ROOT / 'dashboard.html').read_text().replace('__TOKEN__', TOKEN), 'text/html')
        if self.path == '/status':
            with LOCK:
                data = json.dumps(STATE)
            return self.send(data)
        if self.path in ('/report', '/download'):
            with LOCK:
                report = REPORT
            if report:
                return self.send(report.read_bytes(), 'text/html', download=self.path == '/download')
        self.send('{}', status=404)

    def do_POST(self):
        global REPORT
        if not self.authorized() or self.headers.get('X-Local-Token') != TOKEN:
            return self.send('{}', status=403)
        if self.path != '/run':
            return self.send('{}', status=404)
        try:
            size = int(self.headers.get('Content-Length', 0))
            if size < 1 or size > 8192:
                raise ValueError('Invalid request size')
            c = json.loads(self.rfile.read(size))
            names = c['methods']
            if not isinstance(names, list) or not names or len(set(names)) != len(names) or any(n not in METHODS for n in names):
                raise ValueError('Select at least one supported ML method.')
            for key, allowed in [('rows', [200, 400, 800]), ('folds', [3, 5]), ('explain_rows', [30, 60, 120])]:
                if c[key] not in allowed:
                    raise ValueError('Invalid ' + key)
            if type(c['seed']) is not int or not 0 <= c['seed'] <= 999999:
                raise ValueError('Seed must be an integer from 0 to 999999.')
            if c.get('explanation') != 'SHAP':
                raise ValueError('Select SHAP for the Model Evaluation Analyst.')
            with LOCK:
                if STATE['status'] == 'running':
                    return self.send(json.dumps({'error': 'An analysis is already running.'}), status=409)
                REPORT = None
                STATE.clear()
                STATE.update(status='running', progress=2, message='Preparing your analysts…', events=[], configuration=c)
            Thread(target=work, args=(c,), daemon=True).start()
            self.send('{"started":true}')
        except (ValueError, KeyError, TypeError) as exc:
            self.send(json.dumps({'error': str(exc)}), status=400)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'Open {url}\nPress Ctrl+C to stop. Runs and reports are saved in {ROOT / "outputs"}', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
