"""Local browser control tower. Run: python dashboard.py (no web dependencies)."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread, Lock
import argparse
import json
import io
import csv
import secrets
import time
import webbrowser
from simulation_support import simulation_code, validate_code, run_simulation, PRESETS

ROOT = Path(__file__).resolve().parent
TOKEN = secrets.token_urlsafe(32)
LOCK = Lock()
STATE = dict(status='idle', progress=0, message='Your analysts are ready.', events=[])
REPORT = None
METHODS = ['Ridge', 'Lasso', 'Elastic Net', 'CART', 'Random Forest', 'Gradient Boosting']


def update(**values):
    with LOCK:
        STATE.update(values)


def read_upload(c):
    import pandas as pd
    from simulation_support import validate_outputs
    raw = c.get('csv_text')
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError('Choose a CSV file.')
    header = next(csv.reader(io.StringIO(raw)), [])
    if not header or len(set(header)) != len(header):
        raise ValueError('CSV column names must be unique.')
    df = pd.read_csv(io.StringIO(raw))
    label = c.get('dataset_label', 'Uploaded data')
    frame, meta = validate_outputs(dict(df=df, FEATURES=c.get('features'), TARGET=c.get('target'),
        DATASET_LABEL=label, TARGET_UNITS=c.get('target_units')))
    meta['dataset_label'] = label.strip()
    return frame, meta


def work(config):
    global REPORT
    try:
        # Imports happen in the worker so the browser opens immediately.
        from ml_analyst import MLAnalyst, AnalysisConfig, prepare_data
        from model_evaluation_analyst import ModelEvaluationAnalyst
        run = ROOT / 'outputs' / ('browser_' + time.strftime('%Y%m%d_%H%M%S') + '_' + secrets.token_hex(3))
        run.mkdir(parents=True)
        (run / 'user_configuration.json').write_text(json.dumps(config, indent=2))
        if config['source'] == 'upload':
            update(progress=5, message='Reading your uploaded data.')
            df, meta = read_upload(config)
        else:
            update(progress=5, message='Generating data from your chosen DGP.')
            code = (config['simulation_code'] if config['simulation'] == 'custom' else
                    simulation_code(config['simulation'], config['rows'], config['seed']))
            df, meta = run_simulation(code, run / 'simulation')
        cfg = AnalysisConfig(methods=tuple(config['methods']), folds=config['folds'],
                             seed=config['seed'], target=meta['target'],
                             dataset_label=meta['dataset_label'], target_units=meta['target_units'])
        with LOCK:
            STATE['simulation'] = meta
        df.to_csv(run / 'input_data.csv', index=False)
        Xe, Xt, ye, yt = prepare_data(df, meta['features'], cfg)
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
        result = ModelEvaluationAnalyst(seed=cfg.seed, max_explain=config['explain_rows'], top_k=config.get('top_k', 5), progress=progress).run(path, Xt, yt, run / 'evaluation')
        with LOCK:
            REPORT = result['report']
            STATE.update(status='done', progress=100, message='Your report is ready!', selected=result['selection']['selected_model'], metrics=result['evidence']['metrics'], output_path=str(run))
        (run / 'dashboard_events.json').write_text(json.dumps(STATE['events'], indent=2))
    except Exception as exc:
        update(status='error', message=f'{type(exc).__name__}: {exc}')


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, data, kind='application/json', status=200, download=False, filename='Analyst_Report.html'):
        data = data.encode() if isinstance(data, str) else data
        self.send_response(status)
        self.send_header('Content-Type', kind + '; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        if download:
            self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
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
        if self.path == '/simulation-download':
            with LOCK:
                run_path = STATE.get('output_path')
            if run_path:
                code = Path(run_path) / 'simulation' / 'simulation_code.py'
                if not code.exists():
                    return self.send('{}', status=404)
                return self.send(code.read_bytes(), 'text/x-python', download=True,
                                 filename='simulation_code.py')
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
        if self.path not in ('/run', '/simulation-code'):
            return self.send('{}', status=404)
        try:
            size = int(self.headers.get('Content-Length', 0))
            if size < 1 or size > 12000000:
                raise ValueError('Invalid request size')
            c = json.loads(self.rfile.read(size))
            if not isinstance(c, dict):
                raise ValueError('Configuration must be a JSON object.')
            if self.path == '/simulation-code':
                code = simulation_code(c.get('simulation', 'nonlinear'), c['rows'], c['seed'])
                return self.send(json.dumps({'code': code}))
            c.setdefault('source', 'simulation')
            if c['source'] not in ('simulation', 'upload'):
                raise ValueError('Choose upload or simulation.')
            if c['source'] == 'upload':
                if c.get('independent') is not True:
                    raise ValueError('Confirm independent rows for random splitting.')
                read_upload(c)
            else:
                c.setdefault('simulation', 'nonlinear')
                if c['simulation'] not in PRESETS:
                    raise ValueError('Choose a supported simulation option.')
                if c['simulation'] == 'custom':
                    c['simulation_code'] = validate_code(c.get('simulation_code'))
                elif type(c.get('rows')) is not int or not 80 <= c['rows'] <= 20000:
                    raise ValueError('Simulated observations must be from 80 to 20000.')
            c.setdefault('top_k', 5)
            if type(c['top_k']) is not int or not 1 <= c['top_k'] <= 100:
                raise ValueError('Top K must be an integer from 1 to 100.')
            names = c['methods']
            if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names) or len(set(names)) != len(names) or any(n not in METHODS for n in names):
                raise ValueError('Select at least one supported ML method.')
            for key, allowed in [('folds', [3, 5]), ('explain_rows', [30, 60, 120])]:
                if type(c[key]) is not int or c[key] not in allowed:
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
