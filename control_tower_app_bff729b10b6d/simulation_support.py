"""Simulation cells shared by the browser and Jupyter notebook.

Custom code is trusted local Python, NOT sandboxed. Execute only code you trust.
The subprocess separates variables and imposes a timeout; it does not restrict
filesystem/network access or undo side effects of custom code.
"""
from pathlib import Path
import ast
import json
import subprocess
import sys

PRESETS = {
    'nonlinear': 'Nonlinear regional growth',
    'linear': 'Linear regional growth',
    'custom': 'Custom Python cell',
}
FEATURES = ['unemployment', 'credit_spread', 'inflation',
            'investment_growth', 'confidence', 'policy_rate']


def simulation_code(kind='nonlinear', rows=800, seed=422):
    """Return a standalone NumPy/pandas cell, ready to paste into Jupyter."""
    if kind not in ('linear', 'nonlinear'):
        raise ValueError('Choose linear or nonlinear for a template.')
    if type(rows) is not int or not 80 <= rows <= 20000:
        raise ValueError('Simulated observations must be an integer from 80 to 20000.')
    if type(seed) is not int or not 0 <= seed <= 999999:
        raise ValueError('Seed must be an integer from 0 to 999999.')
    equation = (
        'growth = (3.5 - 0.25 * unemployment + 0.4 * investment\n'
        '          - 0.18 * (inflation - 2)**2\n'
        '          - 2.5 * ((unemployment > 6.5) & (spread > 2.3))\n'
        '          + 0.015 * (confidence - 80) + rng.normal(0, 0.65, N))'
        if kind == 'nonlinear' else
        'growth = (3.5 - 0.25 * unemployment - 0.45 * spread\n'
        '          + 0.4 * investment - 0.15 * inflation\n'
        '          + 0.015 * (confidence - 80) + rng.normal(0, 0.65, N))'
    )
    return f'''# Standalone simulation cell: copy this entire cell into Jupyter.
# Independent simulated regions, not a historical time series.
import numpy as np
import pandas as pd

N = {rows}
SEED = {seed}
FEATURES = {FEATURES!r}
TARGET = "growth_next_year"
DATASET_LABEL = "SIMULATED: {PRESETS[kind]}"
TARGET_UNITS = "percentage points"

rng = np.random.default_rng(SEED)
unemployment = rng.uniform(3, 11, N)
spread = rng.uniform(0.3, 4.5, N)
inflation = rng.normal(2.5, 1.2, N)
investment = rng.normal(3, 2, N)
confidence = 100 - 2 * unemployment - 3 * spread + rng.normal(0, 5, N)
policy_rate = 1 + 0.65 * inflation + rng.normal(0, 0.7, N)
{equation}

df = pd.DataFrame(dict(
    unemployment=unemployment, credit_spread=spread,
    inflation=inflation, investment_growth=investment,
    confidence=confidence, policy_rate=policy_rate,
    growth_next_year=growth,
))
# Inject missing predictors; imputation stays inside the CV pipeline.
for column in ["confidence", "investment_growth"]:
    df.loc[rng.choice(N, size=max(1, N // 40), replace=False), column] = np.nan

# Required outputs: df, FEATURES, TARGET, DATASET_LABEL, TARGET_UNITS.
# Optional: df.to_csv("my_simulated_data.csv", index=False)
'''


def validate_code(code):
    if not isinstance(code, str) or not code.strip():
        raise ValueError('Paste a Python cell that creates df and its column definitions.')
    if len(code.encode('utf-8')) > 60000:
        raise ValueError('Keep the simulation code below 60 KB.')
    try:
        ast.parse(code, filename='simulation_code.py')
    except SyntaxError as exc:
        raise ValueError(f'Python syntax error on line {exc.lineno}: {exc.msg}. '
                         'Paste plain Python without Markdown fences or %/! notebook commands.') from exc
    return code


def validate_outputs(namespace):
    import numpy as np
    import pandas as pd
    required = ['df', 'FEATURES', 'TARGET', 'DATASET_LABEL', 'TARGET_UNITS']
    missing = [name for name in required if name not in namespace]
    if missing:
        raise ValueError('Simulation must define: ' + ', '.join(missing))
    df = namespace['df']
    if not isinstance(df, pd.DataFrame):
        raise ValueError('df must be a pandas DataFrame.')
    if not 80 <= len(df) <= 20000:
        raise ValueError('The simulation must produce between 80 and 20000 rows.')
    features, target = namespace['FEATURES'], namespace['TARGET']
    if (not isinstance(features, (list, tuple)) or not 1 <= len(features) <= 100
            or not all(isinstance(c, str) and c for c in features)
            or len(set(features)) != len(features)):
        raise ValueError('FEATURES must list 1 to 100 distinct column names.')
    if not isinstance(target, str) or not target or target in features:
        raise ValueError('TARGET must be a column name excluded from FEATURES.')
    if not df.columns.is_unique:
        raise ValueError('df must have unique column names.')
    absent = [c for c in [*features, target] if c not in df]
    if absent:
        raise ValueError('Missing columns in df: ' + ', '.join(absent))
    frame = df.loc[:, [*features, target]].apply(pd.to_numeric, errors='raise').astype(float)
    if not np.isfinite(frame[target]).all() or frame[target].nunique() < 2:
        raise ValueError('The target must be finite, nonmissing, and nonconstant.')
    if np.isinf(frame[list(features)].to_numpy()).any() or frame[list(features)].isna().all().any():
        raise ValueError('Predictors cannot contain infinity or an entirely missing column.')
    label, units = namespace['DATASET_LABEL'], namespace['TARGET_UNITS']
    if not all(isinstance(v, str) and 1 <= len(v.strip()) <= 200 for v in [label, units]):
        raise ValueError('DATASET_LABEL and TARGET_UNITS must be nonempty text (at most 200 characters).')
    label = label.strip()
    if not label.upper().startswith('SIMULATED'):
        label = 'SIMULATED: ' + label
    metadata = dict(features=list(features), target=target, dataset_label=label,
                    target_units=units.strip(), rows=len(frame))
    return frame.reset_index(drop=True), metadata


def run_simulation(code, output_dir, timeout=60):
    """Run an explicitly selected trusted cell and retain code, data, and log."""
    import pandas as pd
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = out / 'simulation_code.py'
    source.write_text(validate_code(code), encoding='utf-8')
    command = [sys.executable, str(Path(__file__).resolve()), str(source), str(out)]
    with (out / 'simulation.log').open('w', encoding='utf-8') as log:
        try:
            done = subprocess.run(command, cwd=out, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            raise ValueError(f'Simulation exceeded {timeout} seconds. Simplify the data-generation cell.') from exc
    if done.returncode:
        error = (out / 'simulation_error.txt')
        detail = error.read_text(encoding='utf-8') if error.exists() else 'See simulation/simulation.log for details.'
        raise ValueError('Simulation failed: ' + detail[-2000:])
    frame = pd.read_csv(out / 'input_data.csv', float_precision='round_trip')
    meta = json.loads((out / 'simulation_metadata.json').read_text(encoding='utf-8'))
    # Validate persisted outputs again before they reach the analyst.
    return validate_outputs(dict(df=frame, FEATURES=meta['features'], TARGET=meta['target'],
                                 DATASET_LABEL=meta['dataset_label'], TARGET_UNITS=meta['target_units']))


if __name__ == '__main__':
    import traceback
    source, output = map(Path, sys.argv[1:])
    try:
        scope = {'__name__': '__main__', '__file__': str(source)}
        exec(compile(source.read_text(encoding='utf-8'), str(source), 'exec'), scope)
        df, meta = validate_outputs(scope)
        df.to_csv(output / 'input_data.csv', index=False)
        (output / 'simulation_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    except Exception as exc:
        (output / 'simulation_error.txt').write_text(f'{type(exc).__name__}: {exc}', encoding='utf-8')
        traceback.print_exc()
        sys.exit(1)
