"""Two XGBoost DA forecasts and a training-only hypothetical quote benchmark.
Run: python da_prediction.py --demo --output prediction_demo
Real data: set ENTSOE_API_TOKEN; python da_prediction.py --output prediction_output
Dependencies: numpy pandas requests entsoe-py xgboost scikit-learn plotly
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, asdict
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

WEATHER_COLS = ['temperature_2m', 'wind_speed_10m', 'wind_speed_100m',
                'cloud_cover', 'direct_radiation', 'diffuse_radiation']
CALENDAR_COLS = ['hour', 'minute', 'dayofweek', 'month']
FEATURES = {'weather': WEATHER_COLS + CALENDAR_COLS,
            'fundamentals': ['generation_forecast_mw', 'residual_load_forecast_mw'] + CALENDAR_COLS}
PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
              objective='reg:squarederror', random_state=42, n_jobs=2)

@dataclass(frozen=True)
class PredictionConfig:
    zone: str = 'DK_2'
    timezone: str = 'Europe/Copenhagen'
    start_date: str = '2026-06-01'
    end_date: str = '2026-10-01'  # exclusive
    latitude: float = 55.6759
    longitude: float = 12.5655
    weather_model: str = 'dmi_seamless'
    train_fraction: float = 0.8
    high_quantile: float = 0.9
    fixed_high_price: float = 200.0
    decision_hour_local: int = 10  # assumed previous-day decision time, not exchange specification


def validate_config(cfg):
    if not 0 < cfg.train_fraction < 1 or not 0 < cfg.high_quantile < 1:
        raise ValueError('Fractions must be strictly between zero and one.')
    if not 0 <= cfg.decision_hour_local <= 23:
        raise ValueError('decision_hour_local must be 0..23.')
    if not -90 <= cfg.latitude <= 90 or not -180 <= cfg.longitude <= 180:
        raise ValueError('Invalid coordinates.')
    start = pd.Timestamp(cfg.start_date, tz=cfg.timezone)
    end = pd.Timestamp(cfg.end_date, tz=cfg.timezone)
    if end <= start:
        raise ValueError('end_date must be after start_date.')
    return start, end

def as_utc(obj):
    obj = obj.copy()
    if not isinstance(obj.index, pd.DatetimeIndex) or obj.index.tz is None:
        raise ValueError('Expected a timezone-aware DatetimeIndex.')
    obj.index = obj.index.tz_convert('UTC')
    obj = obj.sort_index()
    if obj.index.has_duplicates:
        raise ValueError('Duplicate timestamps: inspect source revisions before modelling.')
    return obj.apply(pd.to_numeric, errors='coerce') if isinstance(obj, pd.DataFrame) else pd.to_numeric(obj, errors='coerce')

def to_quarter_hours(obj, target_grid):
    obj = as_utc(obj)
    differences = obj.index.to_series().diff().dropna()
    if differences.empty:
        raise ValueError('Not enough observations to infer source interval.')
    interval = differences.mode().iloc[0]
    if interval not in [pd.Timedelta('15min'), pd.Timedelta('30min'), pd.Timedelta('1h')]:
        raise ValueError(f'Unsupported source interval: {interval}')
    # Regularise first so an absent source record cannot inherit an earlier value.
    regular = pd.date_range(obj.index.min(), obj.index.max(), freq=interval)
    if not obj.index.isin(regular).all():
        raise ValueError('Mixed/off-grid intervals; inspect and normalise the source first.')
    obj = obj.reindex(regular)
    return obj.reindex(target_grid, method='ffill',
                       tolerance=interval-pd.Timedelta('1ns'))

def select_series(obj, preferred, name):
    if isinstance(obj, pd.Series):
        return obj.rename(name)
    if preferred in obj.columns:
        return obj[preferred].rename(name)
    if obj.shape[1] == 1:
        return obj.iloc[:, 0].rename(name)
    raise ValueError(f'Cannot identify {name}; available columns: {list(obj.columns)}')

def fetch_data(cfg, token=None):
    """Fetch retrospective data. These endpoints do NOT certify pre-auction vintages."""
    import requests
    from entsoe import EntsoePandasClient
    start, end = validate_config(cfg)
    token = token or os.environ.get('ENTSOE_API_TOKEN', '').strip()
    key_file = Path.cwd().parent / 'entsoe-api.txt'
    if not token and key_file.exists():
        token = key_file.read_text(encoding='utf-8').strip()
    if not token:
        raise RuntimeError('Set ENTSO-E token using environment variable ENTSOE_API_TOKEN or ../entsoe-api.txt.')
    client = EntsoePandasClient(api_key=token)
    start_utc, end_utc = start.tz_convert('UTC'), end.tz_convert('UTC')
    grid = pd.date_range(start_utc, end_utc, freq='15min', inclusive='left')
    response = requests.get('https://historical-forecast-api.open-meteo.com/v1/forecast',
        params={'latitude': cfg.latitude, 'longitude': cfg.longitude,
                'start_date': start_utc.date().isoformat(),
                'end_date': (end_utc-pd.Timedelta('1ns')).date().isoformat(),
                'hourly': ','.join(WEATHER_COLS), 'models': cfg.weather_model,
                'wind_speed_unit': 'ms', 'timezone': 'UTC'}, timeout=120)
    response.raise_for_status()
    payload = response.json()
    if 'hourly' not in payload:
        raise ValueError('No hourly weather data returned; check model and date coverage.')
    weather = pd.DataFrame(payload['hourly'])
    weather.index = pd.to_datetime(weather.pop('time'), utc=True)
    weather = to_quarter_hours(weather[WEATHER_COLS], grid)
    prices = as_utc(client.query_day_ahead_prices(cfg.zone, start=start, end=end))
    if not isinstance(prices, pd.Series):
        raise ValueError('Expected one DA price series.')
    differences = prices.index.to_series().diff().dropna()
    if differences.empty or differences.mode().iloc[0] != pd.Timedelta('15min'):
        raise ValueError('15-minute target prices are required; prices are never forward-filled.')
    generation = select_series(client.query_generation_forecast(cfg.zone, start=start, end=end),
                               'Actual Aggregated', 'generation_forecast_mw')
    load = select_series(client.query_load_forecast(cfg.zone, start=start, end=end),
                         'Forecasted Load', 'load_forecast_mw')
    renewable = client.query_wind_and_solar_forecast(cfg.zone, start=start, end=end, psr_type=None)
    required = ['Solar', 'Wind Offshore', 'Wind Onshore']
    missing = set(required)-set(renewable.columns)
    if missing:
        raise ValueError(f'Missing forecast components: {missing}. Verify absence before assuming zero.')
    fundamental = pd.concat([to_quarter_hours(generation, grid), to_quarter_hours(load, grid),
                              to_quarter_hours(renewable[required], grid)], axis=1)
    fundamental['wind_solar_forecast_mw'] = fundamental[required].sum(axis=1, min_count=3)
    fundamental['residual_load_forecast_mw'] = (fundamental['load_forecast_mw']
                                               - fundamental['wind_solar_forecast_mw'])
    return pd.concat([prices.reindex(grid).rename('actual_da'), weather, fundamental], axis=1)


def demo_data(cfg, seed=42):
    """Entirely synthetic data to exercise the pipeline; not observed market history."""
    start, end = validate_config(cfg)
    index = pd.date_range(start.tz_convert('UTC'), end.tz_convert('UTC'), freq='15min', inclusive='left')
    rng = np.random.default_rng(seed)
    h = index.tz_convert(cfg.timezone).hour.to_numpy()
    days = np.arange(len(index))/96
    wind = np.maximum(0, 6 + 3*np.sin(days/3) + rng.normal(0, 0.5, len(index)))
    solar = np.maximum(0, np.sin((h-6)*np.pi/12))
    load = 1800+400*np.cos((h-18)*np.pi/12)+rng.normal(0,35,len(index))
    residual = load-150*wind-500*solar
    df = pd.DataFrame(index=index)
    df['temperature_2m'] = 15+5*np.sin(days/20)+3*solar
    df['wind_speed_10m'], df['wind_speed_100m'] = wind*.7, wind
    df['cloud_cover'] = rng.uniform(0,100,len(index))
    df['direct_radiation'], df['diffuse_radiation'] = solar*600, solar*100
    df['load_forecast_mw'] = load
    df['generation_forecast_mw'] = 1900+100*wind+200*solar
    df['residual_load_forecast_mw'] = residual
    df['actual_da'] = (25+.05*residual + .00008*np.maximum(residual-1000,0)**2
                       + rng.normal(0,15,len(index)))
    return df


def calendar_features(index, timezone):
    local = index.tz_convert(timezone)
    return pd.DataFrame({'hour':local.hour, 'minute':local.minute,
                         'dayofweek':local.dayofweek, 'month':local.month}, index=index)


def decision_times(index, cfg):
    # Subtract calendar dates, not 24 elapsed hours, so DST preserves local decision time.
    day = index.tz_convert(cfg.timezone).tz_localize(None).normalize()-pd.Timedelta(days=1)
    return (day+pd.Timedelta(hours=cfg.decision_hour_local)).tz_localize(
        cfg.timezone, ambiguous='raise', nonexistent='raise').tz_convert('UTC')


def benchmark_quotes(train, test):
    """Fixed train-only mean by weekday/hour/quarter-hour; fallback to hour then global mean."""
    keys = ['dayofweek', 'hour', 'minute']
    grouped = train.groupby(keys)['actual_da'].mean()
    lookup = pd.MultiIndex.from_frame(test[keys])
    quote = pd.Series(grouped.reindex(lookup).to_numpy(), index=test.index)
    return quote.fillna(test['hour'].map(train.groupby('hour')['actual_da'].mean())).fillna(train['actual_da'].mean())


def evaluate(predictions, threshold, fixed_threshold=200.):
    rows = []
    masks = {'overall': pd.Series(True,index=predictions.index),
             'high_train_quantile':predictions.actual_da >= threshold,
             'below_train_p90':predictions.actual_da < threshold,
             'high_fixed':predictions.actual_da >= fixed_threshold}
    for model in ['weather', 'fundamentals', 'benchmark']:
        col = 'quote_mid' if model=='benchmark' else f'pred_{model}'
        for subset, mask in masks.items():
            p = predictions.loc[mask]
            error = p[col]-p.actual_da
            rows.append(dict(model=model,subset=subset,n=len(p),
                mae=mean_absolute_error(p.actual_da,p[col]) if len(p) else np.nan,
                rmse=np.sqrt(mean_squared_error(p.actual_da,p[col])) if len(p) else np.nan,
                bias=error.mean(), underprediction_pct=100*(error<0).mean() if len(p) else np.nan,
                r2=r2_score(p.actual_da,p[col]) if len(p)>1 and p.actual_da.nunique()>1 else np.nan))
    return pd.DataFrame(rows)


def fit_evaluate(raw, cfg, synthetic=False):
    """Both models fit the first 80% only. No test-based early stopping or tuning."""
    from xgboost import XGBRegressor
    start, end = validate_config(cfg)
    raw = as_utc(raw)
    grid = pd.date_range(start.tz_convert('UTC'),end.tz_convert('UTC'),freq='15min',inclusive='left')
    if not raw.index.isin(grid).all():
        raise ValueError('Input contains off-grid/out-of-period observations.')
    raw = raw.reindex(grid)
    needed = list(dict.fromkeys(['actual_da']+WEATHER_COLS+FEATURES['fundamentals'][:-4]))
    missing = set(needed)-set(raw.columns)
    if missing:
        raise ValueError(f'Missing input columns: {missing}')
    raw = raw[needed].replace([np.inf,-np.inf],np.nan)
    missing_report = raw.isna().sum().rename('missing_intervals').to_frame()
    data = raw.join(calendar_features(raw.index,cfg.timezone)).dropna()
    cut = int(len(data)*cfg.train_fraction)
    if cut<100 or len(data)-cut<20:
        raise ValueError('Need at least 100 training rows and 20 test rows.')
    train, test = data.iloc[:cut].copy(),data.iloc[cut:].copy()
    assert train.index.max()<test.index.min()
    result = pd.DataFrame({'actual_da':test.actual_da}, index=test.index)
    models = {}
    for name, features in FEATURES.items():
        model = XGBRegressor(**PARAMS)
        model.fit(train[features],train.actual_da)
        result[f'pred_{name}'] = model.predict(test[features])
        models[name] = model
    result['quote_mid'] = benchmark_quotes(train,test)
    result['decision_time_utc'] = decision_times(test.index,cfg)
    result['duration_hours'] = .25
    # Conservative assumption: all training labels available after last training delivery.
    trained_through = train.index[-1]+pd.Timedelta('15min')
    result['eligible_for_contract'] = result.decision_time_utc > trained_through
    threshold = float(train.actual_da.quantile(cfg.high_quantile))
    result['high_price'] = result.actual_da >= threshold
    result.index.name = 'delivery_start_utc'
    metadata = dict(schema_version=1, synthetic=bool(synthetic),
        forecast_vintages_verified=False,
        purpose='Hypothetical retrospective research; not demonstrated executable market returns',
        config=asdict(cfg), model_params=PARAMS, features=FEATURES,
        quote_method='Frozen training mean by weekday/hour/quarter-hour, fallback hour/global mean',
        train_start=str(train.index[0]), train_end=str(train.index[-1]),
        training_labels_available_assumed_utc=str(trained_through),
        test_start=str(test.index[0]),test_end=str(test.index[-1]),
        n_train=len(train),n_test=len(test),n_expected=len(raw),n_dropped=len(raw)-len(data),
        high_threshold=threshold,
        contract_eligible_intervals=int(result.eligible_for_contract.sum()))
    return dict(predictions=result,metrics=evaluate(result,threshold,cfg.fixed_high_price),
                metadata=metadata, models=models, missing_report=missing_report)


def explain_metrics(metrics):
    messages = []
    for model in ['weather','fundamentals']:
        table = metrics[metrics.model==model].set_index('subset')
        overall,high = table.loc['overall'],table.loc['high_train_quantile']
        messages.append(f"{model}: overall MAE EUR {overall.mae:.2f}/MWh.")
        if high.n:
            ratio = high.mae/overall.mae if overall.mae else np.nan
            messages.append(f"  High-price MAE EUR {high.mae:.2f}/MWh ({ratio:.2f}x overall), "
                            f"n={int(high.n)}, bias EUR {high.bias:.2f}/MWh.")
        else:
            messages.append('  No high-price test observations; high-price MAE unavailable.')
    messages.append('Negative bias means underprediction. MAE is not percentage error or profit.')
    return '\n'.join(messages)


def export_result(result, output):
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    path = output/'test_predictions.csv'
    result['predictions'].to_csv(path)
    meta = dict(result['metadata'])
    meta['predictions_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    result['metrics'].to_csv(output/'prediction_metrics.csv',index=False)
    result['missing_report'].to_csv(output/'data_quality.csv')
    for name,model in result['models'].items():
        model.save_model(output/f'xgboost_{name}.json')
    return output


def prediction_figure(predictions):
    import plotly.express as px
    return px.line(predictions,x=predictions.index,
        y=['actual_da','pred_weather','pred_fundamentals','quote_mid'],
        title='Held-out DA predictions and hypothetical quote benchmark',
        labels={'value':'EUR/MWh','delivery_start_utc':'Delivery (UTC)'},template='plotly_white')


def feature_importance_table(result):
    """Training-tree average gain, normalised within each model; no holdout fitting."""
    rows = []
    for name, model in result['models'].items():
        scores = model.get_booster().get_score(importance_type='gain')
        features = result['metadata']['features'][name]
        gains = np.array([scores.get(f, scores.get(f'f{i}', 0.0)) for i, f in enumerate(features)])
        shares = 100*gains/gains.sum() if gains.sum() else np.zeros(len(features))
        rows.extend(dict(model=name, feature=f, average_gain=float(g), importance_pct=float(s))
                    for f,g,s in zip(features,gains,shares))
    return pd.DataFrame(rows)


def feature_importance_figure(result):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    table = feature_importance_table(result)
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Weather model', 'Fundamentals model'],
                        horizontal_spacing=.25)
    for col,name in enumerate(['weather','fundamentals'],1):
        part = table[table.model==name].sort_values('importance_pct')
        fig.add_trace(go.Bar(x=part.importance_pct,y=part.feature,orientation='h',
            name=name, marker_color='#2563eb' if col==1 else '#d97706',
            text=part.importance_pct.round(1),texttemplate='%{text}%',
            hovertemplate='%{y}<br>Normalised average gain: %{x:.2f}%<extra></extra>'),row=1,col=col)
        fig.update_xaxes(title_text='Normalised average gain (%)',row=1,col=col)
    fig.update_layout(template='plotly_white',height=550,showlegend=False,
        title='Feature importance — relative within each model, not causal effects',margin=dict(l=170,r=50,t=90,b=70))
    return fig


def improvement_table(metrics):
    """Relative reduction in MAE, with Weather as denominator; undefined if it is zero."""
    comparison = metrics.pivot(index='subset',columns='model',values='mae')
    comparison['fundamentals_improvement_pct'] = 100*(comparison['weather']-comparison['fundamentals'])/comparison['weather'].replace(0,np.nan)
    return comparison


def regime_masks(result):
    actual = result['predictions'].actual_da
    return {'below_train_p90':actual < result['metadata']['high_threshold'],
            'high_train_quantile':actual >= result['metadata']['high_threshold'],
            'high_fixed':actual >= result['metadata']['config']['fixed_high_price']}


def regime_metrics_figure(result):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    names = ['below_train_p90','high_train_quantile','high_fixed']
    fig = make_subplots(rows=1,cols=3,subplot_titles=['MAE (lower is better)','RMSE (large misses)','Bias (negative = underprediction)'])
    colours = {'weather':'#2563eb','fundamentals':'#d97706','benchmark':'#64748b'}
    for col,metric in enumerate(['mae','rmse','bias'],1):
        for model in colours:
            part=result['metrics'][result['metrics'].model==model].set_index('subset').reindex(names)
            fig.add_trace(go.Bar(x=names,y=part[metric],name=model,legendgroup=model,
                showlegend=col==1,marker_color=colours[model],customdata=part[['n']].to_numpy(),
                hovertemplate='%{x}<br>Error: %{y:.2f} EUR/MWh<br>N=%{customdata[0]}<extra>%{fullData.name}</extra>'),row=1,col=col)
        fig.update_yaxes(title_text='EUR/MWh',zeroline=True,row=1,col=col)
        fig.update_xaxes(tickangle=-25,row=1,col=col)
    counts=result['metrics'][result['metrics'].model=='weather'].set_index('subset')['n']
    empty=[x for x in names if counts.loc[x]==0]
    note='; '.join(f'{x}: N={int(counts.loc[x])}' for x in names)
    if empty:note += ' | Empty groups have unavailable scores, not zero error.'
    fig.update_layout(template='plotly_white',barmode='group',height=550,
        title='Forecast errors by price regime — high-price groups may overlap',
        margin=dict(b=160),annotations=list(fig.layout.annotations)+[
            dict(text=note,x=.5,y=-.32,xref='paper',yref='paper',showarrow=False)])
    return fig


def regime_prices_figure(result):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    masks=regime_masks(result);p=result['predictions'];meta=result['metadata']
    quantile=100*meta['config']['high_quantile'];threshold=meta['high_threshold']
    fixed=meta['config']['fixed_high_price']
    labels=[f'below_train_p90: actual < train P{quantile:g} ({threshold:.2f}); N={int(masks["below_train_p90"].sum())}',
            f'high_train_quantile: actual >= {threshold:.2f}; N={int(masks["high_train_quantile"].sum())}',
            f'high_fixed: actual >= {fixed:g}; N={int(masks["high_fixed"].sum())}']
    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,subplot_titles=labels,vertical_spacing=.12)
    x=p.index.tz_convert(meta['config']['timezone'])
    for row,(regime,mask) in enumerate(masks.items(),1):
        for col,name,colour in [('actual_da','Actual DA','#111827'),('pred_weather','Weather','#2563eb'),('pred_fundamentals','Fundamentals','#d97706')]:
            # Retain excluded timestamps as NaN so separate episodes are never connected.
            fig.add_trace(go.Scatter(x=x,y=p[col].where(mask),mode='markers',marker=dict(size=4,color=colour),
                name=name,legendgroup=name,showlegend=row==1,connectgaps=False),row=row,col=1)
        if not mask.any():
            fig.add_annotation(text='No test intervals in this group',xref=f'x{row if row>1 else ""} domain',
                yref=f'y{row if row>1 else ""} domain',x=.5,y=.5,showarrow=False)
        fig.update_yaxes(title_text='EUR/MWh',row=row,col=1)
    fig.update_xaxes(title_text=f'Delivery time ({meta["config"]["timezone"]})',row=3,col=1)
    fig.update_layout(template='plotly_white',height=900,title='Actual and predicted prices within each evaluation group',hovermode='closest')
    return fig


def export_prediction_visuals(result, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    feature_importance_table(result).to_csv(output/'feature_importance.csv',index=False)
    improvement_table(result['metrics']).to_csv(output/'model_comparison.csv')
    figures={'feature_importance':feature_importance_figure(result),
             'regime_metrics':regime_metrics_figure(result),'regime_prices':regime_prices_figure(result)}
    for name,fig in figures.items():
        if result['metadata']['synthetic']:fig.update_layout(title_text='SYNTHETIC DEMO | '+fig.layout.title.text)
        fig.write_html(output/f'{name}.html',include_plotlyjs=True,full_html=True)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--demo',action='store_true',help='Use clearly labelled synthetic data, no API token')
    p.add_argument('--output',default='prediction_output')
    p.add_argument('--start',default='2026-06-01'); p.add_argument('--end',default='2026-10-01')
    p.add_argument('--zone',default='DK_2'); p.add_argument('--timezone',default='Europe/Copenhagen')
    p.add_argument('--latitude',type=float,default=55.6759); p.add_argument('--longitude',type=float,default=12.5655)
    p.add_argument('--weather-model',default='dmi_seamless')
    args=p.parse_args()
    cfg=PredictionConfig(zone=args.zone,timezone=args.timezone,start_date=args.start,end_date=args.end,
                         latitude=args.latitude,longitude=args.longitude,weather_model=args.weather_model)
    raw=demo_data(cfg) if args.demo else fetch_data(cfg)
    result=fit_evaluate(raw,cfg,synthetic=args.demo)
    output=export_result(result,args.output)
    export_prediction_visuals(result,output)
    print('SYNTHETIC DEMO' if args.demo else 'RETROSPECTIVE DATA; FORECAST VINTAGES UNVERIFIED')
    print(explain_metrics(result['metrics']))
    print(f'Exported to {output.resolve()}')

if __name__=='__main__':
    main()
