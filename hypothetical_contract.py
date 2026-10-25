"""Financially settled hypothetical fixed-price contracts; no ID prices needed.
Run after da_prediction.py:
python hypothetical_contract.py --input prediction_output --output contract_output
Requires numpy, pandas and Plotly for exported charts.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class ContractConfig:
    spread_eur_mwh: float = 4.0  # FULL bid-ask width; half charged on either side
    slippage_eur_mwh: float = 0.5
    fee_eur_mwh: float = 0.5  # TOTAL contract fee; no closing transaction at settlement
    risk_buffer_eur_mwh: float = 5.0
    max_mw_per_interval: float = 1.0
    max_daily_gross_mwh: float = 10.0
    stress_move_eur_mwh: float = 100.0
    daily_stress_budget_eur: float = 500.0


def validate_config(cfg):
    for field,value in asdict(cfg).items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f'{field} must be finite and nonnegative.')
    if cfg.stress_move_eur_mwh <= 0:
        raise ValueError('Stress price move must be strictly positive.')


def load_bundle(directory):
    """Load the held-out forecast contract; check metadata and content integrity."""
    directory=Path(directory)
    metadata=json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
    csv=directory/'test_predictions.csv'
    if metadata.get('schema_version')!=1:
        raise ValueError('Unsupported prediction export schema.')
    if hashlib.sha256(csv.read_bytes()).hexdigest()!=metadata.get('predictions_sha256'):
        raise ValueError('Prediction CSV and metadata do not match; re-export predictions.')
    frame=pd.read_csv(csv)
    required={'delivery_start_utc','actual_da','pred_weather','pred_fundamentals','quote_mid',
              'decision_time_utc','duration_hours','eligible_for_contract','high_price'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing columns: {required-set(frame.columns)}')
    frame.index=pd.to_datetime(frame.pop('delivery_start_utc'),utc=True)
    frame.index.name='delivery_start_utc'
    frame['decision_time_utc']=pd.to_datetime(frame.decision_time_utc,utc=True)
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError('Predictions must have sorted unique delivery timestamps.')
    numeric=['actual_da','pred_weather','pred_fundamentals','quote_mid','duration_hours']
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError('Prediction bundle has missing/non-finite numbers.')
    if not np.allclose(frame.duration_hours,.25):
        raise ValueError('Expected 15-minute delivery contracts.')
    for col in ['eligible_for_contract','high_price']:
        values=frame[col].astype(str).str.lower()
        if not values.isin(['true','false']).all():
            raise ValueError(f'Invalid boolean column: {col}')
        frame[col]=values.eq('true')
    assumed_fit=pd.Timestamp(metadata['training_labels_available_assumed_utc'])
    expected=frame.decision_time_utc>assumed_fit
    if not np.array_equal(expected.to_numpy(),frame.eligible_for_contract.to_numpy()):
        raise ValueError('Trading eligibility is inconsistent with training completion.')
    if not (frame.decision_time_utc<frame.index).all():
        raise ValueError('Decision times must precede delivery.')
    if len(frame)!=metadata['n_test'] or frame.index.min()<=pd.Timestamp(metadata['train_end']):
        raise ValueError('Export is not the declared held-out period.')
    return frame,metadata


def plan_positions(signals, cfg, timezone):
    """Pre-decision inputs ONLY. No actual prices, high-price flags or realised P&L.

    Every delivery day's positions are chosen together. Rank opportunities by
    expected margin after costs; allocate subject to daily gross volume and a
    conservative stress exposure proxy. This is not a realised daily stop-loss.
    """
    validate_config(cfg)
    allowed={'prediction','quote_mid','duration_hours','decision_time_utc'}
    if set(signals.columns)!=allowed:
        raise ValueError('Planner accepts only prediction, quote, duration and decision time.')
    if not signals.index.is_unique:
        raise ValueError('Signals need unique timestamps.')
    if not np.isfinite(signals[['prediction','quote_mid','duration_hours']]).all().all():
        raise ValueError('Signals contain non-finite numbers.')
    if (signals.duration_hours<=0).any():
        raise ValueError('Contract durations must be positive.')
    plan=signals.copy().sort_index()
    plan['delivery_day']=plan.index.tz_convert(timezone).strftime('%Y-%m-%d')
    plan['bid']=plan.quote_mid-cfg.spread_eur_mwh/2
    plan['ask']=plan.quote_mid+cfg.spread_eur_mwh/2
    plan['buy_edge']=plan.prediction-(plan.ask+cfg.slippage_eur_mwh)-cfg.fee_eur_mwh
    plan['sell_edge']=(plan.bid-cfg.slippage_eur_mwh)-plan.prediction-cfg.fee_eur_mwh
    plan['side']=np.where(plan.buy_edge>cfg.risk_buffer_eur_mwh,1,
                           np.where(plan.sell_edge>cfg.risk_buffer_eur_mwh,-1,0))
    plan['expected_edge']=np.maximum(plan.buy_edge,plan.sell_edge)
    plan['quantity_mwh']=0.0
    # Gross exposure means opposite trades cannot mask each other's risk.
    daily_cap=min(cfg.max_daily_gross_mwh,
                  cfg.daily_stress_budget_eur/cfg.stress_move_eur_mwh)
    for _,group in plan.groupby('delivery_day',sort=True):
        if group.decision_time_utc.nunique()!=1:
            raise ValueError('Each delivery day must share a single pre-auction decision time.')
        remaining=daily_cap
        ranked=group[group.side!=0].sort_values('expected_edge',ascending=False,kind='stable')
        for timestamp,row in ranked.iterrows():
            volume=min(cfg.max_mw_per_interval*row.duration_hours,remaining)
            if volume<=1e-12:
                break
            plan.at[timestamp,'quantity_mwh']=row.side*volume
            remaining-=volume
    plan['action']=np.select([plan.quantity_mwh>0,plan.quantity_mwh<0],['BUY','SELL'],default='NO_TRADE')
    executed_side=np.sign(plan.quantity_mwh)
    plan['execution_price']=np.where(executed_side!=0,
        plan.quote_mid+executed_side*(cfg.spread_eur_mwh/2+cfg.slippage_eur_mwh),np.nan)
    plan['fees_eur']=plan.quantity_mwh.abs()*cfg.fee_eur_mwh
    plan['expected_net_pnl_eur']=(plan.quantity_mwh*(plan.prediction-plan.execution_price)).fillna(0)-plan.fees_eur
    plan['stress_exposure_eur']=plan.quantity_mwh.abs()*cfg.stress_move_eur_mwh
    return plan


def settle_positions(plan, actual, cfg):
    """Only here are actual DA prices introduced. Signed Q handles long and short."""
    if not actual.index.equals(plan.index) or not np.isfinite(actual).all():
        raise ValueError('Settlement prices must align exactly with planned delivery intervals.')
    ledger=plan.copy()
    ledger['actual_da']=actual
    traded=ledger.quantity_mwh.ne(0)
    ledger['gross_pnl_eur']=np.where(traded,
        ledger.quantity_mwh*(ledger.actual_da-ledger.execution_price),0.)
    ledger['net_pnl_eur']=ledger.gross_pnl_eur-ledger.fees_eur
    ledger['cumulative_pnl_eur']=ledger.net_pnl_eur.cumsum()
    ledger['drawdown_eur']=ledger.cumulative_pnl_eur-ledger.cumulative_pnl_eur.cummax().clip(lower=0)
    # +/- stress moves are sensitivities, not predictions or maximum possible loss.
    ledger['pnl_change_if_da_up_stress']=ledger.quantity_mwh*cfg.stress_move_eur_mwh
    ledger['pnl_change_if_da_down_stress']=-ledger.quantity_mwh*cfg.stress_move_eur_mwh
    return ledger


def summarise(ledger, model):
    active=ledger[ledger.quantity_mwh!=0]
    gross_volume=ledger.quantity_mwh.abs().sum()
    n_days=ledger.delivery_day.nunique()
    daily=ledger.groupby('delivery_day').net_pnl_eur.sum()
    return dict(model=model,n_intervals=len(ledger),n_trades=len(active),n_days=n_days,
        long_trades=int((ledger.quantity_mwh>0).sum()),short_trades=int((ledger.quantity_mwh<0).sum()),
        gross_volume_mwh=gross_volume,net_pnl_eur=ledger.net_pnl_eur.sum(),
        fees_eur=ledger.fees_eur.sum(),
        pnl_per_traded_mwh=ledger.net_pnl_eur.sum()/gross_volume if gross_volume else np.nan,
        win_rate_pct=100*(active.net_pnl_eur>0).mean() if len(active) else np.nan,
        worst_day_eur=daily.min() if len(daily) else np.nan,
        losing_days=int((daily<0).sum()),
        max_drawdown_eur=-ledger.drawdown_eur.min() if len(ledger) else 0.)


def run_contract(frame, metadata, cfg):
    validate_config(cfg)
    eligible=frame[frame.eligible_for_contract].copy()
    if eligible.empty:
        raise ValueError('No contract-eligible observations after the training availability cutoff.')
    timezone=metadata['config']['timezone']
    ledgers=[]; summaries=[]; daily_parts=[]; regimes=[]
    for model in ['weather','fundamentals']:
        # Explicit allowlist prevents settlement prices from entering the planner.
        signals=eligible[[f'pred_{model}','quote_mid','duration_hours','decision_time_utc']].rename(
            columns={f'pred_{model}':'prediction'})
        plan=plan_positions(signals,cfg,timezone)
        ledger=settle_positions(plan,eligible.actual_da,cfg)
        ledger['model']=model
        ledger['high_price']=eligible.high_price
        summaries.append(summarise(ledger,model))
        daily=ledger.groupby('delivery_day').agg(net_pnl_eur=('net_pnl_eur','sum'),
            expected_net_pnl_eur=('expected_net_pnl_eur','sum'),
            gross_volume_mwh=('quantity_mwh',lambda x:x.abs().sum()),
            stress_exposure_eur=('stress_exposure_eur','sum'),n_intervals=('net_pnl_eur','size'))
        daily['model']=model
        daily_parts.append(daily.reset_index())
        for label,mask in [('all',pd.Series(True,index=ledger.index)),
                           ('high_price',ledger.high_price),('other',~ledger.high_price)]:
            part=ledger[mask]
            traded=part[part.quantity_mwh!=0]
            regimes.append(dict(model=model,regime=label,n_intervals=len(part),n_trades=len(traded),
                net_pnl_eur=part.net_pnl_eur.sum(),
                prediction_mae=(part.prediction-part.actual_da).abs().mean(),
                quote_mae=(part.quote_mid-part.actual_da).abs().mean()))
        ledgers.append(ledger)
    report={'config':asdict(cfg),'timezone':timezone,'synthetic':metadata['synthetic'],
        'forecast_vintages_verified':metadata['forecast_vintages_verified'],
        'excluded_unavailable_decisions':int((~frame.eligible_for_contract).sum()),
        'source_predictions_sha256':metadata['predictions_sha256'],
        'assumptions':['Hypothetical financially settled fixed-price contracts, not exchange DA orders',
         'Unlimited counterparty quote availability up to configured volume; no rejection/default/collateral costs',
         'Frozen training-only calendar mean is quote midpoint; both models face identical bid/ask quotes',
         'Orders selected before observing settlement prices; actual DA used only in settlement/evaluation',
         'Stress budget is exposure under an assumed adverse move, not a guaranteed loss cap',
         'Each model is a separate alternative strategy, not a combined portfolio',
         'Retrospective forecast vintages are unverified; hypothetical P&L is not demonstrated achievable profit',
         'Only complete-case forecast rows are scored; missing periods are not imputed',
         'Zero-trade cash benchmark has zero P&L; no interest or funding included']}
    return {'ledger':pd.concat(ledgers),'summary':pd.DataFrame(summaries),
            'daily':pd.concat(daily_parts,ignore_index=True),'regimes':pd.DataFrame(regimes),'report':report}


def export_contract(result,directory):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    result['ledger'].to_csv(directory/'trade_ledger.csv')
    for name in ['summary','daily','regimes']:
        result[name].to_csv(directory/f'{name}.csv',index=False)
    (directory/'assumptions.json').write_text(json.dumps(result['report'],indent=2),encoding='utf-8')
    return directory


def pnl_figure(result):
    import plotly.express as px
    return px.line(result['ledger'].reset_index(),x='delivery_start_utc',y='cumulative_pnl_eur',color='model',
        title='Hypothetical cumulative net P&L — not realised market trading',
        labels={'cumulative_pnl_eur':'EUR'},template='plotly_white')


def commercial_performance_figure(result):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=.10,
        subplot_titles=['Cumulative settled net P&L','Daily net P&L','Drawdown from previous peak (starts at zero)'])
    for model,colour in [('weather','#2563eb'),('fundamentals','#d97706')]:
        ledger=result['ledger'][result['ledger'].model==model].sort_index()
        daily=result['daily'][result['daily'].model==model].sort_values('delivery_day')
        fig.add_trace(go.Scatter(x=ledger.index,y=ledger.cumulative_pnl_eur,name=model,legendgroup=model,line_color=colour),row=1,col=1)
        fig.add_trace(go.Bar(x=pd.to_datetime(daily.delivery_day).dt.tz_localize(result['report'].get('timezone','Europe/Copenhagen')).dt.tz_convert('UTC'),y=daily.net_pnl_eur,name=model,legendgroup=model,showlegend=False,marker_color=colour),row=2,col=1)
        fig.add_trace(go.Scatter(x=ledger.index,y=ledger.drawdown_eur,name=model,legendgroup=model,showlegend=False,line_color=colour),row=3,col=1)
    for row in [1,2,3]:
        fig.add_hline(y=0,line_dash='dot',line_color='grey',row=row,col=1)
        fig.update_yaxes(title_text='EUR',row=row,col=1)
    fig.update_layout(template='plotly_white',height=900,barmode='group',hovermode='x unified',
        title='Hypothetical commercial performance — alternative strategies, not combined profit')
    fig.update_xaxes(title_text='UTC timestamps; daily bars are positioned at local midnight converted to UTC',row=3,col=1)
    return fig


def decision_figure(result, model='weather', delivery_day=None):
    """Delivery-time view: decisions were made on the preceding day, not at these prices."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    if model not in ['weather','fundamentals']:raise ValueError('Choose weather or fundamentals.')
    data=result['ledger'][result['ledger'].model==model].sort_index().copy()
    if data.empty:raise ValueError('No eligible intervals for this model.')
    if delivery_day is None:delivery_day=data.delivery_day.iloc[0]  # no hindsight cherry-picking
    data=data[data.delivery_day==str(delivery_day)]
    if data.empty:raise ValueError(f'No eligible rows on {delivery_day}; inspect the daily results.')
    cfg=result['report']['config']
    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=.10,
        row_heights=[.58,.20,.22],subplot_titles=['Prices, decision thresholds and executed trades',
        'Signed position (+ buy, − sell)','Net settlement P&L per interval'])
    series=[('actual_da','Actual DA (known later)','#111827','solid'),
            ('prediction','ML prediction','#2563eb','solid'),('quote_mid','Quote midpoint','#64748b','dot'),
            ('bid','Counterparty bid','#b91c1c','dash'),('ask','Counterparty ask','#15803d','dash')]
    for col,label,colour,dash in series:
        fig.add_trace(go.Scatter(x=data.index,y=data[col],name=label,line=dict(color=colour,dash=dash)),row=1,col=1)
    extra=cfg['slippage_eur_mwh']+cfg['fee_eur_mwh']+cfg['risk_buffer_eur_mwh']
    for y,label,colour in [(data.ask+extra,'Buy threshold (forecast must exceed)','#86a873'),
                            (data.bid-extra,'Sell threshold (forecast must fall below)','#cf9b9b')]:
        fig.add_trace(go.Scatter(x=data.index,y=y,name=label,line=dict(color=colour,dash='dot'),visible='legendonly'),row=1,col=1)
    for action,symbol,colour in [('BUY','triangle-up','#16a34a'),('SELL','triangle-down','#dc2626')]:
        traded=data[data.action==action]
        custom=np.column_stack([traded.quantity_mwh.to_numpy(),traded.net_pnl_eur.to_numpy(),
                                traded.prediction.to_numpy(),traded.actual_da.to_numpy(),
                                traded.decision_time_utc.astype(str).to_numpy()])
        fig.add_trace(go.Scatter(x=traded.index,y=traded.execution_price,mode='markers',name=action,
            marker=dict(symbol=symbol,size=12,color=colour,line=dict(color='white',width=1)),customdata=custom,
            hovertemplate='Delivery %{x}<br>Execution %{y:.2f} EUR/MWh<br>Signed MWh %{customdata[0]:.2f}'
                '<br>Net P&L EUR %{customdata[1]:.2f}<br>Forecast %{customdata[2]:.2f}'
                '<br>Actual %{customdata[3]:.2f}<br>Decision %{customdata[4]}<extra>%{fullData.name}</extra>'),row=1,col=1)
    # Candidates rejected by volume/risk allocation must not appear as executed trades.
    blocked=data[(data.side!=0)&(data.quantity_mwh==0)]
    fig.add_trace(go.Scatter(x=blocked.index,y=blocked.prediction,mode='markers',name='Signal skipped: risk/volume cap',
        marker=dict(symbol='x',size=8,color='#9ca3af')),row=1,col=1)
    colours=np.where(data.quantity_mwh>0,'#16a34a',np.where(data.quantity_mwh<0,'#dc2626','#9ca3af'))
    fig.add_trace(go.Bar(x=data.index,y=data.quantity_mwh,name='Signed energy',marker_color=colours,showlegend=False),row=2,col=1)
    fig.add_trace(go.Bar(x=data.index,y=data.net_pnl_eur,name='Net P&L',marker_color=np.where(data.net_pnl_eur>=0,'#16a34a','#dc2626'),showlegend=False),row=3,col=1)
    fig.update_yaxes(title_text='EUR/MWh',row=1,col=1)
    fig.update_yaxes(title_text='MWh',row=2,col=1)
    fig.update_yaxes(title_text='EUR',row=3,col=1)
    for row in [2,3]:fig.add_hline(y=0,line_color='grey',row=row,col=1)
    prefix='SYNTHETIC DEMO | ' if result['report']['synthetic'] else ''
    decision=str(data.decision_time_utc.iloc[0])
    fig.update_layout(template='plotly_white',height=1000,hovermode='x unified',
        title=f'{prefix}{model}: delivery day {delivery_day} | decision made {decision}',
        legend=dict(orientation='h',y=-.12),margin=dict(b=180))
    fig.update_xaxes(title_text='Delivery time (UTC); arrows mark execution prices of previously chosen contracts',row=3,col=1)
    return fig


def export_commercial_visuals(result, directory, delivery_day=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    dashboard=commercial_performance_figure(result)
    if result['report']['synthetic']:dashboard.update_layout(title_text='SYNTHETIC DEMO | '+dashboard.layout.title.text)
    dashboard.write_html(directory/'commercial_performance.html',include_plotlyjs=True)
    for model in ['weather','fundamentals']:
        decision_figure(result,model,delivery_day).write_html(directory/f'decisions_{model}.html',include_plotlyjs=True)
    return directory


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='prediction_output');p.add_argument('--output',default='contract_output')
    p.add_argument('--plot-day',default=None,help='Local delivery date YYYY-MM-DD for decision plots; defaults to first eligible day')
    p.add_argument('--spread',type=float,default=4.0);p.add_argument('--buffer',type=float,default=5.0)
    p.add_argument('--fee',type=float,default=.5);p.add_argument('--slippage',type=float,default=.5)
    p.add_argument('--max-mw',type=float,default=1.);p.add_argument('--max-daily-mwh',type=float,default=10.)
    p.add_argument('--stress-move',type=float,default=100.);p.add_argument('--stress-budget',type=float,default=500.)
    a=p.parse_args()
    cfg=ContractConfig(spread_eur_mwh=a.spread,risk_buffer_eur_mwh=a.buffer,fee_eur_mwh=a.fee,
        slippage_eur_mwh=a.slippage,max_mw_per_interval=a.max_mw,max_daily_gross_mwh=a.max_daily_mwh,
        stress_move_eur_mwh=a.stress_move,daily_stress_budget_eur=a.stress_budget)
    frame,metadata=load_bundle(a.input)
    result=run_contract(frame,metadata,cfg)
    export_contract(result,a.output)
    export_commercial_visuals(result,a.output,a.plot_day)
    print('SYNTHETIC DEMO' if metadata['synthetic'] else 'HYPOTHETICAL RETROSPECTIVE CONTRACT SIMULATION')
    print(result['summary'].to_string(index=False))
    print(f'Files saved in {Path(a.output).resolve()}')

if __name__=='__main__':
    main()
