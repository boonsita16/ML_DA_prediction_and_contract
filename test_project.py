"""Meaningful invariants: isolation, units, costs, timing and pre-trade risk limits.
Run: python -m unittest -v test_project
All data below are synthetic; these tests do not verify market API credentials.
"""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import da_prediction as dp
import hypothetical_contract as hc

class ContractTests(unittest.TestCase):
    def signals(self, prediction=(120.,80.,100.)):
        index=pd.date_range('2026-07-02',periods=len(prediction),freq='15min',tz='UTC')
        return pd.DataFrame({'prediction':prediction,'quote_mid':100.,'duration_hours':.25,
            'decision_time_utc':pd.Timestamp('2026-07-01 08:00',tz='UTC')},index=index)

    def test_long_short_flat_and_costs(self):
        cfg=hc.ContractConfig()
        plan=hc.plan_positions(self.signals(),cfg,'Europe/Copenhagen')
        self.assertEqual(plan.action.tolist(),['BUY','SELL','NO_TRADE'])
        actual=pd.Series(90.,index=plan.index)
        ledger=hc.settle_positions(plan,actual,cfg)
        np.testing.assert_allclose(ledger.net_pnl_eur,[-3.25,1.75,0.])
        self.assertAlmostEqual(ledger.drawdown_eur.min(),-3.25)
        np.testing.assert_allclose(plan.quantity_mwh,[.25,-.25,0])

    def test_zero_negative_prices(self):
        signals=self.signals((-20.,-80.));signals['quote_mid']=-50.
        plan=hc.plan_positions(signals,hc.ContractConfig(),'UTC')
        ledger=hc.settle_positions(plan,pd.Series([-10.,-90.],index=plan.index),hc.ContractConfig())
        np.testing.assert_allclose(ledger.net_pnl_eur,[9.25,9.25])

    def test_daily_stress_and_volume_cap(self):
        cfg=hc.ContractConfig(daily_stress_budget_eur=30.,stress_move_eur_mwh=100.)
        plan=hc.plan_positions(self.signals((120.,130.,140.)),cfg,'UTC')
        self.assertAlmostEqual(plan.quantity_mwh.abs().sum(),.3)
        self.assertAlmostEqual(plan.quantity_mwh.iloc[2],.25)
        self.assertLessEqual(plan.stress_exposure_eur.sum(),30.+1e-9)

    def test_no_actual_allowed_in_planner(self):
        with self.assertRaises(ValueError):
            hc.plan_positions(self.signals().assign(actual_da=900.),hc.ContractConfig(),'UTC')

    def test_no_trade_and_zero_risk_budget(self):
        for cfg in [hc.ContractConfig(risk_buffer_eur_mwh=1e6),hc.ContractConfig(daily_stress_budget_eur=0.)]:
            plan=hc.plan_positions(self.signals(),cfg,'UTC')
            ledger=hc.settle_positions(plan,pd.Series(50.,index=plan.index),cfg)
            self.assertEqual(ledger.net_pnl_eur.sum(),0.)
            self.assertEqual(hc.summarise(ledger,'test')['n_trades'],0)

class PredictionTests(unittest.TestCase):
    def test_interval_fill_does_not_bridge_missing_hour(self):
        index=pd.date_range('2026-06-01',periods=5,freq='h',tz='UTC')
        source=pd.Series([1.,2.,3.,4.,5.],index=index).drop(index[2])
        grid=pd.date_range(index[0],periods=24,freq='15min')
        value=dp.to_quarter_hours(source,grid)
        self.assertTrue((value.iloc[:4]==1.).all())
        self.assertTrue(value.iloc[8:12].isna().all())
        self.assertTrue(value.iloc[20:].isna().all())

    def test_decision_hour_across_dst(self):
        cfg=dp.PredictionConfig()
        index=pd.date_range('2026-03-28',periods=5,freq='D',tz='UTC')
        decisions=dp.decision_times(index,cfg).tz_convert(cfg.timezone)
        self.assertTrue((decisions.hour==10).all())
        local=index.tz_convert(cfg.timezone).tz_localize(None).normalize()
        self.assertTrue(((local-decisions.tz_localize(None).normalize())==pd.Timedelta(days=1)).all())

    def test_test_labels_never_change_predictions_or_quotes_or_plan(self):
        cfg=dp.PredictionConfig(start_date='2026-06-01',end_date='2026-06-21')
        raw=dp.demo_data(cfg)
        cut=int(len(raw)*.8)
        first=dp.fit_evaluate(raw,cfg,synthetic=True)
        changed=raw.copy();changed.iloc[cut:,changed.columns.get_loc('actual_da')]+=1000.
        second=dp.fit_evaluate(changed,cfg,synthetic=True)
        cols=['pred_weather','pred_fundamentals','quote_mid','eligible_for_contract']
        assert_frame_equal(first['predictions'][cols],second['predictions'][cols])
        self.assertEqual(first['metadata']['high_threshold'],second['metadata']['high_threshold'])
        self.assertEqual(first['metadata']['n_train'],cut)
        with tempfile.TemporaryDirectory() as tmp:
            dp.export_result(first,tmp)
            f,meta=hc.load_bundle(tmp)
            result=hc.run_contract(f,meta,hc.ContractConfig())
            altered=f.copy();altered['actual_da']+=1000.;altered['high_price']=True
            result2=hc.run_contract(altered,meta,hc.ContractConfig())
            assert_frame_equal(result['ledger'][['quantity_mwh','execution_price']],
                               result2['ledger'][['quantity_mwh','execution_price']])
            (Path(tmp)/'test_predictions.csv').write_text('corrupt')
            with self.assertRaises(ValueError): hc.load_bundle(tmp)

if __name__=='__main__':
    unittest.main()
