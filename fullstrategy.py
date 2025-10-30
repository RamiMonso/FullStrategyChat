# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from math import floor
from io import BytesIO
import base64
import datetime

st.set_page_config(layout="wide", page_title="Swing Strategy Backtester")

# -------------------
# Indicator helpers (same logic as script)
# -------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_atr(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def compute_adx(df, period=14):
    high = df['High']; low = df['Low']; close = df['Close']
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_smooth = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean()
    plus_smooth = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean()
    minus_smooth = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean()

    plus_di = 100 * (plus_smooth / tr_smooth)
    minus_di = 100 * (minus_smooth / tr_smooth)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    adx.index = df.index
    return adx

def compute_stoch_rsi(rsi_series, period=14):
    min_rsi = rsi_series.rolling(window=period, min_periods=1).min()
    max_rsi = rsi_series.rolling(window=period, min_periods=1).max()
    stoch = (rsi_series - min_rsi) / (max_rsi - min_rsi)
    return stoch * 100

# -------------------
# Backtest engine (simplified/identical logic)
# -------------------
def backtest(df, params):
    df = df.copy()
    df['EMA'] = df['Close'].ewm(span=params['ema_period'], adjust=False).mean()
    df['RSI'] = compute_rsi(df['Close'], period=params['rsi_period'])
    df['ATR'] = compute_atr(df, period=params['atr_period'])
    df['ADX'] = compute_adx(df, period=params['adx_period'])
    df['StochRSI'] = compute_stoch_rsi(df['RSI'], period=params['rsi_period'])
    df['Vol20'] = df['Volume'].rolling(window=20, min_periods=1).mean()

    trades = []
    equity = params['capital']
    equity_curve = []
    position = None

    for i in range(1, len(df)):
        today = df.index[i]
        row = df.iloc[i]
        prev = df.iloc[i-1]

        # mark-to-market
        if position is None:
            equity_curve.append({'Date': today, 'Equity': equity})
        else:
            market_value = position['shares'] * row['Close']
            cur_equity = equity - position['cost_basis'] + market_value
            equity_curve.append({'Date': today, 'Equity': cur_equity})

        # entry
        if position is None:
            cond_trend = prev['Close'] > prev['EMA']
            cond_rsi = prev['RSI'] <= params['rsi_thresh']
            cond_adx = prev['ADX'] > params['adx_thresh']
            cond_vol = prev['Volume'] >= (prev['Vol20'] * params['vol_factor'])
            if cond_trend and cond_rsi and cond_adx and cond_vol:
                entry_price_raw = row['Open']
                entry_price = entry_price_raw * (1 + params['slippage_pct'])
                atr = prev['ATR'] if not np.isnan(prev['ATR']) else row['ATR']
                stop_price = entry_price - params['atr_mult_stop'] * atr
                target_price = entry_price * (1 + params['target_pct'])
                risk_amount = params['capital'] * params['risk_pct']
                stop_distance = entry_price - stop_price
                if stop_distance <= 0:
                    continue
                shares = floor(risk_amount / stop_distance)
                if shares <= 0:
                    continue
                cost_basis = shares * entry_price
                if cost_basis > params.get('max_exposure_pct', 0.1) * params['capital']:
                    max_cost = params.get('max_exposure_pct', 0.1) * params['capital']
                    shares = floor(max_cost / entry_price)
                    if shares <= 0:
                        continue
                    cost_basis = shares * entry_price
                position = {
                    'entry_date': today,
                    'entry_price': entry_price,
                    'shares': shares,
                    'stop_price': stop_price,
                    'target_price': target_price,
                    'atr': atr,
                    'days_held': 0,
                    'cost_basis': cost_basis,
                    'activated_trailing': False,
                    'highest_price': entry_price
                }
                equity -= params['commission_pct'] * cost_basis
        else:
            position['days_held'] += 1
            high = row['High']; low = row['Low']; close = row['Close']
            atr = row['ATR'] if not np.isnan(row['ATR']) else position['atr']
            if high > position['highest_price']:
                position['highest_price'] = high
            if (not position['activated_trailing']) and (position['highest_price'] >= position['entry_price'] + params['trailing_activation_mult'] * position['atr']):
                position['stop_price'] = max(position['stop_price'], position['entry_price'] + params['trailing_initial_mult'] * position['atr'])
                position['activated_trailing'] = True
            if position['activated_trailing']:
                new_stop = position['highest_price'] - params['trailing_active_mult'] * atr
                position['stop_price'] = max(position['stop_price'], new_stop)

            exit_reason = None; exit_price = None
            if low <= position['stop_price']:
                exit_price = position['stop_price'] * (1 - params['slippage_pct'])
                exit_reason = 'stop'
            elif high >= position['target_price']:
                exit_price = position['target_price'] * (1 - params['slippage_pct'])
                exit_reason = 'target'
            elif position['days_held'] >= params['max_days']:
                exit_price = close * (1 - params['slippage_pct'])
                exit_reason = 'time'
            elif close < row['EMA'] and params.get('exit_on_ema_break', False):
                exit_price = close * (1 - params['slippage_pct'])
                exit_reason = 'ema_break'

            if exit_reason is not None:
                proceeds = position['shares'] * exit_price
                pnl = proceeds - position['cost_basis']
                ret_pct = pnl / position['cost_basis']
                equity += proceeds
                equity -= params['commission_pct'] * proceeds
                trades.append({
                    'entry_date': position['entry_date'],
                    'entry_price': position['entry_price'],
                    'shares': position['shares'],
                    'stop_price': position['stop_price'],
                    'target_price': position['target_price'],
                    'exit_date': today,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'days_held': position['days_held'],
                    'pnl': pnl,
                    'return_pct': ret_pct
                })
                position = None

    eq_df = pd.DataFrame(equity_curve).set_index('Date').sort_index()
    trades_df = pd.DataFrame(trades)
    metrics = compute_metrics(eq_df, trades_df, params['capital'])
    return trades_df, eq_df, metrics

def compute_metrics(equity_df, trades_df, starting_capital):
    metrics = {}
    if equity_df.empty:
        return metrics
    equity_series = equity_df['Equity']
    returns = equity_series.pct_change().fillna(0)
    days = (equity_df.index[-1] - equity_df.index[0]).days
    years = days / 365.25 if days > 0 else 1
    ending = equity_series.iloc[-1]
    cagr = (ending / starting_capital) ** (1 / years) - 1 if years > 0 else 0
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else np.nan
    roll_max = equity_series.cummax()
    drawdown = (equity_series - roll_max) / roll_max
    max_dd = drawdown.min()
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]
    win_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else np.nan
    avg_win = wins['pnl'].mean() if not wins.empty else 0
    avg_loss = losses['pnl'].mean() if not losses.empty else 0
    avg_return = trades_df['return_pct'].mean() if not trades_df.empty else 0
    expectancy = win_rate * (wins['pnl'].mean() if not wins.empty else 0) - (1 - win_rate) * (abs(losses['pnl'].mean()) if not losses.empty else 0)
    metrics.update({
        'cagr': cagr,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'n_trades': len(trades_df),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_return_per_trade': avg_return,
        'expectancy': expectancy,
        'ending_equity': ending
    })
    return metrics

# -------------------
# UI
# -------------------
st.title("Swing Strategy Backtester — RSI+EMA50+ADX+ATR")
with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Ticker (Yahoo)", value="AAPL")
    start = st.date_input("Start date", value=datetime.date(2018,1,1))
    end = st.date_input("End date (inclusive)", value=datetime.date.today())
    capital = st.number_input("Starting capital", value=100000.0, step=1000.0)
    risk = st.number_input("Risk % per trade", value=0.01, format="%.4f")
    slippage = st.number_input("Slippage % (per trade)", value=0.001, format="%.4f")
    commission = st.number_input("Commission % (per trade)", value=0.0005, format="%.5f")
    run_btn = st.button("Run backtest")

params_default = {
    'rsi_period': 14, 'rsi_thresh': 35, 'ema_period': 50,
    'adx_period': 14, 'adx_thresh': 18, 'atr_period': 14,
    'atr_mult_stop': 1.5, 'target_pct': 0.06, 'max_days': 30,
    'trailing_activation_mult': 2.0, 'trailing_initial_mult': 0.5,
    'trailing_active_mult': 1.0, 'risk_pct': risk, 'capital': capital,
    'slippage_pct': slippage, 'commission_pct': commission, 'vol_factor': 0.7,
    'max_exposure_pct': 0.1, 'exit_on_ema_break': False
}

st.info("Tip: pin working package versions in requirements.txt to avoid Cloud install errors.")

if run_btn:
    st.info(f"Downloading {ticker} from Yahoo Finance ...")
    try:
        df = yf.download(ticker, start=start, end=end + datetime.timedelta(days=1), progress=False)
        if df.empty:
            st.error("No data downloaded. Check ticker and date range.")
        else:
            trades_df, eq_df, metrics = backtest(df, params_default)
            st.subheader("Metrics")
            st.write(metrics)
            st.subheader("Equity curve")
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(eq_df.index, eq_df['Equity'])
            ax.set_title("Equity curve")
            ax.grid(True)
            st.pyplot(fig)

            st.subheader("Trades")
            if not trades_df.empty:
                st.dataframe(trades_df)
                csv = trades_df.to_csv(index=False).encode('utf-8')
                b64 = base64.b64encode(csv).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="trades_{ticker}.csv">Download trades CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.write("No trades were opened with the current params.")
    except Exception as e:
        st.error(f"Error: {e}\n(If on Streamlit Cloud and using yfinance, see known issues: caching or timezone errors.)")

