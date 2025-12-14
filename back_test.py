import sys
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import MetaTrader5 as mt5

# 导入现有模块（调整路径根据目录结构）
root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_path)
from mt5.mt5_handler import MT5Handler
from strategies.v6_ma_dynamic_stop_loss_usoil import RSIHighFreqXAUUSD

class Backtester:
    def __init__(self, symbol, time_frame=5, initial_balance=10000.0, volume=0.1, lever=10,
                 dynamic_sl_enabled=True, dynamic_tp_enabled=True,
                 buy_rsi=35, sell_rsi=65, periods=20, atr_threshold=0.1,
                 stop_loss_cooling=30*60, take_profit_cooling=5*60,
                 max_stop_loss=-5, min_take_profit=1.5, dynamic_tp_threshold=-0.2):
        self.symbol = symbol
        self.time_frame = time_frame
        self.initial_balance = initial_balance
        self.volume = volume
        self.lever = lever
        self.dynamic_sl_enabled = dynamic_sl_enabled
        self.dynamic_tp_enabled = dynamic_tp_enabled
        self.buy_rsi = buy_rsi
        self.sell_rsi = sell_rsi
        self.periods = periods
        self.atr_threshold = atr_threshold
        self.stop_loss_cooling = stop_loss_cooling
        self.take_profit_cooling = take_profit_cooling
        self.max_stop_loss = max_stop_loss
        self.min_take_profit = min_take_profit
        self.dynamic_tp_threshold = dynamic_tp_threshold
        self.mt5_handler = MT5Handler()
        self.strategy = RSIHighFreqXAUUSD(handler=self.mt5_handler,
                                          dynamic_sl_enabled=dynamic_sl_enabled,
                                          dynamic_tp_enabled=dynamic_tp_enabled,
                                          buy_rsi=buy_rsi, sell_rsi=sell_rsi, periods=periods, atr_threshold=atr_threshold,
                                          stop_loss_cooling=stop_loss_cooling, take_profit_cooling=take_profit_cooling,
                                          max_stop_loss=max_stop_loss, min_take_profit=min_take_profit, dynamic_tp_threshold=dynamic_tp_threshold)
        self.history_data = None
        self.trades = []  # 存储交易记录
        self.equity_curve = []  # 权益曲线
        self.last_dynamic_stop_loss_time = None
        self.last_dynamic_take_profit_time = None

    def load_data(self, days=365, timeframe=mt5.TIMEFRAME_M5):
        os.makedirs("data", exist_ok=True)
        local_csv = os.path.join(f"./data/{self.symbol}_{str(timeframe)}min_{str(days)}_1min.csv")  # 都是下载1min数据
        """加载历史数据，支持从 MT5 下载或本地 CSV"""
        if os.path.exists(local_csv):
            self.history_data = pd.read_csv(local_csv, parse_dates=['time'], index_col='time')
            print(f"加载本地数据: {local_csv}, 行数: {len(self.history_data)}")
        else:
            self.history_data = self.mt5_handler.download_data(self.symbol, timeframe=1, days=days)
            print(f"从 MT5 下载数据: {self.symbol}, 天数: {days}, 行数: {len(self.history_data)}")
            self.history_data.to_csv(local_csv)
        if self.history_data.empty:
            raise ValueError("历史数据为空，无法进行回测")
        # 确保数据有 OHLC 列
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in self.history_data.columns for col in required_cols):
            raise ValueError("历史数据缺少必要的 OHLC 列")

    def is_in_cooling_period(self, current_time):
        """检查是否处于冷静期（模拟策略中的 is_in_cooling_period）"""
        if self.last_dynamic_stop_loss_time is None and self.last_dynamic_take_profit_time is None:
            return False
        last_sl_time = self.last_dynamic_stop_loss_time or self.last_dynamic_take_profit_time
        cooling_seconds = self.stop_loss_cooling if self.last_dynamic_stop_loss_time else self.take_profit_cooling
        elapsed = (current_time - last_sl_time).total_seconds()
        if elapsed < cooling_seconds:
            return True
        return False

    def monitor_position(self, position, current_price, current_time):
        """模拟动态止损/止盈监控"""
        if not position:
            return None, 0.0

        profit = 0.0
        close_reason = None
        if position['type'] == 'buy':
            profit = (current_price - position['open_price']) * (self.volume / 0.01) * self.lever
            print(f"current profit:{profit}")
            if self.dynamic_sl_enabled and profit <= (self.max_stop_loss * (self.volume / 0.01)):
                close_reason = 'dynamic_sl'
                self.last_dynamic_stop_loss_time = current_time
                print(f"position['type']:{position['type']}, stop loss, profit: {profit}")
            elif self.dynamic_tp_enabled and profit >= (self.min_take_profit * (self.volume / 0.01)):
                # price_change_rate = (current_price - position['prev_price']) / position['prev_price'] if position['prev_price'] else 0
                profit_change_rate = (position['max_profit'] - profit) / position['max_profit'] if position['max_profit'] > 0 else 0
                if profit_change_rate >= -self.dynamic_tp_threshold:
                    close_reason = 'dynamic_tp'
                    self.last_dynamic_take_profit_time = current_time
                    print(f"position['type']:{position['type']}, take profit, profit: {profit}")
        elif position['type'] == 'sell':
            # 利润=（卖出价格-当前价格）* (卖出手数/0.01) * 10
            profit = (position['open_price'] - current_price) * (self.volume / 0.01) * self.lever
            print(f"current profit:{profit}")
            if self.dynamic_sl_enabled and profit <= (self.max_stop_loss * (self.volume / 0.01)):
                close_reason = 'dynamic_sl'
                self.last_dynamic_stop_loss_time = current_time
                print(f"position['type']:{position['type']}, stop loss, profit: {profit}")
            elif self.dynamic_tp_enabled and profit >= (self.min_take_profit * (self.volume / 0.01)):
                # price_change_rate = (current_price - position['prev_price']) / position['prev_price'] if position['prev_price'] else 0
                profit_change_rate = (position['max_profit'] - profit) / position['max_profit'] if position['max_profit'] > 0 else 0
                if profit_change_rate >= -self.dynamic_tp_threshold:  # 对于卖出，价格上涨触发
                    close_reason = 'dynamic_tp'
                    self.last_dynamic_take_profit_time = current_time
                    print(f"position['type']:{position['type']}, take profit, profit: {profit}")

        if profit > position['max_profit']:
            position['max_profit'] = profit
            print(f"uodate position['max_profit']:{position['max_profit']}")

        if close_reason:
            # 新增：计算手续费并扣除（最小改动，仅在此处添加）
            fee = abs(current_price - position['open_price']) * (self.volume / 0.01) * 0.18
            net_profit = profit - fee
            print(f"手续费: {fee:.2f}, 净利润: {net_profit:.2f}")  # 可选调试日志

            self.trades.append({
                'time': current_time,
                'type': f'close_{position["type"]}_{close_reason}',
                'price': current_price,
                'profit': net_profit  # 修改：使用净利润
            })
            return None, net_profit  # 修改：返回净利润
        else:
            # 更新 prev_price
            position['prev_price'] = current_price
            return position, 0.0


    def run_backtest(self):
        """运行回测，模拟逐条 K 线处理"""
        balance = self.initial_balance
        position = None  # 当前持仓: {'type': 'buy/sell', 'open_price': float, 'open_time': datetime, 'prev_price': float}
        self.equity_curve.append((self.history_data.index[0], balance))  # 初始权益

        for i in range(self.periods * self.time_frame, len(self.history_data)):  # 从足够的历史开始计算 RSI/ATR
            current_time = self.history_data.index[i]
            data_window = self.history_data.iloc[:i+1]  # 到当前时间的全部历史

            # 检查冷静期
            if self.is_in_cooling_period(current_time):
                continue

            if len(data_window) < self.periods * self.time_frame:
                continue

            if current_time.minute % self.time_frame != 0:
                signal = None
            else:
                if not position:
                    # resample 1min to 5min for indicator calculation
                    resampled = data_window.resample(f'{self.time_frame}T').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
                    # 计算 ATR 和 RSI（通过策略的 get_signal）
                    atr = self.strategy.calculate_atr(resampled)
                    if atr.iloc[-1] < self.atr_threshold:
                        signal = None
                    else:
                        signal = self.strategy.get_signal(resampled, self.symbol, is_reload_data=False, is_back_test=True)

            current_price = data_window['close'].iloc[-1]

            # 监控现有持仓
            if position:
                position, close_profit = self.monitor_position(position, current_price, current_time)
                if close_profit != 0:
                    print()
                balance += close_profit

            # 开新仓（最多1个持仓）
            if signal and not position:
                position = {
                    'type': signal,
                    'open_price': current_price,
                    'open_time': current_time,
                    'prev_price': current_price,
                    'max_profit': 0
                }
                self.trades.append({
                    'time': current_time,
                    'type': f'open_{signal}',
                    'price': current_price,
                    'profit': 0
                })

            self.equity_curve.append((current_time, balance))  # 更新权益曲线

        print(f"回测完成，最终余额: {balance:.2f}")

    def generate_report(self, output_dir='backtest_results'):
        """生成回测报告"""
        os.makedirs(output_dir, exist_ok=True)
        total_profit = sum(trade['profit'] for trade in self.trades if 'close' in trade['type'])
        win_trades = sum(1 for trade in self.trades if 'close' in trade['type'] and trade['profit'] > 0)
        total_trades = len([t for t in self.trades if 'close' in t['type']])
        win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
        max_drawdown = self.calculate_max_drawdown()

        report = f"总盈亏: {total_profit:.2f}\n胜率: {win_rate:.2f}%\n交易次数: {total_trades}\n最大回撤: {max_drawdown:.2f}%"
        print(report)
        with open(os.path.join(output_dir, 'report.txt'), 'w') as f:
            f.write(report)

        # 保存交易日志
        pd.DataFrame(self.trades).to_csv(os.path.join(output_dir, 'trades.csv'), index=False)

        # 绘制权益曲线
        times, equities = zip(*self.equity_curve)
        plt.figure(figsize=(12, 6))
        plt.plot(times, equities)
        plt.title('Equity Curve')
        plt.xlabel('Time')
        plt.ylabel('Balance')
        plt.savefig(os.path.join(output_dir, 'equity_curve.png'))
        plt.close()

        # 新增：绘制收盘价格图，并标记买入/卖出/平仓点
        buy_times, buy_prices = [], []
        sell_times, sell_prices = [], []
        close_times, close_prices = [], []
        for trade in self.trades:
            if trade['type'] == 'open_buy':
                buy_times.append(trade['time'])
                buy_prices.append(trade['price'])
            elif trade['type'] == 'open_sell':
                sell_times.append(trade['time'])
                sell_prices.append(trade['price'])
            elif trade['type'].startswith('close_'):
                close_times.append(trade['time'])
                close_prices.append(trade['price'])

        plt.figure(figsize=(12, 6))
        plt.plot(self.history_data.index, self.history_data['close'], label='Close Price', color='black')
        plt.scatter(buy_times, buy_prices, color='red', label='Buy (Open)', marker='^', s=50)
        plt.scatter(sell_times, sell_prices, color='green', label='Sell (Open)', marker='v', s=50)
        plt.scatter(close_times, close_prices, color='blue', label='Close', marker='o', s=50)
        plt.title('Close Price with Trade Points')
        plt.xlabel('Time')
        plt.ylabel('Close Price')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'trade_points.png'))
        plt.close()

    def calculate_max_drawdown(self):
        """计算最大回撤"""
        equities = np.array([e for _, e in self.equity_curve])
        peak = np.maximum.accumulate(equities)
        drawdown = (equities - peak) / peak
        return np.min(drawdown) * 100 if len(drawdown) > 0 else 0

def main():
    parser = argparse.ArgumentParser(description="RSI 策略回测")
    parser.add_argument('--symbol', default='USOILm', help='交易品种 (默认: USOILM)')
    parser.add_argument('--time_frame', type=int, default=5)
    parser.add_argument('--days', type=int, default=30, help='历史数据天数')
    parser.add_argument('--initial_balance', type=float, default=10000.0, help='初始余额')
    parser.add_argument('--volume', type=float, default=1, help='交易量')
    parser.add_argument('--lever', type=float, default=10) # 杠杆
    parser.add_argument('--dynamic_sl', type=int, default=1, help='启用动态止损')
    parser.add_argument('--dynamic_tp', type=int, default=1, help='启用动态止盈')
    parser.add_argument('--buy_rsi', type=int, default=35, help='买入 RSI 阈值')
    parser.add_argument('--sell_rsi', type=int, default=65, help='卖出 RSI 阈值')
    parser.add_argument('--periods', type=int, default=14) # 计算rsi和atr的周期
    parser.add_argument('--atr_threshold', type=float, default=0.1, help='ATR 最小阈值')
    parser.add_argument('--stop_loss_cooling', type=int, default=45*60, help='止损冷静期 (秒)')
    parser.add_argument('--take_profit_cooling', type=int, default=45*60, help='止盈冷静期 (秒)')
    parser.add_argument('--max_stop_loss', type=float, default=-4.8, help='最大止损金额')
    parser.add_argument('--min_take_profit', type=float, default=1.5, help='最小止盈金额')
    parser.add_argument('--dynamic_tp_threshold', type=float, default=-0.2, help='动态止盈波动阈值')
    parser.add_argument('--output_dir', default='backtest_results', help='输出目录')
    args = parser.parse_args()

    backtester = Backtester(
        symbol=args.symbol,
        time_frame=args.time_frame,
        initial_balance=args.initial_balance,
        volume=args.volume,
        lever=args.lever,
        dynamic_sl_enabled=args.dynamic_sl,
        dynamic_tp_enabled=args.dynamic_tp,
        buy_rsi=args.buy_rsi,
        sell_rsi=args.sell_rsi,
        periods=args.periods,
        atr_threshold=args.atr_threshold,
        stop_loss_cooling=args.stop_loss_cooling,
        take_profit_cooling=args.take_profit_cooling,
        max_stop_loss=args.max_stop_loss,
        min_take_profit=args.min_take_profit,
        dynamic_tp_threshold=args.dynamic_tp_threshold
    )
    backtester.load_data(days=args.days, timeframe=args.time_frame)
    backtester.run_backtest()
    backtester.generate_report(output_dir=args.output_dir)

if __name__ == "__main__":
    main()