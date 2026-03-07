import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import os

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from mt5.mt5_handler import MT5Handler

UNIT_PROFIT_INFO = {"USOILm": 10,  # 跳动1个点是10u
                    "BTCUSDm": 0.1  # 跳动1个点是0.1u
                    }


class Backtester:
    def __init__(self, symbol="USOILm",
                 start_date="2023-01-01",
                 end_date="2023-12-31",
                 timeframe="5T",
                 initial_capital=10000.0,
                 lot_size=0.01,
                 days_to_download=365,
                 data_folder=None,
                 fee=18):
        self.handler = MT5Handler()
        self.symbol = symbol
        self.start_date_str = start_date
        self.end_date_str = end_date
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.initial_capital = initial_capital
        self.balance = initial_capital  # 【修复1】增加余额追踪变量
        self.lot_size = lot_size
        self.timeframe = timeframe
        self.fee = fee
        self.unit_profit = UNIT_PROFIT_INFO[self.symbol]
        self.days_to_download = days_to_download

        self.strategy = None
        if symbol == "USOILm":
            from strategies.v7_gallon_strategy_usoil import RSIHighFreqXAUUSD
        elif symbol == "BTCUSDm":
            from strategies.v2_ma_dynamic_stop_loss_btcusdm_add_strategy import RSIHighFreqXAUUSD
        else:
            raise ValueError(f"unknown symbol: {symbol}.")

        self.strategy = RSIHighFreqXAUUSD(handler=None, current_initial_volume=lot_size)
        self.strategy.is_back_test = True
        self.equity_curve = []
        self.trades = []
        self.trade_records = []  # 新增：存储详细交易记录
        self.data_folder = data_folder
        self.current_tick_time = None  # 【修复2】保存当前 tick 时间供外部调用

        if not self.data_folder:
            self.data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_res")
        os.makedirs(self.data_folder, exist_ok=True)

    def download_tick_data(self, save_folder):
        """下载历史tick数据（最细粒度），复用MT5Handler的初始化"""
        if not mt5.initialize():
            print("MT5初始化失败")
            sys.exit(1)
        from_time = int(self.start_date.timestamp())
        to_time = int(self.end_date.timestamp())
        batch_size = 7 * 86400
        all_ticks = []
        current_start = from_time
        while current_start < to_time:
            batch_end = min(current_start + batch_size, to_time)
            ticks = mt5.copy_ticks_range(self.symbol, datetime.fromtimestamp(current_start),
                                         datetime.fromtimestamp(batch_end), mt5.COPY_TICKS_ALL)
            if ticks is None or len(ticks) == 0:
                print(
                    f"无法下载tick数据: {datetime.fromtimestamp(current_start)} 到 {datetime.fromtimestamp(batch_end)}")
                current_start = batch_end
                continue
            df_ticks = pd.DataFrame(ticks)
            df_ticks['time'] = pd.to_datetime(df_ticks['time'], unit='s')
            all_ticks.append(df_ticks)
            current_start = batch_end
        if not all_ticks:
            print("无tick数据可用")
            sys.exit(1)
        df = pd.concat(all_ticks)
        df = df[['time', 'bid', 'ask']]
        df.set_index('time', inplace=True)
        df = df[~df.index.duplicated(keep='first')]  # 去重
        print(f"下载tick数据完成: {len(df)} 条")

        save_path = os.path.join(save_folder, f"{self.symbol}_{self.start_date_str}_to_{self.end_date_str}.csv")
        df.to_csv(save_path)
        return df

    def aggregate_to_ohlc(self, df_ticks, time_frame="5T"):
        """将tick数据聚合为OHLC（根据策略timeframe）"""
        timeframe_str = {mt5.TIMEFRAME_M5: "5T",
                         mt5.TIMEFRAME_M1: "1T", }.get(self.timeframe, time_frame)  # '5T' 为5分钟
        df_ohlc = pd.DataFrame()
        df_ohlc['open'] = df_ticks['bid'].resample(timeframe_str).first()
        df_ohlc['high'] = df_ticks['bid'].resample(timeframe_str).max()
        df_ohlc['low'] = df_ticks['bid'].resample(timeframe_str).min()
        df_ohlc['close'] = df_ticks['bid'].resample(timeframe_str).last()  # 用bid模拟close
        df_ohlc['tick_volume'] = df_ticks['bid'].resample(timeframe_str).count()  # 添加tick_volume以匹配策略要求
        df_ohlc = df_ohlc.dropna()  # 去除空bar
        print(f"聚合OHLC完成: {len(df_ohlc)} 条bar")
        return df_ohlc

    def run_backtest(self):
        """运行回测，模拟交易"""
        save_path = os.path.join(self.data_folder, f"{self.symbol}_{self.start_date_str}_to_{self.end_date_str}.csv")
        if not os.path.exists(save_path):
            df_ticks = self.download_tick_data(self.data_folder)
        else:
            df_ticks = pd.read_csv(save_path, index_col=0, parse_dates=True)

        save_path_agg = save_path.replace(".csv", "_agg.csv")
        self.aggregate_to_ohlc(df_ticks, time_frame=self.timeframe).to_csv(save_path_agg)

        # 【修复1】初始化余额和初始净值
        self.balance = self.initial_capital
        current_equity = self.balance
        self.equity_curve.append((df_ticks.index[0], current_equity))

        positions = []
        current_level = 0
        max_profit = 0

        # 【修复2】动态劫持策略的冷静期判断函数，解决时间穿越问题
        def custom_cooling_check(*args):
            if self.strategy.last_dynamic_stop_loss_time is None and self.strategy.last_dynamic_take_profit_time is None:
                return False

            last_time = self.strategy.last_dynamic_stop_loss_time
            cooling_secs = self.strategy.stop_loss_cooling_time_seconds

            if self.strategy.last_dynamic_take_profit_time is not None:
                last_time = self.strategy.last_dynamic_take_profit_time
                cooling_secs = self.strategy.take_profit_cooling_time_seconds

            elapsed = (self.current_tick_time - last_time).total_seconds()
            if elapsed < cooling_secs:
                return True
            else:
                self.strategy.last_dynamic_stop_loss_time = None
                self.strategy.last_dynamic_take_profit_time = None
                return False

        # 替换原策略基于 datetime.now() 的判断方法
        self.strategy.is_in_cooling_period = custom_cooling_check

        last_check_time = df_ticks.index[0]
        current_bar_start = df_ticks.index[0].floor(self.timeframe)
        ohlc_data = []
        current_ohlc = {'time': current_bar_start,
                        'open': df_ticks['bid'][0],
                        'high': df_ticks['bid'][0],
                        'low': df_ticks['bid'][0],
                        'close': df_ticks['bid'][0],
                        'tick_volume': 0}

        for idx, row in df_ticks.iterrows():
            tick_time = idx
            self.current_tick_time = tick_time  # 【修复2】更新供冷却函数计算的时间
            tick_bid = row['bid']
            tick_ask = row['ask']
            current_ohlc['tick_volume'] += 1

            current_ohlc['high'] = max(current_ohlc['high'], tick_bid)
            current_ohlc['low'] = min(current_ohlc['low'], tick_bid)
            current_ohlc['close'] = tick_bid

            next_bar_start = current_bar_start + pd.Timedelta(self.timeframe)
            if tick_time >= next_bar_start:
                ohlc_data.append(current_ohlc.copy())

                # 【修复3】截断OHLC数组，防止随时间膨胀导致回测极大变慢
                if len(ohlc_data) > 200:
                    ohlc_data = ohlc_data[-200:]

                df_ohlc = pd.DataFrame(ohlc_data).set_index('time')

                required_len = max(self.strategy.periods, self.strategy.long_periods) + 1
                if len(df_ohlc) >= required_len:
                    data_slice = df_ohlc.iloc[-required_len:]

                    # 【修复2】删除了你原本写的错误重置事件时间的代码 (self.strategy.last_dynamic_sl_time = tick_time)
                    if not self.strategy.is_in_cooling_period():
                        signal = self.strategy.get_signal(data_slice, self.symbol, is_reload_data=False,
                                                          is_back_test=True)
                        if signal and not positions:
                            direction = signal
                            # bid: 市场上当前愿意买入合约的最高价格（买1） ；ask: 市场上当前愿意卖出合约的最低价格（卖1）
                            open_price = tick_ask if direction == 'buy' else tick_bid  # buy用ask开，sell用bid开（模拟滑点）
                            volume = self.lot_size
                            positions.append(
                                {'type': direction, 'volume': volume, 'open_price': open_price, 'open_time': tick_time})
                            print(f"{tick_time}: 开仓 {direction}，价格 {open_price}，手数 {volume}")
                            current_level = 0
                            max_profit = 0

                current_bar_start = tick_time.floor(self.timeframe)
                current_ohlc = {'time': current_bar_start, 'open': tick_bid, 'high': tick_bid, 'low': tick_bid,
                                'close': tick_bid, 'tick_volume': 1}

            if (tick_time - last_check_time).total_seconds() >= self.strategy.monitor_time_gap:
                last_check_time = tick_time
                if positions:
                    pos_type = positions[0]['type']
                    # bid: 市场上当前愿意买入合约的最高价格（买1） ；ask: 市场上当前愿意卖出合约的最低价格（卖1）
                    current_price = tick_bid if pos_type == 'buy' else tick_ask  # sell平仓用ask（买入平），buy平仓用bid（卖出平）
                    total_profit = 0
                    for pos in positions:
                        if pos['type'] == 'buy':
                            profit = (current_price - pos['open_price']) * (pos['volume']/0.01) * self.unit_profit
                        else:
                            profit = (pos['open_price'] - current_price) * (pos['volume']/0.01) * self.unit_profit
                        total_profit += profit

                    max_profit = max(max_profit, total_profit)
                    scale = self.lot_size / 0.01

                    if self.strategy.dynamic_tp_enabled and total_profit >= self.strategy.addon_tp_mins[
                        current_level] * scale:
                        profit_change = (total_profit - max_profit) / max_profit if max_profit > 0 else 0
                        if profit_change <= self.strategy.dynamic_tp_threshold:
                            self.close_positions(positions, current_price, tick_time, '动态止盈', total_profit)
                            positions = []
                            max_profit = 0
                            current_level = 0
                            self.strategy.last_dynamic_take_profit_time = tick_time
                            self.equity_curve.append((tick_time, self.balance))  # 【修复1】记录平仓后的准确净值
                            continue

                    if self.strategy.dynamic_sl_enabled and current_level < len(self.strategy.addon_loss_thresholds):
                        loss_threshold = self.strategy.addon_loss_thresholds[current_level] * scale
                        if total_profit <= loss_threshold:
                            add_times = self.strategy.add_times_list[current_level]
                            total_volume = sum(pos['volume'] for pos in positions)
                            add_volume = add_times * total_volume
                            # bid: 市场上当前愿意买入合约的最高价格（买1） ；ask: 市场上当前愿意卖出合约的最低价格（卖1）
                            add_open_price = tick_ask if pos_type == 'buy' else tick_bid
                            positions.append({'type': pos_type, 'volume': add_volume, 'open_price': add_open_price,
                                              'open_time': tick_time})
                            print(f"{tick_time}: 加仓 {pos_type}，手数 {add_volume}，级别 {current_level + 1}")
                            current_level += 1
                            self.strategy.last_dynamic_stop_loss_time = tick_time

                    # 【修复1】使用 balance 追踪浮动净值，而非 initial_capital
                    current_equity = self.balance + total_profit
                    self.equity_curve.append((tick_time, current_equity))

        if positions:
            final_bid = df_ticks['bid'][-1]
            final_ask = df_ticks['ask'][-1]
            # bid: 市场上当前愿意买入合约的最高价格（买1） ；ask: 市场上当前愿意卖出合约的最低价格（卖1）
            final_price = final_bid if positions[0]['type'] == 'buy' else final_ask
            total_profit = 0
            for pos in positions:
                if pos['type'] == 'buy':
                    profit = (final_price - pos['open_price']) * (pos['volume']/0.01) * self.unit_profit
                else:
                    profit = (pos['open_price'] - final_price) * (pos['volume']/0.01) * self.unit_profit
                total_profit += profit
            self.close_positions(positions, final_price, df_ticks.index[-1], '回测结束', total_profit)
            self.equity_curve.append((df_ticks.index[-1], self.balance))  # 【修复1】回测结束强制平仓后的收尾

        self.generate_report()

    def close_positions(self, positions, close_price, close_time, reason, profit):
        """模拟平仓"""
        direction = positions[0]['type']
        total_volume = sum(pos['volume'] for pos in positions)
        # 计算平均开仓价格和最早开仓时间
        weighted_open_price = sum(pos['open_price'] * pos['volume'] for pos in positions) / total_volume
        open_time = min(pos['open_time'] for pos in positions)
        print(
            f"{close_time}: 平仓 {direction}，价格 {close_price}，手数 {total_volume}，原因: {reason}，利润: {profit:.2f}")
        self.trades.append({'time': close_time, 'type': direction, 'profit': profit})
        # 新增：记录详细交易信息
        self.trade_records.append({
            'open_time': open_time,
            'open_price': weighted_open_price,
            'volume': total_volume,
            'close_time': close_time,
            'close_price': close_price,
            'profit': profit,
            'reason': reason
        })
        self.balance += profit  # 【新增】更新余额，将实现的profit添加到balance

    def generate_report(self):
        """生成回测报告，包括收益和回撤率"""
        equity_df = pd.DataFrame(self.equity_curve, columns=['time', 'equity'])
        equity_df.set_index('time', inplace=True)

        total_return = (equity_df['equity'][-1] - self.initial_capital) / self.initial_capital * 100
        max_drawdown = self.calculate_max_drawdown(equity_df['equity'])
        num_trades = len(self.trades)
        win_rate = (sum(1 for t in self.trades if t['profit'] > 0) / num_trades * 100) if num_trades > 0 else 0

        print("\n=== 回测报告 ===")
        print(f"总收益: {total_return:.2f}%")
        print(f"最大回撤率 (MDD): {max_drawdown:.2f}%")
        print(f"交易次数: {num_trades}")
        print(f"胜率: {win_rate:.2f}%")

        # 新增：将交易记录保存到CSV
        if self.trade_records:
            trade_df = pd.DataFrame(self.trade_records)
            trade_df.to_csv(os.path.join(self.data_folder, 'trade_records.csv'), index=False, encoding="utf-8")
            print("交易记录已保存为: " + os.path.join(self.data_folder, 'trade_records.csv'))

        # 绘制净值曲线
        plt.figure(figsize=(12, 6))
        plt.plot(equity_df.index, equity_df['equity'], label='净值曲线')
        plt.title('回测净值曲线')
        plt.xlabel('时间')
        plt.ylabel('净值 (USD)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.data_folder, 'backtest_equity_curve.png'))
        print("净值曲线已保存为: " + os.path.join(self.data_folder, 'backtest_equity_curve.png'))

    def calculate_max_drawdown(self, equity_series):
        """计算最大回撤率"""
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak
        return drawdown.min() * 100 * -1  # 转换为正百分比


if __name__ == "__main__":
    timeframe = "5T"  # "1T": 1min, "5T": 5min
    first_volume = 0.01  # 第一次下单量
    backtester = Backtester(symbol="USOILm",
                            start_date="2026-01-01",
                            end_date="2026-03-06",
                            timeframe=timeframe,
                            initial_capital=10000.0,  # 初始本金
                            lot_size=first_volume)
    backtester.run_backtest()