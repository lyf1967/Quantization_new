import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import time
import threading
import numpy as np


class RSIHighFreqXAUUSD:
    def __init__(self, handler=None, dynamic_sl_enabled=True, dynamic_tp_enabled=True,
                 buy_rsi=35,  # 默认值35
                 sell_rsi=65,  # 默认值65
                 periods=14,  # 默认值14
                 atr_threshold=0.1, # 默认值0.1
                 stop_loss_cooling=45*60, # 默认值45*60
                 take_profit_cooling=45*60, # 默认值45*60
                 max_stop_loss=-4.8, # 默认值-4.8，未使用（改为加仓）
                 min_take_profit=1.5,  # 默认值1.5，未使用（改为多级）
                 dynamic_tp_threshold=-0.2, # 默认值-0.2
                 monitor_time_gap=2, # 默认值59s
                 time_frame=5,  # 默认值5
                 long_periods=60,  # 长周期周期，默认60
                 long_atr_threshold_high=0.5,  # 长周期ATR高阈值，默认0.5
                 strict_buy_rsi=30,  # 严格买入RSI，默认30
                 strict_sell_rsi=70,  # 严格卖出RSI，默认70
                 addon_loss_thresholds=[-10, -60],  # 加仓亏损阈值（0.01手美元），第一级-10，第二级-60  [-10, -60]
                 addon_tp_mins=[1.5, 3, 0.0],  # 各级最小止盈（0.01手美元），初始1.5，第一加仓后3.0，第二后0.0  [1.5, 3.0, 0.0]
                 max_positions = 1,
                 current_initial_volume = 0.01
                 ):
        self.handler = handler
        self.max_positions = max_positions # 移除限制
        self.default_symbol = "USOILm"  # 默认交易品种，根据实际调整
        self.position_open_times = {}  # 存储 {ticket: {"open_time": time, "prev_price": price}}
        self.dynamic_sl_thread = None  # 动态止损线程
        self.stop_dynamic_sl_flag = False  # 停止动态止损标志
        self.last_dynamic_sl_time = None  # 记录最近一次止损或止盈触发时间
        self.last_dynamic_stop_loss_time = None
        self.last_dynamic_take_profit_time = None
        self.dynamic_sl_enabled = dynamic_sl_enabled  # 动态止损开关
        self.dynamic_tp_enabled = dynamic_tp_enabled  # 动态止盈开关
        self.take_profit_cooling_time_seconds = take_profit_cooling  # 止盈后冷静期5分钟  基线值：5*60
        self.stop_loss_cooling_time_seconds = stop_loss_cooling  # 止损后冷静30分钟  基线值：30*60
        self.dynamic_tp_threshold = dynamic_tp_threshold  # 动态止盈波动率阈值（下跌 10%）
        self.max_stop_loss = max_stop_loss  # 0.01手最大允许亏损金额  基线值：-20，未使用
        self.min_take_profit = min_take_profit  # 0.01手止盈利润  基线值：2，未使用
        self.sell_rsi = sell_rsi  # 做空rsi阈值，基线值：65
        self.buy_rsi = buy_rsi  # 做多rsi阈值，基线值：35
        self.periods = periods
        self.atr_threshold = atr_threshold
        self.monitor_time_gap = monitor_time_gap # 每个*s判断止盈止损
        self.time_frame = time_frame
        self.max_profit_dict = {}
        self.lock = threading.Lock()  # 新增：线程锁，保护时间变量的读写
        # 新增参数
        self.long_periods = long_periods
        self.long_atr_threshold_high = long_atr_threshold_high
        self.strict_buy_rsi = strict_buy_rsi
        self.strict_sell_rsi = strict_sell_rsi
        self.addon_loss_thresholds = addon_loss_thresholds
        self.addon_tp_mins = addon_tp_mins
        # 跟踪当前持仓组
        self.current_initial_volume = current_initial_volume  # 默认，执行交易时更新
        self.current_level = 0  # 当前加仓级别: 0初始,1第一加仓,2第二加仓
        self.current_direction = None  # 'buy' or 'sell'

    def is_in_cooling_period(self):
        """检查是否处于冷静期"""
        with self.lock:  # 新增：使用锁保护整个检查和重置逻辑
            if self.last_dynamic_stop_loss_time is None and self.last_dynamic_take_profit_time is None:
                return False
            self.last_dynamic_sl_time = self.last_dynamic_stop_loss_time
            cooling_time_seconds = self.stop_loss_cooling_time_seconds
            if self.last_dynamic_take_profit_time is not None:
                self.last_dynamic_sl_time = self.last_dynamic_take_profit_time
                cooling_time_seconds = self.take_profit_cooling_time_seconds

            print(f"self.last_dynamic_stop_loss_time: {self.last_dynamic_stop_loss_time}, self.last_dynamic_take_profit_time: {self.last_dynamic_take_profit_time}")
            print(f"cooling_time_seconds = {cooling_time_seconds}")

            current_time = datetime.now()
            elapsed = (current_time - self.last_dynamic_sl_time).total_seconds()
            if elapsed < cooling_time_seconds:
                print(f"{datetime.now()}: 处于冷静期，剩余 {cooling_time_seconds - elapsed:.0f} 秒，暂停生成信号")
                return True
            else:
                if elapsed >= cooling_time_seconds and self.last_dynamic_sl_time is not None:
                    print(f"{datetime.now()}: 冷静期结束，继续生成交易信号")
                    self.last_dynamic_sl_time = None  # 重置冷静期
                    self.last_dynamic_stop_loss_time = None
                    self.last_dynamic_take_profit_time = None

                return False
        print("lock is unuseful, please attention.")
        return None

    def is_market_open(self):
        """检查市场是否开市（北京时间，周一6:10开启，周六4:40停止）"""
        now = datetime.now()
        weekday = now.weekday()
        if weekday == 6:  # 周日
            return False
        if weekday == 5:  # 周六
            if now.hour < 4 or (now.hour == 4 and now.minute < 40):
                return True
            else:
                return False
        if weekday == 0:  # 周一
            if now.hour > 6 or (now.hour == 6 and now.minute >= 10):
                return True
            else:
                return False
        # 周二到周五全天开市
        return True

    def is_close_to_market_close(self):
        """检查是否接近休市时间（周六4:40前5分钟，即4:35开始平仓）"""
        now = datetime.now()
        if now.weekday() == 5 and now.hour == 4 and now.minute >= 35:
            return True
        return False

    def is_trading_allowed(self):
        """检查是否允许交易（开市且周一开市后满70分钟）"""
        now = datetime.now()
        if not self.is_market_open():
            print(f"{now}: 市场休市，暂停生成信号")
            return False
        if now.weekday() == 0:  # 周一
            open_time = now.replace(hour=6, minute=10, second=0, microsecond=0)
            if now >= open_time:
                elapsed = (now - open_time).total_seconds()
                if elapsed < 70 * 60:
                    print(f"{now}: 开市后未满70分钟，剩余 {70 * 60 - elapsed:.0f} 秒，暂停生成信号")
                    return False
        return True

    def start_dynamic_sl_monitor(self, symbol):
        """启动动态止损和止盈监控线程"""
        if not self.handler:
            print(f"{datetime.now()}: 错误：未提供MT5Handler实例")
            return

        def monitor_thread():
            self.monitor_dynamic_sl(symbol)

        self.stop_dynamic_sl_flag = False
        self.dynamic_sl_thread = threading.Thread(target=monitor_thread, daemon=True)
        self.dynamic_sl_thread.start()
        print(f"{datetime.now()}: 动态止损监控启动 - 品种: {symbol}")

    def get_ohlc_data(self, symbol, timeframe=mt5.TIMEFRAME_M5, count=100):
        """获取OHLC数据"""
        if symbol is None:
            symbol = self.default_symbol

        if not mt5.symbol_select(symbol, True):
            print(f"{datetime.now()}: 无法选择品种 {symbol}")
            return pd.DataFrame()

        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            print(f"{datetime.now()}: 无法获取K线数据 - 品种: {symbol}, 时间框架: {timeframe}, 数量: {count}")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        required_columns = ['open', 'high', 'low', 'close', 'tick_volume']
        if not all(col in df.columns for col in required_columns):
            print(f"{datetime.now()}: K线数据缺少必要列 - 品种: {symbol}, 现有列: {list(df.columns)}")
            return pd.DataFrame()

        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        if df['close'].isnull().any() or not np.all(np.isfinite(df['close'])):
            print(f"{datetime.now()}: K线数据无效 - 品种: {symbol}, close列包含NaN或非有限值: {df['close'].tail(5).to_dict()}")
            df['close'] = df['close'].fillna(method='ffill')
            if df['close'].isnull().any():
                print(f"{datetime.now()}: 清洗后仍包含NaN - 品种: {symbol}")
                return pd.DataFrame()

        print(f"{datetime.now()}: 获取K线数据成功 - 品种: {symbol}, 行数: {len(df)}, 前两行: {df['close'].tail(2).to_dict()}")
        return df[required_columns]

    def get_signal(self, data, symbol=None, is_reload_data=True, is_back_test=False):
        """生成交易信号（基于RSI(30)：>70做空，<30做多）"""
        if is_reload_data:
            count = max(self.periods, self.long_periods) * 2  # 确保长周期数据足够
            data = self.get_ohlc_data(symbol, timeframe=self.time_frame, count=count)

        if len(data) < max(self.periods, self.long_periods):
            print(f"{datetime.now()}: 数据不足，无法生成信号, 数据长度: {len(data)}")
            return None

        if not is_back_test:
            if not self.is_trading_allowed():
                return None

        if not is_back_test:
            print(f"{datetime.now()}: 数据形状: {data.shape}, 列: {list(data.columns)}")
            print(f"{datetime.now()}: 前两行数据: {data[['close']].tail(2).to_dict()}")
        else:
            print(f"{data.index[-1]}: 数据形状: {data.shape}, 列: {list(data.columns)}")

        try:
            rsi = self.calculate_rsi(data, periods=self.periods)
            current_rsi = rsi.iloc[-1]
            if pd.isna(current_rsi):
                print(f"{datetime.now()}: RSI计算结果为NaN")
                return None
            current_rsi = float(current_rsi)
        except Exception as e:
            print(f"{datetime.now()}: RSI计算失败 - 错误: {e}")
            return None

        atr = self.calculate_atr(data, periods=self.periods)
        print(f"atr:{atr.iloc[-1]}")
        if atr.iloc[-1] < self.atr_threshold:
            print(f"atr too small")
            return None

        # 长周期ATR检测
        long_atr = self.calculate_atr(data, periods=self.long_periods).iloc[-1]
        print(f"long_atr:{long_atr}")
        if long_atr > self.long_atr_threshold_high:
            effective_buy_rsi = self.strict_buy_rsi
            effective_sell_rsi = self.strict_sell_rsi
            print(f"长周期ATR高，使用严格RSI: buy<{effective_buy_rsi}, sell>{effective_sell_rsi}")
        else:
            effective_buy_rsi = self.buy_rsi
            effective_sell_rsi = self.sell_rsi

        if not 0 <= current_rsi <= 100:
            print(f"{datetime.now()}: RSI值无效 - RSI: {current_rsi}")
            return None

        signal_type = '买入' if current_rsi < effective_buy_rsi else '卖出' if current_rsi > effective_sell_rsi else None
        if not is_back_test:
            log_message = (f"{datetime.now()}: 当前RSI: {current_rsi:.2f}, 信号: {signal_type}")
        else:
            log_message = (f"{data.index[-1]}: 当前RSI: {current_rsi:.2f}, 信号: {signal_type}")
        print(log_message)
        if not is_back_test and (not signal_type):
            time.sleep(2)

        account_info = mt5.account_info()
        if not account_info:
            print(f"{datetime.now()}: 错误：无法获取账户信息")
            return None

        # 移除持仓数量上限检查
        positions = mt5.positions_get()

        if len(positions) >= self.max_positions:
            if not is_back_test:
                print(f"{datetime.now()}: 持仓数量达到上限 ({self.max_positions})，暂停开仓")
            else:
                print(f"{data.index[-1]}: 持仓数量达到上限 ({self.max_positions})，暂停开仓")
            return None

        if symbol is None:
            symbol = self.default_symbol
            print(f"{datetime.now()}: 未提供交易品种，使用默认值: {symbol}")

        if self.is_in_cooling_period():
            return None

        if current_rsi > effective_sell_rsi:
            if not is_back_test:
                print(f"{datetime.now()}: 卖出信号 - 品种: {symbol}, RSI: {current_rsi:.2f}")
            else:
                print(f"{data.index[-1]}: 卖出信号 - 品种: {symbol}, RSI: {current_rsi:.2f}")
            return 'sell'
        elif current_rsi < effective_buy_rsi:
            if not is_back_test:
                print(f"{datetime.now()}: 买入信号 - 品种: {symbol}, RSI: {current_rsi:.2f}")
            else:
                print(f"{data.index[-1]}: 买入信号 - 品种: {symbol}, RSI: {current_rsi:.2f}")
            return 'buy'

        return None

    def monitor_dynamic_sl(self, symbol):
        """动态止损和止盈监控逻辑（改为加仓）"""
        while not self.stop_dynamic_sl_flag:
            try:
                positions = mt5.positions_get(symbol=symbol)
                if not positions:
                    time.sleep(self.monitor_time_gap)
                    continue

                pos_type = 'buy' if positions[0].type == mt5.ORDER_TYPE_BUY else 'sell'
                if self.current_direction is None:
                    self.current_direction = pos_type
                elif self.current_direction != pos_type:
                    print(f"{datetime.now()}: 检测到混合方向持仓，跳过监控")
                    time.sleep(self.monitor_time_gap)
                    continue

                total_profit = sum(pos.profit for pos in positions)
                total_volume = sum(pos.volume for pos in positions)
                current_price = mt5.symbol_info_tick(symbol).ask if pos_type == 'buy' else mt5.symbol_info_tick(symbol).bid

                if symbol not in self.max_profit_dict:
                    self.max_profit_dict[symbol] = total_profit
                else:
                    self.max_profit_dict[symbol] = max(self.max_profit_dict[symbol], total_profit)

                scale = self.current_initial_volume / 0.01

                # 动态止盈
                if self.dynamic_tp_enabled and total_profit >= self.addon_tp_mins[self.current_level] * scale:
                    profit_change_rate = (total_profit - self.max_profit_dict[symbol]) / self.max_profit_dict[symbol] if self.max_profit_dict[symbol] > 0 else 0
                    if profit_change_rate <= self.dynamic_tp_threshold:
                        print(f"{datetime.now()}: 触发动态止盈 - 品种: {symbol}, 总利润: {total_profit:.2f}, 回落率: {profit_change_rate:.2%}")
                        for pos in positions:
                            self.handler.close_specific_position(symbol, pos.ticket)
                        self.last_dynamic_take_profit_time = datetime.now()
                        self.max_profit_dict.pop(symbol, None)
                        self.current_level = 0
                        self.current_initial_volume = 0.01
                        self.current_direction = None

                # 加仓逻辑（取代止损，最多两次）
                if self.dynamic_sl_enabled:
                    if self.current_level < len(self.addon_loss_thresholds):
                        loss_threshold = self.addon_loss_thresholds[self.current_level] * scale
                        if total_profit <= loss_threshold:
                            add_volume = 2 * total_volume
                            print(f"{datetime.now()}: 触发加仓 - 级别: {self.current_level+1}, 品种: {symbol}, 亏损: {total_profit:.2f}, 加仓手数: {add_volume}")
                            self.handler.execute_trade(symbol, add_volume, 0, 0, pos_type, self.dynamic_sl_enabled, self.dynamic_tp_enabled)  # 加仓同方向，无SL/TP
                            self.current_level += 1
                            self.last_dynamic_stop_loss_time = datetime.now()  # 加仓视为止损触发

                # 休市前平仓：仅当单数==1时
                if self.is_close_to_market_close() and len(positions) == 1:
                    print(f"{datetime.now()}: 接近休市，平仓 - 品种: {symbol}")
                    for pos in positions:
                        self.handler.close_specific_position(symbol, pos.ticket)
                    self.max_profit_dict.pop(symbol, None)
                    self.current_level = 0
                    self.current_initial_volume = 0.01
                    self.current_direction = None

                time.sleep(self.monitor_time_gap)
            except Exception as e:
                print(f"{datetime.now()}: 动态SL监控错误 - {e}")
                time.sleep(self.monitor_time_gap)

    def stop_dynamic_sl_monitor(self):
        """停止动态止损监控"""
        self.stop_dynamic_sl_flag = True
        if self.dynamic_sl_thread:
            self.dynamic_sl_thread.join()
            self.dynamic_sl_thread = None
            print(f"{datetime.now()}: 动态止损监控停止")

    def calculate_rsi(self, data, periods=14):
        """计算RSI"""
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=periods, min_periods=1).mean()
        avg_loss = loss.rolling(window=periods, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_atr(self, data, periods=14):
        """计算ATR"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=periods).mean()
        return atr


def get_signal(data, symbol, handler=None, dynamic_sl_enabled=False, dynamic_tp_enabled=False):
    """顶级的get_signal函数，供MT5Handler调用"""
    strategy = RSIHighFreqXAUUSD(handler=handler, dynamic_sl_enabled=dynamic_sl_enabled, dynamic_tp_enabled=dynamic_tp_enabled)
    return strategy.get_signal(data, symbol)