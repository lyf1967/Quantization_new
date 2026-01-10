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
                 stop_loss_cooling=45*60, # 默认值30*60
                 take_profit_cooling=45*60, # 默认值5*60
                 max_stop_loss=-4.8, # 默认值-5
                 min_take_profit=1.5,  # 默认值1.5
                 dynamic_tp_threshold=-0.2, # 默认值-0.2
                 monitor_time_gap=59, # 默认值59s
                 time_frame=5  # 默认值5
                 ):
        self.handler = handler
        self.max_positions = 1
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
        self.max_stop_loss = max_stop_loss  # 0.01手最大允许亏损金额  基线值：-20
        self.min_take_profit = min_take_profit  # 0.01手止盈利润  基线值：2
        self.sell_rsi = sell_rsi  # 做空rsi阈值，基线值：65
        self.buy_rsi = buy_rsi  # 做多rsi阈值，基线值：35
        self.periods = periods
        self.atr_threshold = atr_threshold
        self.monitor_time_gap = monitor_time_gap # 每个*s判断止盈止损
        self.time_frame = time_frame
        self.max_profit_dict = {}
        self.lock = threading.Lock()  # 新增：线程锁，保护时间变量的读写

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
            print(f"hyc not start_dynamic_sl_monitor")
            return

        def monitor_positions():
            positions_num = len(self.handler.get_open_positions())
            print(f"positions_num = {positions_num}")
            if positions_num == 0:
                self.max_profit_dict = {}

            while not self.stop_dynamic_sl_flag:
                try:
                    positions = self.handler.get_open_positions()
                    current_time = datetime.now()

                    # 检查接近休市时间，无论盈亏平仓所有持仓
                    if self.is_close_to_market_close() and len(positions) > 0:
                        print(f"{datetime.now()}: 接近休市时间，平仓所有持仓")
                        for pos in positions:
                            if self.handler.close_specific_position(pos['symbol'], pos['ticket']):
                                print(f"{datetime.now()}: 订单 {pos['ticket']} 已平仓（休市前）")
                                if pos['ticket'] in self.position_open_times:
                                    del self.position_open_times[pos['ticket']]
                                if pos['ticket'] in self.max_profit_dict:
                                    del self.max_profit_dict[pos['ticket']]

                    for ii in range(len(positions)):
                        pos = positions[ii]
                        ticket = pos['ticket']
                        symbol = pos['symbol']
                        profit = pos['profit'] # 利润
                        print(f"pos: {pos}")
                        volume = pos["volume"]  # 手数
                        # 止损 止盈计算
                        stop_loss = volume / 0.01 * self.max_stop_loss
                        take_profit = volume / 0.01 * self.min_take_profit
                        max_profit = self.max_profit_dict.get(ticket, 0)
                        if profit > max_profit:
                            self.max_profit_dict[ticket] = profit
                            print(f"update max profit from {max_profit} to {profit}")
                            max_profit = profit
                        print(f"max_profit_list:{max_profit}, ii:{ii}")
                        print(f"volume = {volume}, stop_loss = {stop_loss}, take_profit = {take_profit}, max_profit = {max_profit}")
                        open_time = self.position_open_times.get(ticket, {}).get('open_time')
                        prev_price = self.position_open_times.get(ticket, {}).get('prev_price', 0)
                        current_price = pos.get('current_price', 0)
                        tp = pos.get('tp', 0)

                        print(f"open_time = {open_time}")
                        # 动态止损：亏损超10分钟
                        print(f"profit < stop_loss and self.dynamic_sl_enabled: {profit < stop_loss and self.dynamic_sl_enabled}")
                        print(f"profit > take_profit and self.dynamic_tp_enabled: {profit > take_profit and self.dynamic_tp_enabled}")
                        if profit < stop_loss and self.dynamic_sl_enabled:
                            if self.handler.close_specific_position(symbol, ticket):
                                print(f"{datetime.now()}: 订单 {ticket} 已动态止损")
                                # self.last_dynamic_sl_time = datetime.now()
                                self.last_dynamic_stop_loss_time = datetime.now()
                                print(f"{datetime.now()}: 因动态止损进入30秒冷静期")
                                if ticket in self.position_open_times:
                                    del self.position_open_times[ticket]
                                if ticket in self.max_profit_dict.keys():
                                    del self.max_profit_dict[ticket]
                            else:
                                print(f"{datetime.now()}: 订单 {ticket} 动态止损失败，错误代码: {mt5.last_error()}")
                        # 止盈平仓
                        elif profit > take_profit and self.dynamic_tp_enabled:
                            # 动态止盈：基于波动率
                            if self.dynamic_tp_enabled:
                                if max_profit > 0:
                                    profit_change_pct = (profit - max_profit) / max_profit
                                    print(f"profit_change_pct = {profit_change_pct}")
                                    print(f"{datetime.now()}: 订单 {ticket} 动态止盈检测 - 当前利润: {profit:.2f}, "
                                          f"最大利润: {max_profit:.2f}, 波动: {profit_change_pct*100:.4f}%")
                                    if profit_change_pct <= self.dynamic_tp_threshold:  # 下跌 10%
                                        print(f"{datetime.now()}: 订单 {ticket} 达到动态止盈条件，触发止盈平仓")
                                        if self.handler.close_specific_position(symbol, ticket):
                                            print(f"{datetime.now()}: 订单 {ticket} 已动态止盈平仓")
                                            self.last_dynamic_take_profit_time = datetime.now()
                                            print(f"{datetime.now()}: 因动态止盈进入300秒冷静期")
                                            if ticket in self.position_open_times:
                                                del self.position_open_times[ticket]
                                            if ticket in self.max_profit_dict.keys():
                                                del self.max_profit_dict[ticket]
                                        else:
                                            print(f"{datetime.now()}: 订单 {ticket} 动态止盈平仓失败，错误代码: {mt5.last_error()}")
                            else:
                                # 静态止盈：基于用户设置的 tp
                                if tp > 0:
                                    price_diff = abs(current_price - tp)
                                    if price_diff < 100:  # 固定 100 点阈值
                                        print(f"{datetime.now()}: 订单 {ticket} 止盈检测 - 当前价格: {current_price:.2f}, "
                                            f"止盈点位: {tp:.2f}, 差值: {price_diff:.2f}")
                                        print(f"{datetime.now()}: 订单 {ticket} 达到止盈点位，触发止盈平仓")
                                        if self.handler.close_specific_position(symbol, ticket):
                                            print(f"{datetime.now()}: 订单 {ticket} 已止盈平仓")
                                            self.last_dynamic_take_profit_time = datetime.now()
                                            print(f"{datetime.now()}: 因止盈平仓进入30秒冷静期")
                                            if ticket in self.position_open_times:
                                                del self.position_open_times[ticket]
                                        else:
                                            print(f"{datetime.now()}: 订单 {ticket} 止盈平仓失败，错误代码: {mt5.last_error()}")
                                    else:
                                        print(f"{datetime.now()}: 订单 {ticket} 未达到止盈 - 当前价格: {current_price:.2f}, "
                                            f"止盈点位: {tp:.2f}, 差值: {price_diff:.2f}")
                        else:
                            print(f"do nothing!")

                        print(f"time sleep {self.monitor_time_gap}s")
                        time.sleep(self.monitor_time_gap)

                except Exception as e:
                    print(f"{datetime.now()}: 监控错误：{e}")
                    time.sleep(10)

        self.stop_dynamic_sl_flag = False
        self.dynamic_sl_thread = threading.Thread(target=monitor_positions, daemon=True)
        self.dynamic_sl_thread.start()
        print(f"{datetime.now()}: 动态止损和止盈监控启动 - 品种: {symbol}")

    def stop_dynamic_sl_monitor(self):
        """停止动态止损和止盈监控"""
        self.stop_dynamic_sl_flag = True
        if self.dynamic_sl_thread:
            self.dynamic_sl_thread.join()
            self.dynamic_sl_thread = None
            print(f"{datetime.now()}: 动态止损和止盈监控停止")

    def calculate_rsi(self, data, periods=30):
        """计算RSI指标（周期默认为30）"""
        close = data['close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_atr(self, data, periods=14):
        """计算平均真实波幅（ATR）"""
        high = data['high']
        low = data['low']
        close = data['close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=periods).mean()
        return atr

    def get_ohlc_data(self, symbol, timeframe=mt5.TIMEFRAME_M5, count=50):
        print(f"timeframe:{timeframe}")
        # mt5.TIMEFRAME_M1 表示1分钟线
        if not mt5.initialize():
            print(f"{datetime.now()}: MT5初始化失败，无法获取K线数据")
            return pd.DataFrame()

        # 0：从当前最新K线开始获取（0表示最新位置）； count：获取的K线数量（默认为50条）
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)  # 获取历史K线数据，返回为numpy.array
        if rates is None or len(rates) < self.periods + 1:
            print(
                f"{datetime.now()}: 无法获取足够K线数据 - 品种: {symbol}, 获取行数: {len(rates) if rates is not None else 0}")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        # 将时间戳（Unix秒）转为可读时间（unit='s'）
        df['time'] = pd.to_datetime(df['time'], unit='s')
        # 设置时间为索引
        df.set_index('time', inplace=True)

        # 验证 close 列
        # open（开盘价）、high（最高价）、low（最低价）、close（收盘价）
        # tick_volume 记录的是特定时间段内价格跳动的次数，而非成交量（如合约手数）;Tick是价格的最小变动单位，例如期货合约价格从 1000.5 变为 1000.6 即构成一次跳动（tick）
        # real_volume表示真正的成交量
        # all(...)：当所有必需列均存在时返回True，否则返回False
        required_columns = ['open', 'high', 'low', 'close', 'tick_volume']
        if not all(col in df.columns for col in required_columns):
            print(f"{datetime.now()}: K线数据缺少必要列 - 品种: {symbol}, 现有列: {list(df.columns)}")
            return pd.DataFrame()

        # 清洗数据
        # 转化为数值类型； errors='coerce'：非数值内容转为NaN（如空字符串、文本）
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        # .isnull().any()：检查是否存在NaN ； np.isfinite()：排除±inf（无穷大）或NaN
        if df['close'].isnull().any() or not np.all(np.isfinite(df['close'])):
            print(
                f"{datetime.now()}: K线数据无效 - 品种: {symbol}, close列包含NaN或非有限值: {df['close'].tail(5).to_dict()}")
            df['close'] = df['close'].fillna(method='ffill')  # ffill（前向填充）：用最近有效值覆盖NaN
            if df['close'].isnull().any():
                print(f"{datetime.now()}: 清洗后仍包含NaN - 品种: {symbol}")
                return pd.DataFrame()

        print(
            f"{datetime.now()}: 获取K线数据成功 - 品种: {symbol}, 行数: {len(df)}, 前两行: {df['close'].tail(2).to_dict()}")
        return df[required_columns]

    def get_signal(self, data, symbol=None, is_reload_data=True, is_back_test=False):
        """生成交易信号（基于RSI(30)：>70做空，<30做多）"""
        if is_reload_data:
            data = self.get_ohlc_data(symbol, timeframe=self.time_frame, count=self.periods * 2)

        if len(data) < self.periods:  # 需要至少31根K线来计算30周期RSI
            print(f"{datetime.now()}: 数据不足，无法生成信号, 数据长度: {len(data)}")
            return None

        if not is_back_test:
            # 检查是否允许交易（开市且周一开市后满70分钟）
            if not self.is_trading_allowed():
                return None

        # 调试日志：记录数据状态
        if not is_back_test:
            print(f"{datetime.now()}: 数据形状: {data.shape}, 列: {list(data.columns)}")
            print(f"{datetime.now()}: 前两行数据: {data[['close']].tail(2).to_dict()}")
        else:
            print(f"{data.index[-1]}: 数据形状: {data.shape}, 列: {list(data.columns)}")

        # 计算RSI
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

        # 验证RSI有效性
        if not 0 <= current_rsi <= 100:
            print(f"{datetime.now()}: RSI值无效 - RSI: {current_rsi}")
            return None

        # 日志记录
        signal_type = '买入' if current_rsi < self.buy_rsi else '卖出' if current_rsi > self.sell_rsi else None
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

        positions = mt5.positions_get()

        if len(positions) >= self.max_positions:
            if not is_back_test:
                print(f"{datetime.now()}: 持仓数量达到上限 ({self.max_positions})，暂停开仓")
            else:
                print(f"{data.index[-1]}: 持仓数量达到上限 ({self.max_positions})，暂停开仓")
            return None

        # 使用传入的symbol或默认值
        if symbol is None:
            symbol = self.default_symbol
            print(f"{datetime.now()}: 未提供交易品种，使用默认值: {symbol}")

        # 检查是否处于冷静期
        if self.is_in_cooling_period():
            return None  # 冷静期内不生成信号

        # 开仓逻辑：RSI(30) > 70 做空，< 30 做多
        if current_rsi > self.sell_rsi:
            if not is_back_test:
                print(f"{datetime.now()}: 卖出信号 - 品种: {symbol}, RSI: {current_rsi:.2f}")
            else:
                print(f"{data.index[-1]}: 卖出信号 - 品种: {symbol}, RSI: {current_rsi:.2f}")
            return 'sell'
        elif current_rsi < self.buy_rsi:
            if not is_back_test:
                print(f"{datetime.now()}: 卖出信号 - 品种: {symbol}, RSI: {current_rsi:.2f}")
            else:
                print(f"{data.index[-1]}: 卖出信号 - 品种: {symbol}, RSI: {current_rsi:.2f}")
            return 'buy'

        return None


def get_signal(data, symbol, handler=None, dynamic_sl_enabled=False, dynamic_tp_enabled=False):
    """顶级的get_signal函数，供MT5Handler调用"""
    strategy = RSIHighFreqXAUUSD(handler=handler, dynamic_sl_enabled=dynamic_sl_enabled, dynamic_tp_enabled=dynamic_tp_enabled)
    return strategy.get_signal(data, symbol)