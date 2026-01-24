import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import time
import threading
import numpy as np
# from PyQt5.QtMultimedia.QMediaRecorder import volume


class RSIHighFreqXAUUSD:
    def __init__(self, handler=None, dynamic_sl_enabled=False, dynamic_tp_enabled=False):
        self.handler = handler
        self.max_positions = 1
        self.default_symbol = "BTCUSD"  # 默认交易品种，根据实际调整
        self.position_open_times = {}  # 存储 {ticket: {"open_time": time, "prev_price": price}}
        self.dynamic_sl_thread = None  # 动态止损线程
        self.stop_dynamic_sl_flag = False  # 停止动态止损标志
        self.last_dynamic_sl_time = None  # 记录最近一次止损或止盈触发时间
        self.last_dynamic_stop_loss_time = None
        self.last_dynamic_take_profit_time = None
        self.dynamic_sl_enabled = dynamic_sl_enabled  # 动态止损开关
        self.dynamic_tp_enabled = dynamic_tp_enabled  # 动态止盈开关
        self.take_profit_cooling_time_seconds = 5*60  # 止盈后冷静期5分钟  基线值：5*60
        self.stop_loss_cooling_time_seconds = 30*60  # 止损后冷静30分钟  基线值：30*60
        # self.dynamic_tp_threshold = -0.0005  # 动态止盈波动率阈值（下跌 0.05%）
        self.dynamic_tp_threshold = -0.2  # 动态止盈波动率阈值（下跌 10%）
        self.max_stop_loss = -20  # 0.01手最大允许亏损金额  基线值：-20
        self.min_take_profit = 2  # 0.01手止盈利润  基线值：2
        self.sell_rsi = 65  # 做空rsi阈值，基线值：65
        self.buy_rsi = 35   # 做多rsi阈值，基线值：35
        
    def is_in_cooling_period(self):
        """检查是否处于冷静期"""
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
    
    def start_dynamic_sl_monitor(self, symbol):
        """启动动态止损和止盈监控线程"""
        if not self.handler:
            print(f"{datetime.now()}: 错误：未提供MT5Handler实例")
            print(f"hyc not start_dynamic_sl_monitor")
            return
            
        def monitor_positions():
            positions_num = len(self.handler.get_open_positions())
            print(f"positions_num = {positions_num}")
            max_profit_list = [0] * positions_num
            if positions_num == 0:
                max_profit_list = []
            print(f"max_profit_list = {max_profit_list}")

            while not self.stop_dynamic_sl_flag:
                try:
                    positions = self.handler.get_open_positions()
                    current_time = datetime.now()
                    if len(positions) > 0 and len(max_profit_list) == 0:
                        max_profit_list = [0] * len(positions)

                    print(f"positions: {positions}")

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
                        print(f"max_profit_list:{max_profit_list}, ii:{ii}")
                        max_profit = max_profit_list[ii]
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
                            # time_elapsed = (current_time - open_time).total_seconds() / 60.0
                            # if time_elapsed > 10:
                            print(f"{datetime.now()}: 订单 {ticket} 亏损超过10分钟，触发动态止损")
                            if self.handler.close_specific_position(symbol, ticket):
                                print(f"{datetime.now()}: 订单 {ticket} 已动态止损")
                                # self.last_dynamic_sl_time = datetime.now()
                                self.last_dynamic_stop_loss_time = datetime.now()
                                print(f"{datetime.now()}: 因动态止损进入30秒冷静期")
                                if ticket in self.position_open_times:
                                    del self.position_open_times[ticket]
                            else:
                                print(f"{datetime.now()}: 订单 {ticket} 动态止损失败，错误代码: {mt5.last_error()}")
                        # 止盈平仓
                        elif profit > take_profit and self.dynamic_tp_enabled:
                            # 动态止盈：基于波动率
                            # if prev_price > 0:
                            if self.dynamic_tp_enabled:
                                if max_profit > 0:
                                    # price_change_pct = (current_price - prev_price) / prev_price
                                    profit_change_pct = (profit - max_profit) / max_profit
                                    print(f"profit_change_pct = {profit_change_pct}")
                                    print(f"{datetime.now()}: 订单 {ticket} 动态止盈检测 - 当前利润: {profit:.2f}, "
                                          f"最大利润: {max_profit:.2f}, 波动: {profit_change_pct*100:.4f}%")
                                    if profit_change_pct <= self.dynamic_tp_threshold:  # 下跌 10%
                                        print(f"{datetime.now()}: 订单 {ticket} 达到动态止盈条件，触发止盈平仓")
                                        if self.handler.close_specific_position(symbol, ticket):
                                            print(f"{datetime.now()}: 订单 {ticket} 已动态止盈平仓")
                                            # self.last_dynamic_sl_time = datetime.now()
                                            self.last_dynamic_take_profit_time = datetime.now()
                                            print(f"{datetime.now()}: 因动态止盈进入300秒冷静期")
                                            if ticket in self.position_open_times:
                                                del self.position_open_times[ticket]
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
                                            # self.last_dynamic_sl_time = datetime.now()
                                            self.last_dynamic_take_profit_time = datetime.now()
                                            print(f"{datetime.now()}: 因止盈平仓进入30秒冷静期")
                                            if ticket in self.position_open_times:
                                                del self.position_open_times[ticket]
                                        else:
                                            print(f"{datetime.now()}: 订单 {ticket} 止盈平仓失败，错误代码: {mt5.last_error()}")
                                    else:
                                        print(f"{datetime.now()}: 订单 {ticket} 未达到止盈 - 当前价格: {current_price:.2f}, "
                                              f"止盈点位: {tp:.2f}, 差值: {price_diff:.2f}")
                        if profit > max_profit and len(max_profit_list) > 0:
                            max_profit_list[ii] = profit
                            print(f"update max profit from {max_profit} to {profit}")

                    time.sleep(10)
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

    def get_signal(self, data, symbol=None):
        """生成交易信号（基于RSI(30)：>70做空，<30做多）"""
        if len(data) < 31:  # 需要至少31根K线来计算30周期RSI
            print(f"{datetime.now()}: 数据不足，无法生成信号, 数据长度: {len(data)}")
            return None
            
        # 调试日志：记录数据状态
        print(f"{datetime.now()}: 数据形状: {data.shape}, 列: {list(data.columns)}")
        print(f"{datetime.now()}: 前两行数据: {data[['close']].tail(2).to_dict()}")
        
        # 计算RSI
        try:
            rsi = self.calculate_rsi(data, periods=30)
            current_rsi = rsi.iloc[-1]
            if pd.isna(current_rsi):
                print(f"{datetime.now()}: RSI计算结果为NaN")
                return None
            current_rsi = float(current_rsi)
        except Exception as e:
            print(f"{datetime.now()}: RSI计算失败 - 错误: {e}")
            return None
        
        # 验证RSI有效性
        if not 0 <= current_rsi <= 100:
            print(f"{datetime.now()}: RSI值无效 - RSI: {current_rsi}")
            return None
        
        # 日志记录
        signal_type = '买入' if current_rsi < 30 else '卖出' if current_rsi > 70 else '无'
        log_message = (f"{datetime.now()}: 当前RSI: {current_rsi:.2f}, 信号: {signal_type}")
        print(log_message)
        
        account_info = mt5.account_info()
        if not account_info:
            print(f"{datetime.now()}: 错误：无法获取账户信息")
            return None
            
        positions = mt5.positions_get()
        
        if len(positions) >= self.max_positions:
            print(f"{datetime.now()}: 持仓数量达到上限 ({self.max_positions})，暂停开仓")
            return None
        
        # 使用传入的symbol或默认值
        if symbol is None:
            symbol = self.default_symbol
            print(f"{datetime.now()}: 未提供交易品种，使用默认值: {symbol}")
        
        # 检查是否处于冷静期
        if self.is_in_cooling_period():
            return None  # 冷静期内不生成信号
        
        # 开仓逻辑：RSI(30) > 70 做空，< 30 做多
        # if current_rsi > 65:
        if current_rsi > self.sell_rsi:
            print(f"{datetime.now()}: 卖出信号 - 品种: {symbol}, RSI: {current_rsi:.2f}")
            return 'sell'
        # elif current_rsi < 35:
        elif current_rsi < self.buy_rsi:
            print(f"{datetime.now()}: 买入信号 - 品种: {symbol}, RSI: {current_rsi:.2f}")
            return 'buy'
            
        return None

def get_signal(data, symbol, handler=None, dynamic_sl_enabled=False, dynamic_tp_enabled=False):
    """顶级的get_signal函数，供MT5Handler调用"""
    strategy = RSIHighFreqXAUUSD(handler=handler, dynamic_sl_enabled=dynamic_sl_enabled, dynamic_tp_enabled=dynamic_tp_enabled)
    return strategy.get_signal(data, symbol)