from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLineEdit, QPushButton, QFileDialog, 
                            QLabel, QDoubleSpinBox, QTextEdit, QStatusBar, QToolBar, QMainWindow, 
                            QSizePolicy, QFrame, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QIcon, QPalette, QColor
from mt5.mt5_handler import MT5Handler
import os
import sys
from io import StringIO
import time
import importlib.util

class TradingTab(QMainWindow):
    def __init__(self, mt5_handler: MT5Handler):
        super().__init__()
        self.mt5_handler = mt5_handler
        self.initial_balance = self.mt5_handler.get_initial_balance()
        print(f"自动交易选项卡 - 初始余额: {self.initial_balance}")
        self.setWindowTitle("XAI Trading Platform")
        self.setMinimumSize(1000, 700)
        self.init_ui()
        
        self.stdout = sys.stdout
        self.log_stream = StringIO()
        sys.stdout = self.log_stream
        
        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self.update_log_display)
        self.log_timer.start(1000)
        
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(5000)

        self.order_timer = QTimer()
        self.order_timer.timeout.connect(self.update_orders)
        self.order_timer.start(2000)

        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_trade_stats)
        self.stats_timer.start(5000)

        self.update_trade_stats()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2e2b44, stop:1 #1e1e2e);
            }
            QLabel, QLineEdit, QDoubleSpinBox, QComboBox, QTextEdit, QStatusBar, QTableWidget, QCheckBox {
                color: #ffffff;
                font-family: 'Segoe UI', Arial;
            }
            QLabel {
                font-size: 14px;
                padding: 5px;
                font-weight: bold;
            }
            QComboBox {
                background: #313244;
                border: 1px solid #45475a;
                border-radius: 5px;
                padding: 5px;
                min-width: 150px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QLineEdit {
                background: #313244;
                border: 1px solid #45475a;
                border-radius: 5px;
                padding: 5px;
            }
            QDoubleSpinBox {
                background: #313244;
                border: 1px solid #45475a;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #89b4fa, stop:1 #1e66d8);
                color: #ffffff;
                border: none;
                border-radius: 5px;
                padding: 8px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #a6c1ff, stop:1 #3a81eb);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #6d9eff, stop:1 #104e9b);
            }
            QTextEdit {
                background: #313244;
                border: 1px solid #45475a;
                border-radius: 5px;
                font-family: 'Consolas';
                font-size: 12px;
                padding: 10px;
            }
            QTableWidget {
                background: #313244;
                border: 1px solid #45475a;
                border-radius: 5px;
                gridline-color: #585b70;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background: #45475a;
                color: #89b4fa;
                padding: 5px;
                border: 1px solid #585b70;
            }
            QStatusBar {
                background: #2e2b44;
                font-size: 12px;
            }
            QFrame {
                border: 1px solid #45475a;
                border-radius: 5px;
                background: #1e1e2e;
                margin: 5px;
            }
            QTabWidget::pane {
                border: 1px solid #45475a;
                background: #1e1e2e;
                border-radius: 5px;
            }
            QTabBar::tab {
                background: #313244;
                color: #ffffff;
                padding: 5px 10px;
                border: 1px solid #45475a;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background: #89b4fa;
                color: #ffffff;
            }
            QCheckBox {
                padding: 5px;
            }
        """)

        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setStyleSheet("background: #2e2b44; border: none;")
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        tab_widget = QTabWidget()
        trading_tab = QWidget()
        orders_tab = QWidget()

        trading_layout = QVBoxLayout(trading_tab)
        symbol_layout = QHBoxLayout()
        symbol_layout.addWidget(QLabel("交易品种："))
        self.symbol_search = QLineEdit()
        self.symbol_search.setPlaceholderText("搜索交易品种...")
        self.symbol_search.textChanged.connect(self.filter_symbols)
        self.symbol_combo = QComboBox()
        self.symbol_combo.setMinimumWidth(200)
        symbol_layout.addWidget(self.symbol_search)
        symbol_layout.addWidget(self.symbol_combo)
        symbol_widget = QFrame()
        symbol_widget.setLayout(symbol_layout)

        strategy_layout = QHBoxLayout()
        strategy_layout.addWidget(QLabel("交易策略："))
        self.strategy_path = QLineEdit()
        self.strategy_path.setReadOnly(True)
        self.browse_button = QPushButton("浏览")
        self.browse_button.clicked.connect(self.browse_strategy)
        strategy_layout.addWidget(self.strategy_path)
        strategy_layout.addWidget(self.browse_button)
        strategy_widget = QFrame()
        strategy_widget.setLayout(strategy_layout)

        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("交易量："))
        self.volume = QDoubleSpinBox()
        self.volume.setRange(0.01, 100.0)
        self.volume.setValue(0.1)
        params_layout.addWidget(self.volume)
        params_layout.addWidget(QLabel("止损点位："))
        self.sl = QDoubleSpinBox()
        self.sl.setRange(0, 50000)
        self.sl.setValue(50)
        params_layout.addWidget(self.sl)
        params_layout.addWidget(QLabel("止盈点位："))
        self.tp = QDoubleSpinBox()
        self.tp.setRange(0, 50000)
        self.tp.setValue(100)
        params_layout.addWidget(self.tp)
        params_layout.addWidget(QLabel("动态止损："))
        self.dynamic_sl = QCheckBox("启用")
        params_layout.addWidget(self.dynamic_sl)
        params_layout.addWidget(QLabel("动态止盈："))
        self.dynamic_tp = QCheckBox("启用")
        params_layout.addWidget(self.dynamic_tp)
        params_widget = QFrame()
        params_widget.setLayout(params_layout)

        control_layout = QHBoxLayout()
        self.start_button = QPushButton("开始自动交易")
        self.start_button.clicked.connect(self.start_trading)
        self.stop_button = QPushButton("停止自动交易")
        self.stop_button.clicked.connect(self.stop_trading)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_widget = QFrame()
        control_widget.setLayout(control_layout)

        stats_layout = QHBoxLayout()
        self.initial_balance_label = QLabel(f"初始余额: ￥{self.initial_balance:.2f}")
        self.win_rate_label = QLabel("胜率: 0%")
        self.total_profit_label = QLabel("总盈亏: $0.00")
        self.history_profit_label = QLabel("历史交易盈亏: $0.00")
        self.trade_count_label = QLabel("交易次数: 0")
        stats_layout.addWidget(self.initial_balance_label)
        stats_layout.addWidget(self.win_rate_label)
        stats_layout.addWidget(self.total_profit_label)
        stats_layout.addWidget(self.history_profit_label)
        stats_layout.addWidget(self.trade_count_label)
        stats_widget = QFrame()
        stats_widget.setLayout(stats_layout)

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMinimumHeight(200)

        trading_layout.addWidget(symbol_widget)
        trading_layout.addWidget(strategy_widget)
        trading_layout.addWidget(params_widget)
        trading_layout.addWidget(control_widget)
        trading_layout.addWidget(stats_widget)
        trading_layout.addWidget(self.log_display)

        orders_layout = QVBoxLayout(orders_tab)
        self.order_table = QTableWidget()
        self.order_table.setColumnCount(8)
        self.order_table.setHorizontalHeaderLabels(["订单号", "品种", "类型", "开仓价格", "当前价格", "止损", "止盈", "盈亏"])
        self.order_table.setSortingEnabled(True)
        self.order_table.setMinimumHeight(300)
        self.order_table.setStyleSheet("""
            QTableWidget::item:selected {
                background-color: #585b70;
            }
        """)
        self.order_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        order_control_layout = QHBoxLayout()
        self.refresh_button = QPushButton("刷新订单")
        self.refresh_button.clicked.connect(self.update_orders)
        self.close_order_button = QPushButton("关闭选中订单")
        self.close_order_button.clicked.connect(self.close_selected_order)
        order_control_layout.addWidget(self.refresh_button)
        order_control_layout.addWidget(self.close_order_button)
        order_control_widget = QFrame()
        order_control_widget.setLayout(order_control_layout)

        orders_layout.addWidget(self.order_table)
        orders_layout.addWidget(order_control_widget)

        tab_widget.addTab(trading_tab, "交易")
        tab_widget.addTab(orders_tab, "持有订单")
        layout.addWidget(tab_widget)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("状态: 停止")
        self.account_label = QLabel("账户: 未连接")
        self.status_bar.addPermanentWidget(self.status_label)
        self.status_bar.addPermanentWidget(self.account_label)

        self.load_symbols()
        self.update_status()

    def load_symbols(self):
        symbols = self.mt5_handler.get_symbols()
        self.symbol_combo.clear()
        if not symbols:
            self.status_bar.showMessage("无法加载交易品种，请检查MT5连接", 5000)
            return
        for symbol in symbols:
            self.symbol_combo.addItem(symbol)
        index = self.symbol_combo.findText("XAUUSD")
        if index >= 0:
            self.symbol_combo.setCurrentIndex(index)

    def filter_symbols(self):
        search_text = self.symbol_search.text().lower()
        self.symbol_combo.clear()
        symbols = self.mt5_handler.get_symbols()
        for symbol in symbols:
            if search_text in symbol.lower():
                self.symbol_combo.addItem(symbol)

    def browse_strategy(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择策略文件", "", "Python Files (*.py)")
        if file_path:
            try:
                spec = importlib.util.spec_from_file_location("strategy", file_path)
                strategy_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(strategy_module)
                if hasattr(strategy_module, 'UltraHighFreqXAUUSD') or hasattr(strategy_module, 'HighFreqXAUUSD') or hasattr(strategy_module, 'get_signal'):
                    self.strategy_path.setText(file_path)
                    self.status_bar.showMessage("策略文件加载成功", 5000)
                else:
                    self.status_bar.showMessage("无效策略文件：缺少支持的类或get_signal函数", 5000)
            except Exception as e:
                self.status_bar.showMessage(f"加载策略文件失败：{str(e)}", 5000)

    def start_trading(self):
        if self.mt5_handler.trading_thread and self.mt5_handler.trading_thread.is_alive():
            self.status_bar.showMessage("自动交易已在运行，请先停止", 5000)
            return
        strategy_path = self.strategy_path.text()
        symbol = self.symbol_combo.currentText()
        volume = self.volume.value()
        sl = self.sl.value() if not self.dynamic_sl.isChecked() else 0  # 动态止损时SL设为0
        tp = self.tp.value() if not self.dynamic_tp.isChecked() else 0  # 动态止盈时TP设为0
        dynamic_sl = self.dynamic_sl.isChecked()
        dynamic_tp = self.dynamic_tp.isChecked()

        if not strategy_path:
            self.status_bar.showMessage("请选择策略文件", 5000)
            return
        if not symbol:
            self.status_bar.showMessage("请选择交易品种", 5000)
            return
        if not self.mt5_handler.symbol_info(symbol):
            self.status_bar.showMessage(f"无效交易品种: {symbol}，请检查MT5终端", 5000)
            return
        if not dynamic_sl and not dynamic_tp and sl <= 0 and tp <= 0:
            self.status_bar.showMessage("请设置止损或止盈点位，或启用动态止损/止盈", 5000)
            return
        if self.mt5_handler.start_auto_trading(strategy_path, symbol, volume, sl, tp, dynamic_sl=dynamic_sl, dynamic_tp=dynamic_tp):
            self.status_label.setText(f"状态: 运行中 - 品种: {symbol}")
            self.trade_count_label.setText("交易次数: 0")
            self.update_trade_stats()
            self.status_bar.showMessage(f"自动交易启动成功 - 品种: {symbol}, 动态止损: {'启用' if dynamic_sl else '禁用'}, 动态止盈: {'启用' if dynamic_tp else '禁用'}", 5000)

    def stop_trading(self):
        self.mt5_handler.stop_auto_trading()
        self.status_label.setText("状态: 停止")
        self.update_trade_stats()

    def update_log_display(self):
        self.log_stream.seek(0)
        log_text = self.log_stream.read()
        if log_text:
            log_lines = self.log_display.toPlainText().split('\n') + log_text.split('\n')
            log_lines = log_lines[-10:]
            self.log_display.clear()
            for line in log_lines:
                if line:
                    timestamp = line[:19]
                    content = line[19:].strip()
                    self.log_display.append(f"<font color='#89b4fa'>{timestamp}</font> {content}")
            self.log_stream.seek(0)
            self.log_stream.truncate(0)

    def update_status(self):
        account_info = self.mt5_handler.get_account_info()
        if account_info:
            total_profit = account_info['balance'] - self.initial_balance
            self.account_label.setText(f"账户: 余额 ${account_info['balance']:.2f}, 总盈亏 ${total_profit:.2f}, 浮动盈亏 ${account_info['profit']:.2f}")
        else:
            self.account_label.setText("账户: 未连接")

    def update_orders(self):
        orders = self.mt5_handler.get_open_positions()
        self.order_table.setRowCount(len(orders))
        for row, order in enumerate(orders):
            self.order_table.setItem(row, 0, QTableWidgetItem(str(order["ticket"])))
            self.order_table.setItem(row, 1, QTableWidgetItem(order["symbol"]))
            self.order_table.setItem(row, 2, QTableWidgetItem(order["type"].upper()))
            self.order_table.setItem(row, 3, QTableWidgetItem(f"{order['open_price']:.2f}"))
            self.order_table.setItem(row, 4, QTableWidgetItem(f"{order['current_price']:.2f}"))
            self.order_table.setItem(row, 5, QTableWidgetItem(f"{order.get('sl', 0):.2f}"))
            self.order_table.setItem(row, 6, QTableWidgetItem(f"{order.get('tp', 0):.2f}"))
            profit_item = QTableWidgetItem(f"${order['profit']:.2f}")
            profit_item.setForeground(QColor('#a6e3a1' if order['profit'] >= 0 else '#f38ba8'))
            self.order_table.setItem(row, 7, QTableWidgetItem(profit_item))

    def close_selected_order(self):
        selected_items = self.order_table.selectedItems()
        if not selected_items:
            self.status_bar.showMessage("请先选择一个订单", 5000)
            return
        row = selected_items[0].row()
        ticket = int(self.order_table.item(row, 0).text())
        symbol = self.order_table.item(row, 1).text()
        if self.mt5_handler.close_specific_position(symbol, ticket):
            self.status_bar.showMessage(f"订单 {ticket} 已关闭", 5000)
            self.update_orders()
        else:
            self.status_bar.showMessage(f"关闭订单 {ticket} 失败", 5000)

    def update_trade_stats(self):
        history = self.mt5_handler.get_history()
        trade_count = len(history)
        history_profit = sum(trade['profit'] for trade in history)
        win_trades = sum(1 for trade in history if trade['profit'] > 0)
        win_rate = (win_trades / trade_count * 100) if trade_count > 0 else 0

        account_info = self.mt5_handler.get_account_info()
        total_profit = 0.0
        if account_info:
            total_profit = account_info['balance'] - self.initial_balance
            print(f"自动交易选项卡 - 当前余额: {account_info['balance']}, 初始余额: {self.initial_balance}, 总盈亏: {total_profit}")
        else:
            print("自动交易选项卡 - 无法获取账户信息，MT5可能未连接")

        self.initial_balance_label.setText(f"初始余额: ￥{self.initial_balance:.2f}")
        self.trade_count_label.setText(f"交易次数: {trade_count}")
        self.total_profit_label.setText(f"总盈亏: ${total_profit:.2f}")
        self.history_profit_label.setText(f"历史交易盈亏: ${history_profit:.2f}")
        self.win_rate_label.setText(f"胜率: {win_rate:.2f}%")

if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    mt5_handler = MT5Handler()
    window = TradingTab(mt5_handler)
    window.show()
    try:
        sys.exit(app.exec_())
    finally:
        sys.stdout = sys.__stdout__