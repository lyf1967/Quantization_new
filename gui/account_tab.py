from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton
from PyQt5.QtGui import QColor
from mt5.mt5_handler import MT5Handler

class AccountTab(QWidget):
    def __init__(self, mt5_handler: MT5Handler):
        super().__init__()
        self.mt5_handler = mt5_handler
        self.initial_balance = self.mt5_handler.get_initial_balance()  # 从入金记录计算初始余额
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        self.setStyleSheet("""
            QWidget {
                background-color: #252537;
                color: #cdd6f4;
            }
            QLabel {
                font-size: 14px;
            }
            QLineEdit {
                background-color: #1e1e2e;
                color: #cdd6f4;
                border: 1px solid #45475a;
                padding: 5px;
            }
            QPushButton {
                background-color: #45475a;
                color: #89b4fa;
                border: 1px solid #585b70;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #585b70;
            }
        """)
        
        self.account_id_label = QLabel("账户ID：")
        self.account_id = QLineEdit()
        self.account_id.setReadOnly(True)
        
        self.initial_balance_label = QLabel(f"初始余额：￥{self.initial_balance:.2f}")
        
        self.balance_label = QLabel("当前余额：")
        self.balance = QLineEdit()
        self.balance.setReadOnly(True)
        
        self.profit_label = QLabel("总盈亏百分比：")
        self.profit = QLineEdit()
        self.profit.setReadOnly(True)
        
        self.floating_profit_label = QLabel("浮动盈亏：")
        self.floating_profit = QLineEdit()
        self.floating_profit.setReadOnly(True)
        
        self.update_button = QPushButton("更新账户信息")
        self.update_button.clicked.connect(self.update_account_info)
        
        layout.addWidget(self.account_id_label)
        layout.addWidget(self.account_id)
        layout.addWidget(self.initial_balance_label)
        layout.addWidget(self.balance_label)
        layout.addWidget(self.balance)
        layout.addWidget(self.profit_label)
        layout.addWidget(self.profit)
        layout.addWidget(self.floating_profit_label)
        layout.addWidget(self.floating_profit)
        layout.addWidget(self.update_button)
        layout.addStretch()
        
        self.setLayout(layout)
        self.update_account_info()
        
    def update_account_info(self):
        account_info = self.mt5_handler.get_account_info()
        if account_info:
            self.account_id.setText(str(account_info['login']))
            self.balance.setText(f"￥{account_info['balance']:.2f}")
            total_profit = account_info['balance'] - self.initial_balance
            profit_percent = (total_profit / self.initial_balance) * 100 if self.initial_balance != 0 else 0
            self.profit.setText(f"{profit_percent:.2f}% (￥{total_profit:.2f})")
            color = QColor('#a6e3a1') if profit_percent >= 0 else QColor('#f38ba8')
            self.profit.setStyleSheet(f"color: {color.name()}; background-color: #1e1e2e; border: 1px solid #45475a; padding: 5px;")
            self.floating_profit.setText(f"￥{account_info['profit']:.2f}")
            floating_color = QColor('#a6e3a1') if account_info['profit'] >= 0 else QColor('#f38ba8')
            self.floating_profit.setStyleSheet(f"color: {floating_color.name()}; background-color: #1e1e2e; border: 1px solid #45475a; padding: 5px;")
        else:
            self.account_id.setText("未连接")
            self.balance.setText("未连接")
            self.profit.setText("未连接")
            self.floating_profit.setText("未连接")