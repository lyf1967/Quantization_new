from getmac import get_mac_address
from datetime import datetime, timezone, timedelta
import requests
import pytz

# Define expiration date (e.g., one month from now)
BEIJING_TZ = timezone(timedelta(hours=8))
EXPIRATION_DATE = datetime(2026, 2, 4, 14, 18, tzinfo=BEIJING_TZ)  # Adjust this date as needed


# MAC address whitelist
MAC_WHITELIST = ["70:A6:CC:78:2B:70",  # 本机
                 "50:EB:71:5F:94:FC",  # 悟
                 "CC:F9:E4:B6:37:85", # 悟
                 # "74:56:3C:4A:09:2A", # 悟朋友
                 ]  # Add authorized MAC addresses here

def get_api_time():
    response = requests.get("http://worldtimeapi.org/api/timezone/Asia/Shanghai")
    data = response.json()
    return datetime.fromisoformat(data["datetime"]).astimezone(BEIJING_TZ)


def check_restrictions():
    # Check expiration date
    # 获取 UTC 时间
    beijing_tz = pytz.timezone('Asia/Shanghai')
    # 获取带时区信息的当前时间
    beijing_time = datetime.now(beijing_tz)


    if beijing_time > EXPIRATION_DATE:
        print("软件已过期，请联系管理员获取新版本")
        return False

    # Check MAC address
    current_mac = get_mac_address().upper()
    if current_mac not in MAC_WHITELIST:
        print(f"此设备未授权，当前MAC地址：{current_mac}")
        return False

    print(f"授权验证通过，MAC地址：{current_mac}")
    return True