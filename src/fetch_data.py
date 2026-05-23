import pandas as pd
from pathlib import Path

# Đường dẫn tự động tìm thư mục gốc dựa trên vị trí file fetch_data.py
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

# Đảm bảo thư mục tồn tại
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Khai báo mã dữ liệu trên FRED
series_dict = {
    'NASDAQ100': 'NASDAQ-100',
    'CPIAUCSL': 'CPI',
    'FEDFUNDS': 'Lãi suất Quỹ Liên bang (FFR)'
}

def download_fred_data():
    print("Đang tải dữ liệu trực tiếp từ URL của FRED...")

    for series_code, name in series_dict.items():
        try:
            # Link tải file CSV trực tiếp từ biểu đồ của FRED
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_code}"
            
            # Dùng pandas đọc trực tiếp từ link web
            df = pd.read_csv(url)
            
            # Lưu file xuống thư mục local
            file_path = RAW_DIR / f"{series_code}.csv"
            df.to_csv(file_path, index=False)
            print(f"✓ Đã khôi phục thành công: {name} -> data/raw/{file_path.name}")
            
        except Exception as e:
            print(f"⚠ Lỗi khi tải {series_code}: {e}")

    print("\nHoàn tất!")

if __name__ == "__main__":
    download_fred_data()