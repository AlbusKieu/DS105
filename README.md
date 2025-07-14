# Hệ Thống Gợi Ý Phim (Movie Recommendation System)

Dự án này xây dựng một hệ thống gợi ý phim sử dụng dữ liệu từ IMDB và Rotten Tomatoes. Hệ thống bao gồm các bước thu thập dữ liệu, tiền xử lý, phân tích dữ liệu, xây dựng mô hình gợi ý và giao diện web với Streamlit.


## Chức năng các file chính

- **IMDB_crawler.ipynb**: Thu thập dữ liệu phim từ IMDB, lưu ra file CSV.
- **ROTTEN_TOMATO_crawler.ipynb**: Thu thập dữ liệu phim từ Rotten Tomatoes, lưu ra file CSV.
- **main.ipynb**:
  - Đọc dữ liệu từ các file đã crawl.
  - Tiền xử lý dữ liệu: chuẩn hóa, mã hóa (LabelEncoder), xử lý thể loại phim, xử lý dữ liệu thiếu.
  - Phân tích dữ liệu khám phá (EDA): thống kê, trực quan hóa dữ liệu.
  - Xây dựng hệ thống gợi ý phim với các mô hình/thuật toán:
    - **User-based Filtering**: Sử dụng Collaborative Filtering dựa trên ma trận người dùng - phim (User-Item Matrix), tính toán độ tương đồng giữa các người dùng bằng Cosine Similarity hoặc Pearson Correlation, gợi ý phim dựa trên các phim mà người dùng tương tự đã đánh giá cao.
    - **Content-based Filtering**: Sử dụng các đặc trưng nội dung phim (thể loại, mô tả, diễn viên, đạo diễn...), vector hóa đặc trưng phim bằng TF-IDF hoặc CountVectorizer, tính toán độ tương đồng giữa các phim bằng Cosine Similarity, gợi ý phim có nội dung tương tự với các phim người dùng đã thích.
    - **Hybrid Approach**: Kết hợp cả hai phương pháp trên, có thể sử dụng trung bình cộng hoặc trọng số giữa điểm gợi ý của User-based và Content-based, hoặc sử dụng mô hình học máy (ví dụ: Linear Regression) để kết hợp các đặc trưng từ cả hai phương pháp nhằm tối ưu hóa kết quả gợi ý.
  - Đánh giá mô hình bằng các chỉ số như Precision, Recall, RMSE, v.v.
  - Xuất kết quả gợi ý cho từng người dùng hoặc theo truy vấn cụ thể.
- **app.py**: Xây dựng giao diện web với Streamlit, cho phép người dùng nhập thông tin và nhận gợi ý phim trực tiếp.

## Cấu trúc thư mục

```
├── app.py                        # Ứng dụng web Streamlit
├── main.ipynb                    # Notebook chính: tiền xử lý, EDA, xây dựng mô hình gợi ý
├── IMDB_crawler.ipynb            # Notebook crawl dữ liệu IMDB
├── ROTTEN_TOMATO_crawler.ipynb   # Notebook crawl dữ liệu Rotten Tomatoes
├── demo_info_from_link.ipynb     # Notebook để demo thử cho việc trích xuất dữ liệu từ các link đã có
├── Preprocessing_EDA.ipynb       # Notebook để demo cho phương pháp tiền xử lý và mô hình hóa dữ liệu
```

## Hướng dẫn sử dụng

### 1. Tải mã nguồn

```bash
git clone https://github.com/AlbusKieu/DS105.git
cd DS105
```

### 2. Cài đặt thư viện cần thiết

Nên sử dụng môi trường ảo (virtual environment):

```bash
pip install pandas numpy scikit-learn streamlit requests beautifulsoup4
```

### 3. Thu thập dữ liệu

- Mở và chạy lần lượt hai notebook:
  - `IMDB_crawler.ipynb`: Thu thập dữ liệu phim từ IMDB.
  - `ROTTEN_TOMATO_crawler.ipynb`: Thu thập dữ liệu phim từ Rotten Tomatoes.

Kết quả sẽ được lưu ra các file CSV tương ứng.

### 4. Tiền xử lý, phân tích và xây dựng mô hình gợi ý

- Mở và chạy `main.ipynb`:
  - Đọc dữ liệu từ các file CSV đã thu thập.
  - Tiền xử lý dữ liệu: chuẩn hóa, mã hóa, xử lý thể loại phim, xử lý dữ liệu thiếu.
  - Phân tích dữ liệu khám phá (EDA): thống kê, trực quan hóa dữ liệu.
  - Xây dựng mô hình gợi ý phim dựa trên các thuật toán học máy.
  - Đánh giá mô hình và xuất kết quả gợi ý.

### 5. Chạy giao diện web với Streamlit

```bash
streamlit run app.py
```

Sau đó mở đường dẫn hiển thị trên terminal để sử dụng hệ thống gợi ý phim trực tuyến.

## Lưu ý

- Đảm bảo đã chạy xong các bước crawl dữ liệu trước khi chạy `main.ipynb` hoặc `app.py`.
---

## License
Dự án phục vụ mục đích học tập.
