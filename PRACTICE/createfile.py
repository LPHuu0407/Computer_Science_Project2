import os
def create_txt_files(num_files, directory):
    # Lặp để tạo nhiều file .txt
    for i in range(1, num_files+1):
        file_name = f"{i}.txt"
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'w', encoding='utf-8') as f:  # Sử dụng utf-8 để ghi
            f.write(f"Đây là nội dung của file {file_name}")

# Số lượng file cần tạo
num_files = 100

# Đường dẫn thư mục để lưu file
directory = "D:\MYLEARNING\THE_JOURNEY_IV\COMPUTER_SCIENCE_PROJECT_2\PRACTICE\Dataset\CNN"

create_txt_files(num_files, directory)