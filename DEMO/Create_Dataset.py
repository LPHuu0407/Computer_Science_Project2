import os
# Đường dẫn của tập dữ liệu và tạo danh sách các label
BASE_DIR = 'D:\\MYLEARNING\\THE_JOURNEY_IV\\COMPUTER_SCIENCE_PROJECT_2\\PRACTICE\\bbc'
LABELS = ['business', 'entertainment', 'politics', 'sport', 'tech']
# Khởi tạo bộ dữ liệu
def create_data_set():
    with open('data.txt', 'w', encoding='utf8') as outfile:
        for label in LABELS:
            dir = '%s/%s' % (BASE_DIR, label)
            for filename in os.listdir(dir):
                fullfilename = '%s/%s' % (dir, filename)
                print(fullfilename)
                with open(fullfilename, 'rb') as file:
                    text = file.read().decode(errors= 'replace').replace('\n', '')
                    outfile.write('%\t%s\t%s\n' % (label, filename, text))