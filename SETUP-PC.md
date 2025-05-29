# NLP - HUFLIT 2025

## Hướng dẫn cài đặt cho Windows

Chào mừng các bạn dùng PC!

Tôi sử dụng một nền tảng gọi là Anaconda để thiết lập môi trường. Đây là một công cụ mạnh mẽ xây dựng môi trường khoa học hoàn chỉnh. Anaconda đảm bảo bạn sử dụng đúng phiên bản Python và tất cả các package đều tương thích với tôi, kể cả khi hệ thống của chúng ta khác nhau. Quá trình cài đặt mất thời gian hơn và tốn nhiều dung lượng ổ cứng (5+ GB), nhưng rất đáng tin cậy khi đã hoạt động.

Nếu bạn gặp vấn đề với Anaconda, tôi cũng cung cấp một phương án thay thế. Cách này nhanh hơn, đơn giản hơn và có thể giúp bạn khởi động nhanh chóng, dù mức độ tương thích không được đảm bảo tuyệt đối.

### Trước khi bắt đầu - Lưu ý! Hãy kiểm tra các điểm "khó chịu" trên Windows:

Nếu bạn còn mới với Command Prompt, đây là [hướng dẫn](https://chatgpt.com/share/67b0acea-ba38-8012-9c34-7a2541052665) kèm bài tập. Hãy thử qua để tự tin hơn.

Có 4 điểm lưu ý phổ biến khi lập trình trên Windows:   

1. Quyền truy cập. Hãy xem [hướng dẫn](https://chatgpt.com/share/67b0ae58-d1a8-8012-82ca-74762b0408b0) về quyền trên Windows  
2. Antivirus, Firewall, VPN. Các chương trình này có thể gây cản trở cài đặt và truy cập mạng; hãy tắt tạm thời khi cần  
3. Giới hạn tên file Windows 260 ký tự – [giải thích & cách khắc phục](https://chatgpt.com/share/67b0afb9-1b60-8012-a9f7-f968a5a910c7)!  
4. Nếu chưa từng làm việc với các gói Data Science trên máy, bạn có thể cần cài Microsoft Build Tools. Đây là [hướng dẫn](https://chatgpt.com/share/67b0b762-327c-8012-b809-b4ec3b9e7be0). Sinh viên cũng chia sẻ [hướng dẫn này](https://github.com/bycloudai/InstallVSBuildToolsWindows) hữu ích cho Windows 11.    

### Phần 1: Clone Repo

Để tải mã nguồn về máy bạn.

1. **Cài đặt Git** (nếu chưa có):

- Tải Git tại https://git-scm.com/download/win
- Cài đặt và chọn các tùy chọn mặc định (bấm OK nhiều lần!). 
- Sau khi cài, có thể bạn cần mở lại Powershell (hoặc khởi động lại máy) để sử dụng Git.

2. **Mở Command Prompt:**

- Nhấn Win + R, gõ `cmd`, nhấn Enter

3. **Chuyển đến thư mục lưu project:**

Nếu đã có thư mục cho project, chuyển đến đó bằng lệnh cd. Ví dụ:  
`cd C:\Users\YourUsername\Documents\Projects`  
Thay YourUsername bằng tên user thực tế của bạn.

Nếu chưa có thư mục, hãy tạo mới:
```
mkdir C:\Users\YourUsername\Documents\Projects
cd C:\Users\YourUsername\Documents\Projects
```

4. **Clone repository:**

Nhập lệnh này ở Command Prompt trong thư mục Projects:

`git clone https://github.com/KenzyTran/nlp-llm.git`

Sẽ tạo ra thư mục `nlp-llm` và tải mã nguồn về. Dùng `cd nlp-llm` để vào thư mục này. Đây gọi là "thư mục gốc của project".

### Phần 2: Cài đặt môi trường Anaconda

Nếu bước này có vấn đề, hãy dùng Phần 2B bên dưới.

1. **Cài đặt Anaconda:**

- Tải Anaconda tại https://docs.anaconda.com/anaconda/install/windows/
- Cài đặt theo hướng dẫn. Lưu ý phần mềm này khá nặng và cài đặt lâu, nhưng rất mạnh mẽ.

2. **Thiết lập môi trường:**

- Mở **Anaconda Prompt** (tìm trong Start menu)
- Điều hướng đến "thư mục gốc của project" bằng lệnh như `cd C:\Users\YourUsername\Documents\Projects\nlp-llm` (sửa đường dẫn cho đúng). Dùng `dir` để kiểm tra xem có các thư mục con từng tuần học chưa.
- Tạo môi trường: `conda env create -f environment.yml`
- **Nếu gặp lỗi ArchiveError, nguyên nhân là do giới hạn 260 ký tự – xem lưu ý số 3 phía trên**
- Đợi vài phút để cài các package – nếu lần đầu dùng Anaconda có thể mất đến 30 phút hoặc lâu hơn tùy internet. Nếu chạy quá 1h15', hoặc lỗi khác, hãy chuyển sang Phần 2B.  
- Sau khi xong, bạn đã có môi trường AI biệt lập, sẵn sàng chạy các tác vụ LLM, vector database, v.v. Kích hoạt môi trường bằng lệnh: `conda activate llms`  

Bạn sẽ thấy dòng lệnh có tiền tố `(llms)`, nghĩa là đã kích hoạt môi trường.

3. **Khởi động Jupyter Lab:**

- Trong Anaconda Prompt, từ thư mục `nlp-llm`, gõ: `jupyter lab`

...Jupyter Lab sẽ mở trong trình duyệt. Nếu chưa biết Jupyter Lab, tôi sẽ giải thích sau! Giờ hãy đóng tab jupyter lab, đóng Anaconda prompt, và chuyển sang Phần 3.

### Phần 2B - Thay thế nếu Anaconda gặp sự cố

1. **Mở Command Prompt**

Nhấn Win + R, gõ `cmd`, nhấn Enter  

Chạy `python --version` để kiểm tra phiên bản Python.  
Nên dùng Python 3.11 để đồng bộ.  
Python 3.12 cũng dùng được, nhưng (tính đến 2/2025) Python 3.13 **chưa** tương thích do thiếu một số package.  
Nếu cần cài/đổi version, tải tại:  
https://www.python.org/downloads/

2. Chuyển tới "thư mục gốc của project" với lệnh như `cd C:\Users\YourUsername\Documents\Projects\nlp-llm`. Dùng `dir` để kiểm tra thư mục con của các tuần học.  

Tạo môi trường ảo mới với lệnh:  
`python -m venv llms`

3. Kích hoạt môi trường ảo:  
`llms\Scripts\activate`
Bạn sẽ thấy (llms) ở dấu nhắc lệnh.

4. Chạy `python -m pip install --upgrade pip` và tiếp theo là `pip install -r requirements.txt`  
Có thể mất vài phút. Nếu gặp lỗi như:

> Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

Hãy làm theo liên kết để cài Build Tools. Hướng dẫn cho Windows 11: [tại đây](https://github.com/bycloudai/InstallVSBuildToolsWindows).

Nếu vẫn lỗi, thử lệnh chắc chắn nhưng chậm hơn:  
`pip install --retries 5 --timeout 15 --no-cache-dir --force-reinstall -r requirements.txt`

6. **Khởi động Jupyter Lab:**

Từ thư mục `nlp-llm`, gõ: `jupyter lab`  
Jupyter Lab sẽ mở – hãy vào thư mục `week1` rồi mở file `day1.ipynb`. Thành công! Đóng jupyter lab và tiếp tục Phần 3.

Có vấn đề gì, hãy liên hệ tôi!

### Phần 3 - API Key OpenAI (KHÔNG BẮT BUỘC nhưng nên làm)

1. Tạo tài khoản OpenAI nếu chưa có:  
https://platform.openai.com/

2. OpenAI yêu cầu nạp tối thiểu 5 USD để dùng API. Các lệnh gọi API sẽ trừ vào số dư này. Khóa học chỉ dùng rất ít. Bạn nên nạp để tận dụng, nhưng nếu không muốn trả phí, tôi sẽ hướng dẫn dùng Ollama.

Nạp tiền tại Settings > Billing:  
https://platform.openai.com/settings/organization/billing/overview

Nên tắt tự động nạp lại tiền!

3. Tạo API key

Tạo key tại https://platform.openai.com/api-keys - nhấn 'Create new secret key', rồi 'Create secret key'. Lưu key này ở nơi an toàn; bạn sẽ không xem lại được trên website. Key thường bắt đầu bằng `sk-proj-`.

Thiết lập thêm key cho Anthropic và Google (hiện tại đang miễn phí):
- Claude API: https://console.anthropic.com/ (Anthropic)
- Gemini API: https://ai.google.dev/gemini-api (Google)

Sau này bạn sẽ dùng HuggingFace: https://huggingface.co - tạo token tại Avatar menu >> Settings >> Access Tokens.

Dùng Weights & Biases: https://wandb.ai để theo dõi quá trình train model. Tài khoản miễn phí,
