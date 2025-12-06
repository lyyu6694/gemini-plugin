import os
import requests
import google.generativeai as genai
from flask import Flask, request, jsonify
from google.generativeai.types import HarmCategory, HarmBlockThreshold

app = Flask(__name__)

# 默认配置
DEFAULT_MODEL = "gemini-1.5-flash"

def download_image(url):
    """下载图片并转换为 Gemini SDK 需要的数据格式"""
    try:
        # 设置 User-Agent 防止被某些图床拦截
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', 'image/jpeg')
        return {
            "mime_type": content_type,
            "data": response.content
        }
    except Exception as e:
        print(f"Error downloading image {url}: {e}")
        return None

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "service": "Gemini Native Plugin"}), 200

@app.route('/api/gemini/chat', methods=['POST'])
def chat():
    data = request.json
    
    # 1. 获取参数
    api_key = data.get('api_key')
    query = data.get('query', '')
    image_urls = data.get('image_urls', [])
    model_name = data.get('model_name', DEFAULT_MODEL)

    # 校验
    if not api_key:
        return jsonify({"error": "Missing api_key"}), 401
    if not query and not image_urls:
        return jsonify({"error": "Query or image_urls must be provided"}), 400

    # 2. 配置 Gemini
    genai.configure(api_key=api_key)
    
    # 安全设置：全放开，避免拒答
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    try:
        # 3. 准备 Prompt 内容
        prompt_parts = []
        if query:
            prompt_parts.append(query)
            
        if image_urls:
            # 兼容字符串格式输入 (逗号分隔)
            if isinstance(image_urls, str):
                image_urls = [url.strip() for url in image_urls.split(',') if url.strip()]
            
            for url in image_urls:
                img_data = download_image(url)
                if img_data:
                    prompt_parts.append(img_data)
                else:
                    return jsonify({"error": f"Failed to download image: {url}"}), 400

        # 4. 调用模型
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt_parts, safety_settings=safety_settings)

        # 5. 返回结果
        return jsonify({
            "result": response.text,
            "used_model": model_name
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)