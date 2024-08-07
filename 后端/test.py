import requests
from PIL import Image

url = "https://7072-prod-3galym23181a9314-1327678087.tcb.qcloud.la/"+img_name
temp_name = "temp.png"
img_data = requests.get(url=url).content

with open(temp_name, mode="wb") as f:
    f.write(img_data)
    print("保存完成：", temp_name)
    
image = Image.open(temp_name)
print("成功")
# # 替换成你的云函数的HTTP触发器URL或者云开发HTTP API的URL
# cloud_function_url = 'https://your-cloud-function-url'
# # 或者微信云开发的HTTP API地址
# get_temp_file_url_api = 'https://api.weixin.qq.com/tcb/getTempFileURL?access_token=ACCESS_TOKEN'

# # 通过云函数获取文件临时访问链接
# def get_file_temp_url():
#     # 替换成你的环境ID和资源ID
#     fileID = 'cloud://prod-3galym23181a9314.7072-prod-3galym23181a9314-1327678087/test.png'
#     data = {'file_list': [fileID]}

#     try:
#         response = requests.post(cloud_function_url, data=json.dumps(data))
#         result = response.json()
#         temp_url = result.get('file_list')[0].get('tempFileURL')
#         print('文件临时访问链接:', temp_url)
#     except Exception as e:
#         print('获取文件临时访问链接失败:', str(e))
