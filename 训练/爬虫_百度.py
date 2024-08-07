import requests
import parsel
import os

from urllib3 import Retry
from requests.adapters import HTTPAdapter


datase_path = "dataset/train/韭菜/"

if not os.path.exists(datase_path):
    os.makedirs(datase_path)
    print(f"文件夹 '{datase_path}' 创建成功.")

headers = {
    "User-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36 SLBrowser/9.0.3.5211 SLBChan/33",
    "Cookie": 'BDqhfp=韭菜&&NaN-1undefined&&1224&&2; PSTM=1665928122; BIDUPSID=9E241270BBC3CDE5016571061D16B914; BDUSS=ZCLU9RSzFqQ1oweUlwVlRLZUV6cldSQUYybG0yUVp6bjliano4cUVOVWdnS05sSVFBQUFBJCQAAAAAAQAAAAEAAABmdKEeAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACDze2Ug83tlbG; BDUSS_BFESS=ZCLU9RSzFqQ1oweUlwVlRLZUV6cldSQUYybG0yUVp6bjliano4cUVOVWdnS05sSVFBQUFBJCQAAAAAAQAAAAEAAABmdKEeAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACDze2Ug83tlbG; BAIDUID=9574B963BAB57E2D2AD42E5DAC2D2C31:FG=1; H_WISE_SIDS=60363_60446; H_WISE_SIDS_BFESS=60363_60446; H_PS_PSSID=60453_60470_60491_60499_60472; BDORZ=FFFB88E999055A3F8A630C64834BD6D0; indexPageSugList=["韭菜","童靴小学女生","女孩小脚","童靴控 吧","童靴控 分吧","童靴控 贴吧","童靴控吧 贴吧","童靴控吧 被封","布鞋家园"]; BAIDUID_BFESS=9574B963BAB57E2D2AD42E5DAC2D2C31:FG=1; delPer=0; PSINO=7; BDRCVFR[dG2JNJb_ajR]=mk3SLVN4HKm; userFrom=www.baidu.com; BDRCVFR[-pGxjrCMryR]=mk3SLVN4HKm; ab_sr=1.0.1_ZjRmOTEzMjg3MTk5ZmFkOTA3MTNkOTIzZGE3YzQ5OWNkOGYyMWRkZjgyMmVkYTk5MTZhY2Y0MDlmNmM0NjkxODBhZjYyYjk3Nzk0ZjU1OWQ5NjlmN2FhZjI0ZDlhZTk1ZTAzNGRhMWUwMDQxMGFlZjYzMjhjODQ5NDM3ZGU1YzNlNDM3ODg5YTkzZjkzMDEwMDUxYmViMGIzOGNkZGI3OA=='.encode(
        "latin-1", errors="ignore"
    ).decode(
        "latin-1"
    ),
}


def download_img(url, name, retries=3):
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        status_forcelist=[429, 500, 502, 503, 504],  # 可以根据需要扩展
        allowed_methods=["GET"],  # 允许重试的HTTP方法
        backoff_factor=1,  # 重试之间的等待时间因子
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    try:
        img_data = session.get(url=url, headers=headers, timeout=6).content
        with open(datase_path + name, mode="wb") as f:
            f.write(img_data)
            print("保存完成：", name)
    except requests.RequestException as e:
        print(f"下载图片 {name} 失败: {e}")
    except Exception as e:
        print(f"保存图片 {name} 时发生其他错误：{e}")


def download_page(url):

    response = requests.get(url=url, headers=headers)

    json_data = response.json()
    # print(json_data)
    lis = json_data["data"]

    # print(lis)

    for li in lis[:-1]:
        # print(li)
        pic_url = li["middleURL"]
        print(pic_url)
        pic_name = li["fromPageTitle"] + ".jpg"
        print(pic_name)
        if (
            pic_name.find("食") == -1
            and pic_name.find("饮") == -1
            and pic_name.find("拍") == -1
            and pic_name.find("肉") == -1
            and pic_name.find("餐") == -1
            and pic_name.find("盘") == -1
            and pic_name.find("田") == -1
            and pic_name.find("烤") == -1
            and pic_name.find("炒") == -1
            and pic_name.find("烧") == -1
            and pic_name.find("炸") == -1
            and pic_name.find("炖") == -1
            and pic_name.find("烩") == -1
            and pic_name.find("煎") == -1
            and pic_name.find("闷") == -1
            and pic_name.find("煲") == -1
            and pic_name.find("锅") == -1
            and pic_name.find("碗") == -1
            and pic_name.find("汤") == -1
            and pic_name.find("蛋") == -1
            and pic_name.find("蒸") == -1
            and pic_name.find("煮") == -1
            and pic_name.find("凉") == -1
            and pic_name.find("拌") == -1
            and pic_name.find("切") == -1
            and pic_name.find("炝") == -1
            and pic_name.find("沙拉") == -1
            and pic_name.find("酱") == -1
            and pic_name.find("汁") == -1
            and pic_name.find("泥") == -1
            and pic_name.find("水") == -1
            and pic_name.find("片") == -1
            and pic_name.find("条") == -1
            and pic_name.find("丝") == -1
            and pic_name.find("块") == -1
            and pic_name.find("丁") == -1
            and pic_name.find("卷") == -1
            and pic_name.find("派") == -1
            and pic_name.find("叶") == -1
            and pic_name.find("花") == -1
            and pic_name.find("树") == -1
            and pic_name.find("园") == -1
            and pic_name.find("籽") == -1
            and (pic_name.find("韭菜") != -1)
        ):
            # print(pic_title, pic_url)
            download_img(pic_url, pic_name)


# __main__

# url = weburl + f"?page={i}"
for page in range(1, 11):
    weburl = f"https://image.baidu.com/search/acjson?tn=resultjson_com&logid=8534825734403590170&ipn=rj&ct=201326592&is=&fp=result&fr=&word=韭菜&queryWord=韭菜&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=&z=&ic=&hd=&latest=&copyright=&s=&se=&tab=&width=&height=&face=&istype=&qc=&nc=1&expermode=&nojc=&isAsync=&pn={page * 30}&rn=30&gsm=1e&1722250851044="
    download_page(weburl)
    print("保存完成")
