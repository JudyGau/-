import requests
import parsel
import os

from urllib3 import Retry
from requests.adapters import HTTPAdapter

weburl = "https://www.vcg.com/creative-image/xiangcai/"
datase_path = "Dataset/香菜/"
if not os.path.exists(datase_path):
    os.makedirs(datase_path)
    print(f"文件夹 '{datase_path}' 创建成功.")


# sajssdk_2015_cross_new_user=1; FZ_STROAGE.vcg.com=eyJTRUVTSU9OSUQiOiI5YzZhNzZiNDAyMWNlY2ZmIiwiU0VFU0lPTkRBVEUiOjE3MTkzMTAyMDQ5NDN9; ARK_ID=undefined; Hm_lvt_5fd2e010217c332a79f6f3c527df12e9=1719277451,1719293837,1719310667; clientIp=60.175.62.69; uuid=c0ded61a-2807-4e95-8127-9250ffb2bb94; _c_WBKFRo=C6e51d4VuPT8VeuJe9mBokCm2Y4etgZmWFEWXtpx; _nb_ioWEgULi=; acw_tc=276082a817193190881216185e09875d2746353fba11cf6411aed3b898c280; api_token=ST-830-310fa86e447cbbf47c856cd8f67bc2432; name=18055606646; sensorsdata2015jssdkcross={"distinct_id":"f00eb119748b18324e228834886a11737","first_id":"1904cebf6c5fe4-0b19ea82365c498-2003017e-1327104-1904cebf6c61104","props":{"$latest_traffic_source_type":"自然搜索流量","$latest_search_keyword":"未取到值","$latest_referrer":"https://www.bing.com/"},"$device_id":"1904cebf6c5fe4-0b19ea82365c498-2003017e-1327104-1904cebf6c61104"}; fingerprint=863c48aa0e4def2cfef86d8b07c60e5a; Hm_lpvt_5fd2e010217c332a79f6f3c527df12e9=1719319112; _fp_=eyJpcCI6IjYwLjE3NS42Mi42OSIsImZwIjoiODYzYzQ4YWEwZTRkZWYyY2ZlZjg2ZDhiMDdjNjBlNWEiLCJocyI6IiQyYSQwOCROd0ZDTkh1bWI4TXdSLy9WbnBZTFQuZmtmTHFESWRvbzM4OFp5ODQ5WHc5NzM0Rng2c24wLiJ9; ssxmod_itna=Yq+xyDBQi=DQLOxl8DCEa5Rqxmq7KN4+KeeDkmFboDBw74iNDnD8x7YDvmfxAxExpKQzjApwNkFlgWdWZCtjaeA=8PjveDHxY=DUZrq4oD435GwD0eG+DD4DWDmmHDnxAQDjxGp9uXwV=Dm4GWSqDgB4GgejiD0RU9tuiD4qDBIhdDKd29DGYUQnUQEMwt5E=DjdrD/8hakA63OccKBLub5O0FtqGyBPGu0qRgRj4BldXxEDfxR3AxKwO78BmKSEqFb7mrKA8pSEhKl7x4nGDo4SlKDDfPIY1xD=; ssxmod_itna2=Yq+xyDBQi=DQLOxl8DCEa5Rqxmq7KN4+KeeDkmFtx8w1ODGIND/i88DFgkAd5zK=D67Y5hCxw6qgD8q9rBgheYKFDoRTqoGB7yjkVGYyGvA8iG9hC455dK6Np7yrEhKfOsfB+q28fwtxoeH56GkciBTPZr+P=FwF6GrQ9ehwrbtYmBDvi95MApiN=go8EqAhiqfMDAI6tn7/0wabZe6wfehgnxxs7D=/ieba=iqG3nouUD=2QKh5oHCwEu3nBEiu4Qh4VdjTt8rrcruYXGN2ApUT/yiG09DAe8QExZr4PqSF/2QjPcmsidzf8BE5KoaQkx3ODYQtT+DDjK54oAfozDTAO5cD+zSq1D1IGOeb55oKkTKKroq8xTbhYxtw+mzfNUqdB+D3xG0jDNOhAAHfxkjwOKT7mY8Eg1oMFEoe0w1cw50LqARDimPD7QlDGcDG7=iDD===
headers = {
    "Cookie": 'FZ_STROAGE.vcg.com=eyJTRUVTSU9OSUQiOiI5YzZhNzZiNDAyMWNlY2ZmIiwiU0VFU0lPTkRBVEUiOjE3MTkzMTAyMDQ5NDN9; ARK_ID=undefined; _c_WBKFRo=C6e51d4VuPT8VeuJe9mBokCm2Y4etgZmWFEWXtpx; api_token=ST-830-310fa86e447cbbf47c856cd8f67bc2432; name=18055606646; fingerprint=863c48aa0e4def2cfef86d8b07c60e5a; uuid=c1fde539-1636-4d49-bc21-bd61ea3802fd; clientIp=60.175.62.69; Hm_lvt_5fd2e010217c332a79f6f3c527df12e9=1719310667,1719324174,1719396662,1719450601; acw_sc__v2=667cf7f14d2ac9947991ce69c799391b5bacc338; acw_sc__v3=667cfee64a4360eff415740b2107b8d24cac7f05; acw_tc=276077ba17194677687021594eb1ca3cb0932977df2b1e5cde550b5e16d810; sensorsdata2015jssdkcross={"distinct_id":"f00eb119748b18324e228834886a11737","first_id":"1904cebf6c5fe4-0b19ea82365c498-2003017e-1327104-1904cebf6c61104","props":{"$latest_traffic_source_type":"直接流量","$latest_search_keyword":"未取到值_直接打开","$latest_referrer":""},"$device_id":"1904cebf6c5fe4-0b19ea82365c498-2003017e-1327104-1904cebf6c61104"}; Hm_lpvt_5fd2e010217c332a79f6f3c527df12e9=1719468007; _fp_=eyJpcCI6IjYwLjE3NS42Mi42OSIsImZwIjoiODYzYzQ4YWEwZTRkZWYyY2ZlZjg2ZDhiMDdjNjBlNWEiLCJocyI6IiQyYSQwOCRva3FNU2JHUHJWSkMuT2pkWjguOUN1N1NzZk5mYUZ4VFBzM3RubHJuQzBUWi9MQm5iQVlUeSJ9; ssxmod_itna=eqAxBDg7DQ9D9iDzxmxBRRbpq0=qe+5IPG8iipiFBDBkAW4iNDnD8x7YDvmjN7EjGboTp8re0QC3hlrdqrz2nOvbe86Dd5oDU4i8DCqgKqbDemtD5xGoDPxDeDADYo0DAqiOD7qDdXLvvz2xGWDmRsDYvHDQ5h54DFBZF8Y4i7DD5=zx07d/KDegP+B6E1YZgDqlKD96YDsa+Ldl9pXKM4/ZgEEF7dEx0kV40OSoO1k8hCDUHhaCPPTDWONqu4NjDxYlD51Bifqi2cdr04dDboGXcDk6DDiOiQTQGDD=; ssxmod_itna2=eqAxBDg7DQ9D9iDzxmxBRRbpq0=qe+5IPG8iipiFD8wZEqGX+hbGaAFAjhwTjxn4GO9oniQhq442KhP2DfQf6x4rDPm6t=P+1Lik4M4mDafdKDZh6PhTNaOEyqW9FYhLry1wDh7WFqAtQDv5S+i03AejUhhIlkA+3wGg3q2g9r00zi6GAAQ2G4jhUguLWfY3iRjfmxY4okdrGfQGRoj29R9cUdI9njysa8DqWc9GDdWR4SKhFCYmqu708MGvXdQemKfT8fg2Y7d3n9OcBqnvd2tptinRAiENGbCfzBEHOdQYEiyXKuAp7v2n6EW4toASeIMgjG0dwaGlreAjqMmv6W=trxqCwAmek7iTDp+Oqe0vletzjI/DP6rv0lQiZ2M6I+mdNoTjA6hYijeY+hdjYQz+NAcLAnIhFwIDWl2L2BbHK3qhwkOGzQQCYrGCwXDo6rUVrPpYrM0w5tGQge6rDhebm9mqFp+eIw7Laee3ZYF4d3nQaom/ZrqPq9U9ICeN0qi6GRPPqWIfaH6a62DAGGYw228bLkIetXc6OLQZpEPYe=itSirIEAbxFDG2DCtibk3I9A0B4T5I656=acF5FWk6has5q65li+b3zFXroQDDFqD+6DxD'.encode(
        "latin-1", errors="ignore"
    ).decode(
        "latin-1"
    ),
    "User-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36 SLBrowser/9.0.3.5211 SLBChan/33",
}

# def download_img(url, name):
#     img_data = requests.get(url=url, headers=headers).content
#     with open(datase_path + name, mode="wb") as f:
#         f.write(img_data)
#         print("保存完成：", name)


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
        # 可以选择在这里抛出异常或者记录日志等
    except Exception as e:
        print(f"保存图片 {name} 时发生其他错误：{e}")


def download_page(url):

    response = requests.get(url=url, headers=headers)

    html_str = response.text
    # print(html_str)
    selector = parsel.Selector(html_str)

    lis = selector.xpath('//div[@id="imageContent"]/section/div/figure')

    for li in lis:
        # print(li)
        pic_url = "https:" + li.xpath("./a/img/@data-src").get()
        pic_name = (
            li.xpath("./div[1]/span[2]/text()").get().split("/")[-1]
            + pic_url.split("/")[-1]
        )
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
            and (pic_name.find("香菜") != -1 or pic_name.find("芫荽") != -1)
        ):
            # print(pic_title, pic_url)
            download_img(pic_url, pic_name)


# __main__

for i in range(21, 31):
    url = weburl + f"?page={i}"
    download_page(url)
    print(f"第{i}页保存完成")
