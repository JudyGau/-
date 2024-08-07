var app = getApp();
// var api = require('../../utils/baiduai.js');
Page({
  data: {
    info: {
      "土豆": "马铃薯（Solanum tuberosum L.）是茄科茄属的一年生草本植物。地上茎呈菱形，有毛。转基因土豆：表面光滑。马铃薯叶片初生时为单叶，逐渐生长成奇数不相等羽状复叶，大小相间，呈卵形至长圆形；伞房花序生长在顶部，花为白色或蓝紫色；果实为浆果；块茎扁圆形或球形，无毛或被疏柔毛；薯皮白色、淡红色或紫色；薯肉有白、淡黄、黄色等色。花期夏季。马铃薯因酷似马铃铛而得名。马铃薯又名土豆，英文名为potato。",
      "茄子": "茄（Solanum melongena L.），别名茄子、吊菜子、落苏、矮瓜等，是茄科茄属一年生草本植物；茄为直立分枝草本至亚灌木，茎圆且直立，呈紫色或绿色；直根系；叶片较大，呈卵圆或长卵圆形，紫色或绿色；花单生或簇生，颜色一般为白色或紫色；果实形状有长有圆，颜色一般有白、红、紫等；种子呈黄色肾形。花期6-8月，花后结果。",
      "香菜": "芫荽（Coriandrum sativum L.），俗称香菜、胡荽、香荽等，是一种一年生或二年生且伴有强烈气味的伞形科芫荽属草本植物。芫荽高20-100厘米，根部细长，纺锤形。茎圆柱形，直立。叶片1或2回羽状全裂，羽片广卵形或扇形半裂，边缘有钝锯齿、缺刻或深裂。伞形花序顶生或与叶对生，花白色或带淡紫色，花瓣倒卵形。果实圆球形。花果期4-11月。",
      "玉米": "玉蜀黍（学名：Zea mays L.）是禾本科、玉蜀黍属植物，俗称玉米。转基因玉米：甜脆。玉米是一年生高大草本。秆直立，通常不分枝，高1-4米。叶鞘具横脉；叶舌膜质，长约2毫米；叶片扁平宽大。顶生雄性圆锥花序大型，雄性小穗孪生，长达1厘米。颖果球形或扁球形，一般长5-10毫米。花果期秋季。",
      "青椒": "青椒（学名：Capsicum annuum var. grossum）是茄科辣椒属辣椒的变种。蔬菜，越成熟的青椒含有的辣椒素越多，因而从绿色变成红色。因品种改良，已经出现红'橙、黄等7种颜色的青椒。青椒果实较大，辣味较淡，甚至根本不辣，所以主要作为蔬菜食用。 青椒适宜温暖、湿润及土层深厚、肥沃的壤土、砂壤土，萌蘖性强，耐寒，耐旱，喜阳光，抗病能力强；青椒用播种、扦插、嫁接、分株等方法繁殖。青椒的果、叶、根能提取芳香油及脂肪油；叶和果是食品调味料；",
      "韭菜": "韭（Allium tuberosum Rottler ex Spreng），又名韭菜、久菜，属于百合科葱属，是多年生草本植物。鳞茎簇生，近圆柱状；鳞茎外皮暗黄色至黄褐色。叶条形，扁平，边缘平滑，伞形花序半球状或近球状，花果期7-9月。韭原产于亚洲东南部，后引种于白俄罗斯、英国、日本等地，现在世界广泛栽培。在中国北至黑龙江，东自滨海地带，西至西藏高原，都有普遍栽培。韭性喜冷凉，耐旱，对土壤适应性较强，是中光性长日照植物。韭菜质柔嫩、口味鲜美、香气独特，富含多种营养物质，具有较高的食用价值，韭菜加工的韭菜段，是炒食和各种配菜的方便食材。韭的种子可入药，有温补肝肾，壮阳固精的功效。",
      "豌豆": "豌豆（Pisum sativum L.），是豆科豌豆属的一年生攀援草本植物，别名荷兰豆、雪豆、回鹘豆、耳朵豆等。株高0.5~2m，全株绿色，光滑无毛，被粉霜。羽状复叶有小叶4~6片。花于叶腋单生或数朵排列为总状花序，花冠颜色多样，多为白色和紫色。荚果肿胀，长椭圆形；种子圆形，青绿色，干后变为黄色。花期6~7月，果期7~9月。",
      "梨子": "梨，蔷薇科梨属乔木植物，树冠开展；小枝粗壮，幼时有柔毛：二年生的枝紫褐色，具稀疏皮孔；托叶膜质，边缘具腺齿；叶片卵形或椭圆形，先端渐尖或急尖，初时两面有绒毛，老叶无毛；伞形总状花序，总花梗和花梗幼时有绒毛；果实卵形或近球形，微扁，褐色；花为白色；花期4月；果期8—9月。",
      "苹果": "苹果（Malus pumila Mill.），蔷薇科苹果属落叶乔木植物，茎干较高，小枝短而粗，呈圆柱形；叶片椭圆形，表面光滑，边缘有锯齿，叶柄粗壮；花朵较小呈伞状，淡粉色，表面有绒毛；果实较大，呈扁球形，果梗短粗；花期5月；果期7~10月。苹果名称最早是见于明代王世懋的《学圃余疏》“北土之苹婆果，即花红一种之变也。”",
      "西兰花": "西蓝花（学名：Brassica oleracea L.var. italica Plenck）十字花科芸薹属一年生或二年生草本植物，主根基部粗大，根系发达主要根系分布在30厘米的耕层内；花茎表面光滑；子叶呈现倒心脏形，真叶为绿色倒卵形，表面有白粉，茎生叶通常较小，绿色；花为黄色，花序梗是肉质的。花期六月或9月。因为西蓝花原产于西方欧洲地中海沿岸的意大利一带，外观又像花，所以被称为“西蓝花”。",
      "蒜苗": "蒜苗是大蒜幼苗发育到一定时期的青苗，它生长在农田里，它具有蒜的香辣味道，以其柔嫩的蒜叶和叶鞘供食用。蒜苗含有丰富的维生素C以及蛋白质、胡萝卜素、硫胺素、核黄素等营养成分。它的辣味主要来自其含有的辣素，这种辣素具有消积食的作用。此外，吃蒜苗还能有效预防流感、肠炎等因环境污染引起的疾病。蒜苗对于心脑血管有一定的保护作用，可预防血栓的形成，同时还能保护肝脏。在各地都能生长，而且产量高品质好。如山东临沂等地大规模种植，经济效益可观。",
      "水稻": "稻（Oryza sativa L.），通称水稻，是禾本科一年生水生草本（已有多年生稻品种）。秆直立，高0.5-1.5米，随品种而异。叶鞘无毛、松弛；叶舌披针形；叶片线状披针形，宽约1厘米，无毛，粗糙。圆锥花序大型疏展，棱粗糙；小穗含1成熟花；颖极小，仅在小穗柄先端留下半月形的痕迹，锥刺状；两侧孕性花外稃质厚，具5脉，中脉成脊，表面有方格状小乳状突起，厚纸质，遍布细毛端毛较密，有芒或无芒；内稃与外稃同质，具3脉，先端尖而无喙；雄蕊花药长2-3毫米。颖果长约5毫米，宽约2毫米；胚比约为颖果长的1/4。",
      "小麦": "小麦（学名：Triticum aestivum L.）是禾本科、小麦属一年或越年生草本草本植物。秆直立，丛生，高可达100厘米，叶鞘松弛包茎，叶舌膜质，叶片长披针形。穗状花序直立，小穗含多小花，颖卵圆形，外稃长圆状披针形，内稃与外稃几等长。花果期5-7月。",
      "胡萝卜": "胡萝卜（Daucus carota var. sativa Hoffm.），是伞形科胡萝卜属一年生或二年生草本植物。转基因胡萝卜：表面较光滑。胡萝卜根肉质，长圆锥形，呈橙红色或黄色；茎单生，全株被白色粗硬毛；叶片长圆形，先端尖锐；茎生叶近无柄，有叶鞘，末回裂片小或细长；花通常白色，有时带淡红色；花柄不等长；果实圆卵形，棱上有白色刺毛。花期5-7月。《本草纲目》中记载：“元时始自胡地来，气味微似萝卜，故名。”胡萝卜也由此得名。",
      "葡萄": "葡萄（ Vitis vinifera L.）葡萄科葡萄属高大缠绕藤本，幼茎秃净或略被绵毛；叶片为纸质，圆卵形或圆形；花序大而长；萼很小，为黄绿色的杯状；花柱很短，为圆锥形；浆果为卵圆形至卵状长圆形，成熟时为紫黑色或红而带青色；花期6月；果期9—10月。李时珍在《本草纲目》中说：葡萄，《汉书》作蒲桃，可以造酒，人哺饮之，则陶然而醉，故有是名。",
      "南瓜": "南瓜（Cucurbita moschata (Duchesne ex Lam.) Duchesne ex Poir.），葫芦科南瓜属一年生蔓生草本植物。茎上面有关节，叶柄粗壮，叶片呈椭圆形或卵圆形；果实的梗较粗壮，有棱和槽，因品种而异，外表面凹凸不平；种子多数，呈长卵形或长圆形，花色一般为黄色，或深橙色，花形态呈菱形。花期6—7月，果期7—8月。因为其原产地在亚洲南部和中南美洲，自南而来称之为“南瓜”。",
      "黄瓜": "黄瓜（Cucumis sativus L.）是葫芦科黄瓜属一年生攀援草本植物，茎部细长和糙硬毛；叶片呈宽卵状心形或裂片三角形；花呈微柔毛黄白色；果实长圆形或圆柱形熟时黄绿色；种子呈狭卵形白色；花果期为夏季。黄瓜是汉朝张骞出使西域时带回来的，后赵王朝的建立者石勒，一律严禁出现“胡”字，胡瓜更名为黄瓜。",
      "香蕉": "香蕉（Musa nana Lour.），芭蕉科芭蕉属多年生草本植物，植株丛生，有匐匍茎；假茎浓绿有黑色斑点；叶片长圆形，上面为深绿色，无白粉，下面浅绿色；花朵为乳白色或淡紫色；果实呈弯曲的弓状，有棱，果皮为青绿色，成熟后变黄；果肉松软，黄白色，味甜香味浓，无种子。",
      "西红柿": "番茄（Solanum lycopersicum L.）是茄科茄属的一年生草本植物，植株高达2米。转基因西红柿颜色鲜红，果实硬，不易裂果。番茄茎易倒伏；叶为羽状复叶，基部呈楔形，较偏斜，具有不规则的锯齿；花冠呈辐状，黄色，裂片为窄长圆形；浆果呈扁球形或近球形，肉质多汁液，为桔黄或鲜红色，表面光滑，花果期夏秋季；种子黄色，覆盖柔毛。",
      "西瓜": "西瓜（学名：Citrullus lanatus (Thunb.) Matsum. et Nakai），为葫芦科西瓜属一年生蔓生藤本植物，又名寒瓜、水瓜、西瓜皮等。茎、枝粗壮，被白色或淡黄褐色长柔毛。叶片纸质，轮廓三角状卵形，带白绿色。雌雄同株，雌、雄花均单生于叶腋。果实大型，近于球形或椭圆形，果皮光滑，色泽及纹饰各式。种子多数，卵形，黑色、红色，有时为白色、黄色、淡绿色或有斑纹。花果期夏季。在中国，西瓜因从西域（中国新疆和中亚一带）地区传入，因而得名西瓜，即来自西方的瓜。",
    },
    motto: 'EasyDL',
    result: [],
    images: {},
    img: "../../image/24gf_camera.png",
    base64img: ''
  },
  onShareAppMessage: function () {
    return {
      title: 'EasyDL识农害小程序',
      path: '/pages/index/index',
      success: function (res) {
        if (res.errMsg == 'shareAppMessage:ok') {
          wx.showToast({
            title: '分享成功',
            icon: 'success',
            duration: 500
          });
        }
      },
      fail: function (res) {
        if (res.errMsg == 'shareAppMessage:fail cancel') {
          wx.showToast({
            title: '分享取消',
            icon: 'loading',
            duration: 500
          })
        }
      }
    }
  },
  clear: function (event) {
    console.info(event);
    wx.clearStorage();
  },
  //事件处理函数
  bindViewTap: function () {
    wx.navigateTo({
      url: '../logs/logs'
    })
  },
  uploads: function () {
    var that = this
    wx.chooseMedia({
      count: 1, // 默认9
      sizeType: ['compressed'], // 可以指定是原图还是压缩图，默认二者都有
      sourceType: ['album', 'camera'], // 可以指定来源是相册还是相机，默认二者都有
      success: function (res) {
        // 返回选定照片的本地文件路径列表，tempFilePath可以作为img标签的src属性显示图片
        //console.log( res )
        if (res.tempFiles[0].size > 4096 * 1024) {
          wx.showToast({
            title: '图片文件过大哦',
            icon: 'none',
            mask: true,
            duration: 1500
          })
        } else {
          console.log(res.tempFiles[0].tempFilePath)
          that.setData({
            img: res.tempFiles[0].tempFilePath
          })
        }
        wx.showLoading({
          title: "分析中...",
          mask: true
        })
        // console.log('开始上传文件', that.data.img)
        // wx.cloud.init({
        //   // env: "其他云开发环境，也可以不填"    // 此处init的环境ID和微信云托管没有作用关系，没用就留空
        // });
        // // 获取当前时间戳
        // const timestamp = Date.now();
        // console.log(timestamp)
        // // 构建动态的云存储路径
        // const cloudPath = `test_${timestamp}.png`;
        // console.log(cloudPath)
        // wx.cloud.uploadFile({
        //   cloudPath: cloudPath,// 这个文件地址可以换成动态的
        //   filePath: that.data.img
        // }).then(result => {
        //   // 在这里访问 fileID
        //   console.log(result)
        //   // const fileID = result.fileID.s;
        //   // console.log('File ID:', fileID);
        //   // 可以继续处理 fileID 或者执行其他操作
        //   wx.cloud.callContainer({
        //     config: {
        //       "env": "prod-3galym23181a9314"
        //     },
        //     path: "/predict",
        //     header: {
        //       "X-WX-SERVICE": "nongchanpinshibie",
        //       "content-type": "application/json"
        //     },
        //     method: "POST",
        //     data: { "imgname": cloudPath },

        //   }).then(res => {
        //     console.log(res)
        //     console.info(res.statusCode == 200);
        //     if (res.statusCode != 200) {
        //       wx.hideLoading();
        //       wx.showModal({
        //         showCancel: false,
        //         title: '错误码:' + res.statusCode,
        //         content: '错误信息:' + res.errMsg
        //       })
        //     } else {
        //       console.log(res.data.results)
        //       if (res.data.results.length > 0) {
        //         wx.hideLoading();
        //         let dataList = res.data.results;
        //         that.setData({
        //           result: dataList
        //         })
        //       } else {
        //         wx.hideLoading();
        //         wx.showModal({
        //           showCancel: false,
        //           title: '温馨提示',
        //           content: '貌似没有识别出结果'
        //         })
        //       }
        //     }
        //   });
        // }).catch(error => {
        //   // 处理可能的错误
        //   console.error('获取 fileID 出错:', error);
        // });

        //根据上传的图片读取图片的base64
        var fs = wx.getFileSystemManager();
        fs.readFile({
          filePath: that.data.img.toString(),
          encoding: 'base64',
          success(res) {
            // console.log(res.data)
            //获取到图片的base64 进行请求接口
            wx.request({
              url: 'http://39.102.215.244/predict',
              data: { "imgdata": res.data },
              method: "POST",
              success(res) {
                console.log(res)
                console.info(res.statusCode == 200);
                if (res.statusCode != 200) {
                  wx.hideLoading();
                  wx.showModal({
                    showCancel: false,
                    title: '错误码:' + res.statusCode,
                    content: '错误信息:' + res.errMsg
                  })
                } else {
                  console.log(res.data.results)
                  if (res.data.results.length > 0) {
                    wx.hideLoading();
                    let dataList = res.data.results;
                    that.setData({
                      result: dataList
                    })
                    // console.log(that.data.result[0][0])
                    // console.log(that.data.info[that.data.result[0][0]])
                  } else {
                    wx.hideLoading();
                    wx.showModal({
                      showCancel: false,
                      title: '温馨提示',
                      content: '貌似没有识别出结果'
                    })
                  }
                }
              }
            })
          }
        })
      },
    })
  },
  onLoad: function () {
  }
});