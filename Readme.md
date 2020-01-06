# 分类网络应用（C++）<a name="ZH-CN_TOPIC_0208837235"></a>

本Application支持运行在Atlas 200 DK或者AI加速云服务器上，实现了对常见的分类网络的推理功能并输出前n个推理结果。

## 前提条件<a name="zh-cn_topic_0203223265_section137245294533"></a>

部署此Sample前，需要准备好以下环境：

-   已完成Mind Studio的安装。
-   已完成Atlas 200 DK开发者板与Mind Studio的连接，交叉编译器的安装，SD卡的制作及基本信息的配置等。

## 软件准备<a name="zh-cn_topic_0203223265_section181111827718"></a>

运行此Sample前，需要按照此章节获取源码包，并进行相关的环境配置。

1.  <a name="zh-cn_topic_0203223265_li953280133816"></a>获取源码包。

    将[https://gitee.com/Atlas200DK/sample-classification/tree/1.3x.0.0/](https://gitee.com/Atlas200DK/sample-classification/tree/1.3x.0.0/)仓中的代码以Mind Studio安装用户下载至Mind Studio所在Ubuntu服务器的任意目录，这两个文件必须存放到同一个目录下。例如代码存放路径为：_$HOME/AscendProjects/sample-classification_。

2.  <a name="zh-cn_topic_0203223265_li29641938112018"></a>获取此应用中所需要的原始网络模型。

    参考[表1](#zh-cn_topic_0203223265_table1119094515272)获取此应用中所用到的原始网络模型及其对应的权重文件，并将其存放到Mind Studio所在Ubuntu服务器的任意目录，这两个文件必须存放到同一个目录下。例如：_$HOME/models/classification_。

    **表 1**  通用分类网络应用使用模型

    <a name="zh-cn_topic_0203223265_table1119094515272"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0203223265_row677354502719"><th class="cellrowborder" valign="top" width="12.85%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0203223265_p167731845122717"><a name="zh-cn_topic_0203223265_p167731845122717"></a><a name="zh-cn_topic_0203223265_p167731845122717"></a>模型名称</p>
    </th>
    <th class="cellrowborder" valign="top" width="12.57%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0203223265_p277317459276"><a name="zh-cn_topic_0203223265_p277317459276"></a><a name="zh-cn_topic_0203223265_p277317459276"></a>模型说明</p>
    </th>
    <th class="cellrowborder" valign="top" width="74.58%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0203223265_p9773114512270"><a name="zh-cn_topic_0203223265_p9773114512270"></a><a name="zh-cn_topic_0203223265_p9773114512270"></a>模型下载路径</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0203223265_row3122314144215"><td class="cellrowborder" valign="top" width="12.85%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0203223265_p14407251134314"><a name="zh-cn_topic_0203223265_p14407251134314"></a><a name="zh-cn_topic_0203223265_p14407251134314"></a>alexnet</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.57%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0203223265_p116141194720"><a name="zh-cn_topic_0203223265_p116141194720"></a><a name="zh-cn_topic_0203223265_p116141194720"></a>图片分类推理模型。</p>
    <p id="zh-cn_topic_0203223265_p86191184712"><a name="zh-cn_topic_0203223265_p86191184712"></a><a name="zh-cn_topic_0203223265_p86191184712"></a>是基于Caffe的AlexNet模型。</p>
    </td>
    <td class="cellrowborder" valign="top" width="74.58%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0203223265_p67311330479"><a name="zh-cn_topic_0203223265_p67311330479"></a><a name="zh-cn_topic_0203223265_p67311330479"></a>请参考<a href="https://gitee.com/HuaweiAscend/models/tree/master/computer_vision/classification/alexnet" target="_blank" rel="noopener noreferrer">https://gitee.com/HuaweiAscend/models/tree/master/computer_vision/classification/alexnet</a>目录中README.md下载原始网络模型文件及其对应的权重文件。</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0203223265_row2399521134819"><td class="cellrowborder" valign="top" width="12.85%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0203223265_p3400192113488"><a name="zh-cn_topic_0203223265_p3400192113488"></a><a name="zh-cn_topic_0203223265_p3400192113488"></a>caffenet</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.57%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0203223265_p5645133234810"><a name="zh-cn_topic_0203223265_p5645133234810"></a><a name="zh-cn_topic_0203223265_p5645133234810"></a>图片分类推理模型。</p>
    <p id="zh-cn_topic_0203223265_p14645153244816"><a name="zh-cn_topic_0203223265_p14645153244816"></a><a name="zh-cn_topic_0203223265_p14645153244816"></a>是基于Caffe的CaffeNet模型。</p>
    </td>
    <td class="cellrowborder" valign="top" width="74.58%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0203223265_p1537844912482"><a name="zh-cn_topic_0203223265_p1537844912482"></a><a name="zh-cn_topic_0203223265_p1537844912482"></a>请参考<a href="https://gitee.com/HuaweiAscend/models/tree/master/computer_vision/classification/caffenet" target="_blank" rel="noopener noreferrer">https://gitee.com/HuaweiAscend/models/tree/master/computer_vision/classification/caffenet</a>目录中README.md下载原始网络模型文件及其对应的权重文件。</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0203223265_row3773114518271"><td class="cellrowborder" valign="top" width="12.85%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0203223265_p17738455277"><a name="zh-cn_topic_0203223265_p17738455277"></a><a name="zh-cn_topic_0203223265_p17738455277"></a>densenet</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.57%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0203223265_p16773124511270"><a name="zh-cn_topic_0203223265_p16773124511270"></a><a name="zh-cn_topic_0203223265_p16773124511270"></a>图片分类推理模型。</p>
    <p id="zh-cn_topic_0203223265_p2773745162718"><a name="zh-cn_topic_0203223265_p2773745162718"></a><a name="zh-cn_topic_0203223265_p2773745162718"></a>是基于Caffe的DenseNet121模型。</p>
    </td>
    <td class="cellrowborder" valign="top" width="74.58%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0203223265_p187731945132715"><a name="zh-cn_topic_0203223265_p187731945132715"></a><a name="zh-cn_topic_0203223265_p187731945132715"></a>请参考<a href="https://gitee.com/HuaweiAscend/models/tree/master/computer_vision/classification/densenet" target="_blank" rel="noopener noreferrer">https://gitee.com/HuaweiAscend/models/tree/master/computer_vision/classification/densenet</a>目录中README.md下载原始网络模型文件及其对应的权重文件。</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0203223265_row137731845122710"><td class="cellrowborder" valign="top" width="12.85%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0203223265_p477316457275"><a name="zh-cn_topic_0203223265_p477316457275"></a><a name="zh-cn_topic_0203223265_p477316457275"></a>googlenet</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.57%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0203223265_p197731456270"><a name="zh-cn_topic_0203223265_p197731456270"></a><a name="zh-cn_topic_0203223265_p197731456270"></a>图片分类推理模型。</p>
    <p id="zh-cn_topic_0203223265_p1877394515274"><a name="zh-cn_topic_0203223265_p1877394515274"></a><a name="zh-cn_topic_0203223265_p1877394515274"></a>是基于Caffe的GoogLeNet模型。</p>
    </td>
    <td class="cellrowborder" valign="top" width="74.58%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0203223265_p197738453275"><a name="zh-cn_topic_0203223265_p197738453275"></a><a name="zh-cn_topic_0203223265_p197738453275"></a>请参考<a href="https://gitee.com/HuaweiAscend/models/tree/master/computer_vision/classification/googlenet" target="_blank" rel="noopener noreferrer">https://gitee.com/HuaweiAscend/models/tree/master/computer_vision/classification/googlenet</a>目录中README.md下载原始网络模型文件及其对应的权重文件。</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0203223265_row977374512716"><td class="cellrowborder" valign="top" width="12.85%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0203223265_p1977324512272"><a name="zh-cn_topic_0203223265_p1977324512272"></a><a name="zh-cn_topic_0203223265_p1977324512272"></a><span>inception_v2</span></p>
    </td>
    <td class="cellrowborder" valign="top" width="12.57%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0203223265_p14773445122712"><a name="zh-cn_topic_0203223265_p14773445122712"></a><a name="zh-cn_topic_0203223265_p14773445122712"></a>图片分类推理模型。</p>
    <p id="zh-cn_topic_0203223265_p877311459270"><a name="zh-cn_topic_0203223265_p877311459270"></a><a name="zh-cn_topic_0203223265_p877311459270"></a>是基于Caffe的Inception V2模型。</p>
    </td>
    <td class="cellrowborder" valign="top" width="74.58%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0203223265_p16773145162719"><a name="zh-cn_topic_0203223265_p16773145162719"></a><a name="zh-cn_topic_0203223265_p16773145162719"></a>请参考<a href="https://gitee.com/HuaweiAscend/models/tree/master/computer_vision/classification/inception_v2" target="_blank" rel="noopener noreferrer">https://gitee.com/HuaweiAscend/models/tree/master/computer_vision/classification/inception_v2</a>目录中README.md下载原始网络模型文件及其对应的权重文件。</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0203223265_row429165985115"><td class="cellrowborder" valign="top" width="12.85%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0203223265_p1050712711523"><a name="zh-cn_topic_0203223265_p1050712711523"></a><a name="zh-cn_topic_0203223265_p1050712711523"></a><span>inception_v3</span></p>
    </td>
    <td class="cellrowborder" valign="top" width="12.57%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0203223265_p18641141115218"><a name="zh-cn_topic_0203223265_p18641141115218"></a><a name="zh-cn_topic_0203223265_p18641141115218"></a>图片分类推理模型。</p>
    <p id="zh-cn_topic_0203223265_p06411511105213"><a name="zh-cn_topic_0203223265_p06411511105213"></a><a name="zh-cn_topic_0203223265_p06411511105213"></a>是基于Caffe的Inception V3模型。</p>
    </td>
    <td class="cellrowborder" valign="top" width="74.58%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0203223265_p1241971612520"><a name="zh-cn_topic_0203223265_p1241971612520"></a><a name="zh-cn_topic_0203223265_p1241971612520"></a>请参考<a href="https://gitee.com/HuaweiAscend/models/tree/master/computer_vision/classification/inception_v3" target="_blank" rel="noopener noreferrer">https://gitee.com/HuaweiAscend/models/tree/master/computer_vision/classification/inception_v3</a>目录中README.md下载原始网络模型文件及其对应的权重文件。</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0203223265_row6482142185210"><td class="cellrowborder" valign="top" width="12.85%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0203223265_p12508168115214"><a name="zh-cn_topic_0203223265_p12508168115214"></a><a name="zh-cn_topic_0203223265_p12508168115214"></a><span>inception_v4</span></p>
    </td>
    <td class="cellrowborder" valign="top" width="12.57%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0203223265_p8785612105217"><a name="zh-cn_topic_0203223265_p8785612105217"></a><a name="zh-cn_topic_0203223265_p8785612105217"></a>图片分类推理模型。</p>
    <p id="zh-cn_topic_0203223265_p47851512105212"><a name="zh-cn_topic_0203223265_p47851512105212"></a><a name="zh-cn_topic_0203223265_p47851512105212"></a>是基于Caffe的Inception V4模型。</p>
    </td>
    <td class="cellrowborder" valign="top" width="74.58%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0203223265_p1028761705212"><a name="zh-cn_topic_0203223265_p1028761705212"></a><a name="zh-cn_topic_0203223265_p1028761705212"></a>请参考<a href="https://gitee.com/HuaweiAscend/models/tree/master/computer_vision/classification/inception_v4" target="_blank" rel="noopener noreferrer">https://gitee.com/HuaweiAscend/models/tree/master/computer_vision/classification/inception_v4</a>目录中README.md下载原始网络模型文件及其对应的权重文件。</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0203223265_row77732045152717"><td class="cellrowborder" valign="top" width="12.85%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0203223265_p0773114572715"><a name="zh-cn_topic_0203223265_p0773114572715"></a><a name="zh-cn_topic_0203223265_p0773114572715"></a><span>mobilenet_v1</span></p>
    </td>
    <td class="cellrowborder" valign="top" width="12.57%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0203223265_p11774645162715"><a name="zh-cn_topic_0203223265_p11774645162715"></a><a name="zh-cn_topic_0203223265_p11774645162715"></a>图片分类推理模型。</p>
    <p id="zh-cn_topic_0203223265_p47741455273"><a name="zh-cn_topic_0203223265_p47741455273"></a><a name="zh-cn_topic_0203223265_p47741455273"></a>是基于Caffe的MobileNet V1模型。</p>
    </td>
    <td class="cellrowborder" valign="top" width="74.58%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0203223265_p5586471511"><a name="zh-cn_topic_0203223265_p5586471511"></a><a name="zh-cn_topic_0203223265_p5586471511"></a>请参考<a href="https://gitee.com/HuaweiAscend/models/tree/master/computer_vision/classification/mobilenet_v1" target="_blank" rel="noopener noreferrer">https://gitee.com/HuaweiAscend/models/tree/master/computer_vision/classification/mobilenet_v1</a>目录中README.md下载原始网络模型文件及其对应的权重文件。</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0203223265_row12774164515277"><td class="cellrowborder" valign="top" width="12.85%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0203223265_p187741345112718"><a name="zh-cn_topic_0203223265_p187741345112718"></a><a name="zh-cn_topic_0203223265_p187741345112718"></a><span>mobilenet_v2</span></p>
    </td>
    <td class="cellrowborder" valign="top" width="12.57%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0203223265_p277414519274"><a name="zh-cn_topic_0203223265_p277414519274"></a><a name="zh-cn_topic_0203223265_p277414519274"></a>图片分类推理模型。</p>
    <p id="zh-cn_topic_0203223265_p8774174502713"><a name="zh-cn_topic_0203223265_p8774174502713"></a><a name="zh-cn_topic_0203223265_p8774174502713"></a>是基于Caffe的MobileNet V2模型。</p>
    </td>
    <td class="cellrowborder" valign="top" width="74.58%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0203223265_p1677414514274"><a name="zh-cn_topic_0203223265_p1677414514274"></a><a name="zh-cn_topic_0203223265_p1677414514274"></a>请参考<a href="https://gitee.com/HuaweiAscend/models/tree/master/computer_vision/classification/mobilenet_v2" target="_blank" rel="noopener noreferrer">https://gitee.com/HuaweiAscend/models/tree/master/computer_vision/classification/mobilenet_v2</a>目录中README.md下载原始网络模型文件及其对应的权重文件。</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0203223265_row1577434516271"><td class="cellrowborder" valign="top" width="12.85%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0203223265_p3774194512713"><a name="zh-cn_topic_0203223265_p3774194512713"></a><a name="zh-cn_topic_0203223265_p3774194512713"></a><span>resnet18</span></p>
    </td>
    <td class="cellrowborder" valign="top" width="12.57%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0203223265_p7774245122713"><a name="zh-cn_topic_0203223265_p7774245122713"></a><a name="zh-cn_topic_0203223265_p7774245122713"></a>图片分类推理模型。</p>
    <p id="zh-cn_topic_0203223265_p577494517271"><a name="zh-cn_topic_0203223265_p577494517271"></a><a name="zh-cn_topic_0203223265_p577494517271"></a>是基于Caffe的ResNet 18模型。</p>
    </td>
    <td class="cellrowborder" valign="top" width="74.58%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0203223265_p16774144510270"><a name="zh-cn_topic_0203223265_p16774144510270"></a><a name="zh-cn_topic_0203223265_p16774144510270"></a>请参考<a href="https://gitee.com/HuaweiAscend/models/tree/master/computer_vision/classification/resnet18" target="_blank" rel="noopener noreferrer">https://gitee.com/HuaweiAscend/models/tree/master/computer_vision/classification/resnet18</a>目录中README.md下载原始网络模型文件及其对应的权重文件。</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0203223265_row377414452276"><td class="cellrowborder" valign="top" width="12.85%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0203223265_p10774545142714"><a name="zh-cn_topic_0203223265_p10774545142714"></a><a name="zh-cn_topic_0203223265_p10774545142714"></a><span>resnet50</span></p>
    </td>
    <td class="cellrowborder" valign="top" width="12.57%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0203223265_p97741245142712"><a name="zh-cn_topic_0203223265_p97741245142712"></a><a name="zh-cn_topic_0203223265_p97741245142712"></a>图片分类推理模型。</p>
    <p id="zh-cn_topic_0203223265_p177412456271"><a name="zh-cn_topic_0203223265_p177412456271"></a><a name="zh-cn_topic_0203223265_p177412456271"></a>是基于Caffe的ResNet 50模型。</p>
    </td>
    <td class="cellrowborder" valign="top" width="74.58%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0203223265_p57747459272"><a name="zh-cn_topic_0203223265_p57747459272"></a><a name="zh-cn_topic_0203223265_p57747459272"></a>请参考<a href="https://gitee.com/HuaweiAscend/models/tree/master/computer_vision/classification/resnet50" target="_blank" rel="noopener noreferrer">https://gitee.com/HuaweiAscend/models/tree/master/computer_vision/classification/resnet50</a>目录中README.md下载原始网络模型文件及其对应的权重文件。</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0203223265_row377484514279"><td class="cellrowborder" valign="top" width="12.85%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0203223265_p1777454516275"><a name="zh-cn_topic_0203223265_p1777454516275"></a><a name="zh-cn_topic_0203223265_p1777454516275"></a><span>resnet101</span></p>
    </td>
    <td class="cellrowborder" valign="top" width="12.57%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0203223265_p15774124516274"><a name="zh-cn_topic_0203223265_p15774124516274"></a><a name="zh-cn_topic_0203223265_p15774124516274"></a>图片分类推理模型。</p>
    <p id="zh-cn_topic_0203223265_p7774134552720"><a name="zh-cn_topic_0203223265_p7774134552720"></a><a name="zh-cn_topic_0203223265_p7774134552720"></a>是基于Caffe的ResNet 101模型。</p>
    </td>
    <td class="cellrowborder" valign="top" width="74.58%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0203223265_p117741545132710"><a name="zh-cn_topic_0203223265_p117741545132710"></a><a name="zh-cn_topic_0203223265_p117741545132710"></a>请参考<a href="https://gitee.com/HuaweiAscend/models/tree/master/computer_vision/classification/resnet101" target="_blank" rel="noopener noreferrer">https://gitee.com/HuaweiAscend/models/tree/master/computer_vision/classification/resnet101</a>目录中README.md下载原始网络模型文件及其对应的权重文件。</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0203223265_row14774154513279"><td class="cellrowborder" valign="top" width="12.85%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0203223265_p1077413452272"><a name="zh-cn_topic_0203223265_p1077413452272"></a><a name="zh-cn_topic_0203223265_p1077413452272"></a><span>resnet15</span>2</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.57%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0203223265_p177434517275"><a name="zh-cn_topic_0203223265_p177434517275"></a><a name="zh-cn_topic_0203223265_p177434517275"></a>图片分类推理模型。</p>
    <p id="zh-cn_topic_0203223265_p877515459276"><a name="zh-cn_topic_0203223265_p877515459276"></a><a name="zh-cn_topic_0203223265_p877515459276"></a>是基于Caffe的ResNet 152模型。</p>
    </td>
    <td class="cellrowborder" valign="top" width="74.58%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0203223265_p777514582712"><a name="zh-cn_topic_0203223265_p777514582712"></a><a name="zh-cn_topic_0203223265_p777514582712"></a>请参考<a href="https://gitee.com/HuaweiAscend/models/tree/master/computer_vision/classification/resnet152" target="_blank" rel="noopener noreferrer">https://gitee.com/HuaweiAscend/models/tree/master/computer_vision/classification/resnet152</a>目录中README.md下载原始网络模型文件及其对应的权重文件。</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0203223265_row37752450270"><td class="cellrowborder" valign="top" width="12.85%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0203223265_p977544513278"><a name="zh-cn_topic_0203223265_p977544513278"></a><a name="zh-cn_topic_0203223265_p977544513278"></a><span>vgg16</span></p>
    </td>
    <td class="cellrowborder" valign="top" width="12.57%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0203223265_p1177514522713"><a name="zh-cn_topic_0203223265_p1177514522713"></a><a name="zh-cn_topic_0203223265_p1177514522713"></a>图片分类推理模型。</p>
    <p id="zh-cn_topic_0203223265_p10775194582713"><a name="zh-cn_topic_0203223265_p10775194582713"></a><a name="zh-cn_topic_0203223265_p10775194582713"></a>是基于Caffe的VGG16模型。</p>
    </td>
    <td class="cellrowborder" valign="top" width="74.58%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0203223265_p18775124582720"><a name="zh-cn_topic_0203223265_p18775124582720"></a><a name="zh-cn_topic_0203223265_p18775124582720"></a>请参考<a href="https://gitee.com/HuaweiAscend/models/tree/master/computer_vision/classification/vgg16" target="_blank" rel="noopener noreferrer">https://gitee.com/HuaweiAscend/models/tree/master/computer_vision/classification/vgg16</a>目录中README.md下载原始网络模型文件及其对应的权重文件。</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0203223265_row2775194518272"><td class="cellrowborder" valign="top" width="12.85%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0203223265_p5775154516272"><a name="zh-cn_topic_0203223265_p5775154516272"></a><a name="zh-cn_topic_0203223265_p5775154516272"></a><span>vgg19</span></p>
    </td>
    <td class="cellrowborder" valign="top" width="12.57%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0203223265_p0775204532711"><a name="zh-cn_topic_0203223265_p0775204532711"></a><a name="zh-cn_topic_0203223265_p0775204532711"></a>图片分类推理模型。</p>
    <p id="zh-cn_topic_0203223265_p1477554519275"><a name="zh-cn_topic_0203223265_p1477554519275"></a><a name="zh-cn_topic_0203223265_p1477554519275"></a>是基于Caffe的VGG19模型。</p>
    </td>
    <td class="cellrowborder" valign="top" width="74.58%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0203223265_p777554542713"><a name="zh-cn_topic_0203223265_p777554542713"></a><a name="zh-cn_topic_0203223265_p777554542713"></a>请参考<a href="https://gitee.com/HuaweiAscend/models/tree/master/computer_vision/classification/vgg19" target="_blank" rel="noopener noreferrer">https://gitee.com/HuaweiAscend/models/tree/master/computer_vision/classification/vgg19</a>目录中README.md下载原始网络模型文件及其对应的权重文件。</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0203223265_row17513194404914"><td class="cellrowborder" valign="top" width="12.85%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0203223265_p7513164419495"><a name="zh-cn_topic_0203223265_p7513164419495"></a><a name="zh-cn_topic_0203223265_p7513164419495"></a>squeezenet</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.57%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0203223265_p1315111145015"><a name="zh-cn_topic_0203223265_p1315111145015"></a><a name="zh-cn_topic_0203223265_p1315111145015"></a>图片分类推理模型。</p>
    <p id="zh-cn_topic_0203223265_p1515131114501"><a name="zh-cn_topic_0203223265_p1515131114501"></a><a name="zh-cn_topic_0203223265_p1515131114501"></a>是基于Caffe的SqueezeNet模型。</p>
    </td>
    <td class="cellrowborder" valign="top" width="74.58%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0203223265_p16265437125015"><a name="zh-cn_topic_0203223265_p16265437125015"></a><a name="zh-cn_topic_0203223265_p16265437125015"></a>请参考<a href="https://gitee.com/HuaweiAscend/models/tree/master/computer_vision/classification/squeezenet" target="_blank" rel="noopener noreferrer">https://gitee.com/HuaweiAscend/models/tree/master/computer_vision/classification/squeezenet</a>目录中README.md下载原始网络模型文件及其对应的权重文件。</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0203223265_row17757454270"><td class="cellrowborder" valign="top" width="12.85%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0203223265_p17759452279"><a name="zh-cn_topic_0203223265_p17759452279"></a><a name="zh-cn_topic_0203223265_p17759452279"></a><span>dpn98</span></p>
    </td>
    <td class="cellrowborder" valign="top" width="12.57%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0203223265_p4775545162716"><a name="zh-cn_topic_0203223265_p4775545162716"></a><a name="zh-cn_topic_0203223265_p4775545162716"></a>图片分类推理模型。</p>
    <p id="zh-cn_topic_0203223265_p1577504516278"><a name="zh-cn_topic_0203223265_p1577504516278"></a><a name="zh-cn_topic_0203223265_p1577504516278"></a>是基于Caffe的<span>dpn98</span>模型。</p>
    </td>
    <td class="cellrowborder" valign="top" width="74.58%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0203223265_p19776154592711"><a name="zh-cn_topic_0203223265_p19776154592711"></a><a name="zh-cn_topic_0203223265_p19776154592711"></a>请参考<a href="https://gitee.com/HuaweiAscend/models/tree/master/computer_vision/classification/dpn98" target="_blank" rel="noopener noreferrer">https://gitee.com/HuaweiAscend/models/tree/master/computer_vision/classification/dpn98</a>目录中README.md下载原始网络模型文件及其对应的权重文件。</p>
    </td>
    </tr>
    </tbody>
    </table>

3.  以Mind Studio安装用户登录Mind Studio所在Ubuntu服务器，确定当前使用的DDK版本号并设置环境变量DDK\_HOME，tools\_version，NPU\_DEVICE\_LIB和LD\_LIBRARY\_PATH。
    1.  <a name="zh-cn_topic_0203223265_zh-cn_topic_0203223294_li61417158198"></a>查询当前使用的DDK版本号。

        可通过Mind Studio工具查询，也可以通过DDK软件包进行获取。

        -   使用Mind Studio工具查询。

            在Mind Studio工程界面依次选择“File \> Settings \> System Settings \> Ascend DDK“，弹出如[图 DDK版本号查询](zh-cn_topic_0203223294.md#fig94023140222)所示界面。

            **图 1**  DDK版本号查询<a name="zh-cn_topic_0203223265_zh-cn_topic_0203223294_fig17553193319118"></a>  
            ![](figures/DDK版本号查询.png "DDK版本号查询")

            其中显示的**DDK Version**就是当前使用的DDK版本号，如**1.31.T15.B150**。

        -   通过DDK软件包进行查询。

            通过安装的DDK的包名获取DDK的版本号。

            DDK包的包名格式为：**Ascend\_DDK-\{software version\}-\{interface version\}-x86\_64.ubuntu16.04.tar.gz**

            其中**software version**就是DDK的软件版本号。

            例如：

            DDK包的包名为Ascend\_DDK-1.31.T15.B150-1.1.1-x86\_64.ubuntu16.04.tar.gz，则此DDK的版本号为1.31.T15.B150。

    2.  设置环境变量。

        **vim \~/.bashrc**

        执行如下命令在最后一行添加DDK\_HOME及LD\_LIBRARY\_PATH的环境变量。

        **export tools\_version=_1.31.X.X_**

        **export DDK\_HOME=$HOME/.mindstudio/huawei/ddk/_1.31.X.X_/ddk**

        **export NPU\_DEVICE\_LIB=$DDK\_HOME/../RC/host-aarch64\_Ubuntu16.04.3/lib**

        **export LD\_LIBRARY\_PATH=$DDK\_HOME/lib/x86\_64-linux-gcc5.4**

        >![](public_sys-resources/icon-note.gif) **说明：**   
        >-   **_1.31.X.X_**是[1](#zh-cn_topic_0203223265_zh-cn_topic_0203223294_li61417158198)中查询到的DDK版本号，需要根据查询结果对应填写，如**1.31.T15.B150**  
        >-   如果此环境变量已经添加，则此步骤可跳过。  

        输入**:wq!**保存退出。

        执行如下命令使环境变量生效。

        **source \~/.bashrc**

4.  将原始网络模型转换为适配昇腾AI处理器的模型。

    1.  在Mind Studio操作界面的顶部菜单栏中选择**Tool \> Convert Model**，进入模型转换界面。

        ![](figures/zh-cn_image_0205108796.png)

    2.  在弹出的**Convert Model**操作界面中，进行模型转换配置。
        -   Model File选择[步骤2](#zh-cn_topic_0203223265_li29641938112018)中下载的模型文件，此时会自动匹配到权重文件并填写在Weight File中。
        -   Model Name填写为[表1](#zh-cn_topic_0203223265_table1119094515272)中对应的**模型名称**。

            ![](figures/zh-cn_image_0208264607.png)

        -   googlenet、inception\_v2、resent18模型转换时中，由于通用分类网络应用一次处理一张图片，所以转换时需要将Nodes配置中的**N**修改为1，AIPP配置中的**Input Image Size\[W\]\[H\]**需要分别修改为256、224，此处需要128\*16对齐，**Model Image Format**  选择BGR888\_U8。

            **图 2**  Nodes配置示例<a name="zh-cn_topic_0203223265_fig95695336322"></a>  
            ![](figures/Nodes配置示例.png "Nodes配置示例")

            **图 3**  AIPP配置示例<a name="zh-cn_topic_0203223265_fig14632122193310"></a>  
            ![](figures/AIPP配置示例.png "AIPP配置示例")

    3.  单击OK开始转换模型。

        模型转换成功后，后缀为.om的离线模型存放地址为_$HOME/modelzoo/XXX__/device_。

    >![](public_sys-resources/icon-note.gif) **说明：**   
    >-   Mindstudio模型转换中每一步的具体意义和参数说明可以参考[https://ascend.huawei.com/doc/mindstudio/2.1.0\(beta\)/zh/zh-cn\_topic\_0188462651.html](https://ascend.huawei.com/doc/mindstudio/2.1.0(beta)/zh/zh-cn_topic_0188462651.html)  
    >-   XXX表示当前转换的模型名称，如googlenet.om存放地址为$HOME/modelzoo/googlenet/device。  

5.  <a name="zh-cn_topic_0203223265_li470213205618"></a>将转换好的模型文件（.om文件）上传到[步骤1](#zh-cn_topic_0203223265_li953280133816)中源码所在路径下的“**sample-classification/script**”目录下。

## 编译<a name="zh-cn_topic_0203223265_section18931344873"></a>

1.  打开对应的工程。

    以Mind Studio安装用户在命令行进入安装包解压后的“MindStudio-ubuntu/bin”目录，如$HOME/MindStudio-ubuntu/bin。执行如下命令启动Mind Studio。

    **./MindStudio.sh**

    启动成功后，打开**sample-classification**工程，如[图 打开classification工程](#zh-cn_topic_0203223265_fig11106241192810)所示。

    **图 4**  打开classification工程<a name="zh-cn_topic_0203223265_fig11106241192810"></a>  
    ![](figures/打开classification工程.png "打开classification工程")

2.  在**src/param\_configure.conf**文件中配置相关工程信息。

    **图 5**  配置文件路径<a name="zh-cn_topic_0203223265_fig0391184062214"></a>  
    ![](figures/配置文件路径.png "配置文件路径")

    该配置文件内容如下：

    ```
    remote_host=
    model_name=
    ```

    需要手动添加参数配置：

    -   remote\_host：Atlas 200 DK开发者板的IP地址。
    -   model\_name : 离线模型名称。

    配置示例：

    ```
    remote_host=192.168.1.2
    model_name=googlenet.om
    ```

    >![](public_sys-resources/icon-note.gif) **说明：**   
    >-   参数必须全部填写，否则无法通过build。  
    >-   注意参数填写时不需要使用“”符号。  
    >-   配置文件中只能填入单个模型名称，填入的模型必须为[步骤5](#zh-cn_topic_0203223265_li470213205618)中存储的模型之一。本示例是以googlenet举例，用户可以使用本样例列举的其它模型按照文档步骤进行替换运行。  

3.  开始编译，打开Mind Studio工具，在工具栏中点击**Build \> Build \> Build-Configuration**。如[图 编译操作及生成文件](#zh-cn_topic_0203223265_fig1741464713019)所示，会在目录下生成build和run文件夹。

    **图 6**  编译操作及生成文件<a name="zh-cn_topic_0203223265_fig1741464713019"></a>  
    ![](figures/编译操作及生成文件.png "编译操作及生成文件")

    >![](public_sys-resources/icon-note.gif) **说明：**   
    >首次编译工程时，**Build \> Build**为灰色不可点击状态。需要点击**Build \> Edit Build Configuration**，配置编译参数后再进行编译。  
    >![](figures/build_configuration.png)  

4.  将需要推理的图片上传至Host侧任一属组为HwHiAiUser用户的目录。

    图片要求如下：

    -   格式：jpg、png、bmp。
    -   输入图片宽度：16px\~4096px之间的整数。
    -   输入图片高度：16px\~4096px之间的整数。


## 运行<a name="zh-cn_topic_0203223265_section372782554919"></a>

1.  在Mind Studio工具的工具栏中找到Run按钮，单击  **Run \> Run 'sample-classification'**，如[图 程序已执行示意图](#zh-cn_topic_0203223265_fig93931954162719)所示，可执行程序已经在开发者板执行。

    **图 7**  程序已执行示意图<a name="zh-cn_topic_0203223265_fig93931954162719"></a>  
    ![](figures/程序已执行示意图.png "程序已执行示意图")

    以上报错信息请忽略，因为Mind Studio无法为可执行程序传参，上述步骤是将可执行程序与依赖的库文件部署到开发者板，此步骤需要ssh登录到开发者板至相应的目录文件下手动执行，具体请参考以下步骤。

2.  在Mind Studio所在Ubuntu服务器中，以HwHiAiUser用户SSH登录到Host侧。

    **ssh HwHiAiUser@**_host\_ip_

    对于Atlas 200 DK，host\_ip默认为192.168.1.2（USB连接）或者192.168.0.2（NIC连接）。

3.  进入通用分类网络应用的可执行文件所在路径。

    **cd \~/HIAI\_PROJECTS/workspace\_mind\_studio/classification/out**

4.  执行应用程序。

    执行**run\_classification.py**脚本会将推理结果在执行终端直接打印显示。

    命令示例如下所示：

    **python3 run\_classification.py -w  _224_  -h  _224_  -i** **_./example.jpg_  -n  _10_**

    -   -w/model\_width：模型的输入图片宽度，为16\~4096之间的整数，请参考[表1](#zh-cn_topic_0203223265_table1119094515272)在Gitee上查看所使用模型文件的Readme，获取模型要求的输入数据的宽和高。
    -   -h/model\_height：模型的输入图片高度，为16\~4096之间的整数，请参考[表1](#zh-cn_topic_0203223265_table1119094515272)在Gitee上查看所使用模型文件的Readme，获取模型要求的输入数据的宽和高。
    -   -i/input\_path：输入图片的路径，可以是目录，表示当前目录下的所有图片都作为输入（可以指定多个输入）。
    -   -n/top\_n：输出前n个推理结果。

    其他详细参数请执行**python3 run\_classification.py --help**命令参见帮助信息。


