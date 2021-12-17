# netvlad_ros
rospy 实现netvlad接入
# 1 使用python3 编译cv_bridge
# 2 下载模型和pca文件
# 3 修改订阅图片节点和模型
 global_feature_node.py 28 line 修改数据集需要的节点
 topics = rospy.get_param('~topics', '/cam0/image_raw')  
 global_feature_node.py 47 修改使用的模型目前支持的参数
  'vgg16' 'mobilenet_v2' 'mbv2_ca'
# 4 修改发布节点
global_feature_node.py 75 
默认使用  '/d400/features'
# 5 修改模型路径
global_feature_node.py 26
net_path = rospy.get_param('path','/home/lhw/ros/netvlad_ros/src/deepfeature/global_feature/')
