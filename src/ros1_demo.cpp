#include "netTensorRT.hpp"
#include "pointcloud_io.h"
#include "ros/ros.h"
#include <filesystem>
#include <functional>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>

//rgb
std::map<std::string, float> semantic_map{
  {"666", 0.0},
  // 动态物体
  {"100150245", 0.2},
  {"100230245", 0.2},
  {"10080250", 0.2},
  {"3060150", 0.2},
  {"00255", 0.2},
  {"8030180", 0.2},
  {"2553030", 0.2},
  {"25540200", 0.2},
  {"1503090", 0.2},
  // 柱状物
  {"255240150", 0.4},
  {"135600", 0.4},
  // 地面
  {"2550255", 0.6},
  {"75075", 0.6}, // 人行道
  {"255150255", 0.6},
  {"175075", 0.6},
  // 建筑平面
  {"25512050", 0.8},
  {"2552000", 0.8},
  {"2551500", 0.8},
  // 其他,如树木、植被等
  {"01750", 1.0},
  {"15024080", 1.0}
};

// 定义颜色结构体
struct Color {
    uint8_t r;
    uint8_t g;
    uint8_t b;

    bool operator<(const Color& other) const {
        return std::tie(r, g, b) < std::tie(other.r, other.g, other.b);
    }
};

// 定义类别标签
enum class Label {
    Other,
    Dynamic,
    Pillar,
    Ground,
    Construction,
    Vegetation
    // 添加更多类别
};


// 定义颜色到类别的映射
std::map<Color, Label> color_to_label = {
    {{6, 6, 6}, Label::Other},

    {{100, 150, 245}, Label::Dynamic},
    {{100, 230, 245}, Label::Dynamic},
    {{100, 80, 250}, Label::Dynamic},
    {{30, 60, 150}, Label::Dynamic},
    {{0, 0, 255}, Label::Dynamic},
    {{80, 30, 180}, Label::Dynamic},
    {{255, 30, 30}, Label::Dynamic},
    {{255, 40, 200}, Label::Dynamic},
    {{150, 30, 90}, Label::Dynamic},

    {{255, 240, 150}, Label::Pillar},
    {{135, 60, 0}, Label::Pillar},

    {{255, 0, 255}, Label::Ground},
    {{75, 0, 75}, Label::Ground},
    {{255, 150, 255}, Label::Ground},
    {{175, 0, 75}, Label::Ground},

    {{255, 120, 50}, Label::Construction},
    {{255, 200, 0}, Label::Construction},
    {{255, 150, 0}, Label::Construction},

    {{0, 175, 0}, Label::Vegetation},
    {{150, 240, 80}, Label::Vegetation}

};


// 定义类别到新颜色的映射
std::map<Label, Color> label_to_new_color = {

    {Label::Other, {48, 48, 48}},
    {Label::Dynamic, {183, 196, 16}},
    {Label::Pillar, {0, 191, 46}},
    {Label::Ground, {11, 180, 192}},
    {Label::Construction, {9, 1, 231}},
    {Label::Vegetation, {182, 15, 248}}

};



class ROS_DEMO {
public:
  explicit ROS_DEMO(ros::NodeHandle *pnh);

private:
  void pointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr &pc_msg);
  ros::NodeHandle *pnh_;
  ros::Publisher pub_;
  //2.24myz
  ros::Publisher pub_classifiedRGB;
  ros::Subscriber sub_;
  std::unique_ptr<rangenet::segmentation::Net> net_;
};

ROS_DEMO::ROS_DEMO(ros::NodeHandle *pnh) : pnh_(pnh) {

  //velodyne
  // std::string point_topic = "/points_raw";

  //hesai
  std::string point_topic = "/hesai/pandar";

  std::filesystem::path file_path(__FILE__);
  std::string model_dir = std::string(file_path.parent_path().parent_path() / "model/Hesai-Model/");
  ROS_INFO("model_dir: %s", model_dir.c_str());

  sub_ = pnh_->subscribe<sensor_msgs::PointCloud2>(point_topic, 10, &ROS_DEMO::pointcloudCallback, this);
  pub_ = pnh_->advertise<sensor_msgs::PointCloud2>("/label_pointcloud", 1, true);

  //2.24myz
  pub_classifiedRGB = pnh_->advertise<sensor_msgs::PointCloud2>("/label_pointcloud_rgb", 1, true);

  net_ = std::unique_ptr<rangenet::segmentation::Net>(new rangenet::segmentation::NetTensorRT(model_dir, false));
};

void ROS_DEMO::pointcloudCallback(
  const sensor_msgs::PointCloud2::ConstPtr &pc_msg) {

  //velodyne
  // int offset_intensity = 12;
  // int offset_ring = 16;
  // int ring_index = 4;

  //hesai
  int offset_intensity = 16;
  int offset_ring = 32;
  int ring_index = 5;

  // 获取intensity,ring字段 fields
  std::vector<float> intensity;
  std::vector<uint16_t> ring;
  intensity.resize(pc_msg->width * pc_msg->height);
  ring.resize(pc_msg->width * pc_msg->height);
  for(int i = 0; i < pc_msg->width * pc_msg->height; i++) {
    memcpy(&intensity[i], &pc_msg->data[i * pc_msg->point_step + offset_intensity], sizeof(float));
    memcpy(&ring[i], &pc_msg->data[i * pc_msg->point_step + offset_ring], sizeof(uint16_t));
  }
  sensor_msgs::PointField intensity_field, ring_field;
  intensity_field = pc_msg->fields[3];
  ring_field = pc_msg->fields[ring_index];
  cout<<pc_msg->fields<<endl;

  // ROS 消息类型 -> PCL 点云类型
  pcl::PointCloud<PointType>::Ptr pc_ros(new pcl::PointCloud<PointType>());
  pcl::fromROSMsg(*pc_msg, *pc_ros);

  // 语义分割
  auto labels = std::make_unique<int[]>(pc_ros->size());
  net_->doInfer(*pc_ros, labels.get());
  pcl::PointCloud<pcl::PointXYZRGB> color_pc;

  // 发布点云
  sensor_msgs::PointCloud2 rosrgb_msg;
  dynamic_cast<rangenet::segmentation::NetTensorRT *>(net_.get())->paintPointCloud(*pc_ros, color_pc, labels.get());
  pcl::toROSMsg(color_pc, rosrgb_msg);
  rosrgb_msg.header = pc_msg->header;

  sensor_msgs::PointCloud2 ros_msg = rosrgb_msg;//写入intensity用


  pcl::PointCloud<pcl::PointXYZRGB> classified_cloud;
  for (const auto& point : color_pc.points) {
    Color point_color = {point.r, point.g, point.b};

    auto it = color_to_label.find(point_color);
    if (it != color_to_label.end()) {

      Label label = it->second;
      Color new_color = label_to_new_color[label];

      pcl::PointXYZRGB new_point;
      new_point.x = point.x;
      new_point.y = point.y;
      new_point.z = point.z;
      new_point.r = new_color.r;
      new_point.g = new_color.g;
      new_point.b = new_color.b;

      classified_cloud.points.push_back(new_point);
    }
  }

  // 更新点云的宽度和高度
  classified_cloud.width = classified_cloud.points.size();
  classified_cloud.height = 1;

  sensor_msgs::PointCloud2 rosclassifiedRGB_msg;
  pcl::toROSMsg(classified_cloud, rosclassifiedRGB_msg);
  rosclassifiedRGB_msg.header = pc_msg->header;
  pub_classifiedRGB.publish(rosclassifiedRGB_msg);


  // 插入ring字段，将RGB变成intensity幻影坦克
  ros_msg.fields.resize(ros_msg.fields.size() + 1);
  ros_msg.fields[3].name = "intensity";
  ros_msg.fields[4] = ring_field;
  ros_msg.fields[4].offset = 20;

  // RGB解码方法
  float rgbtemp = 0;
  for(size_t i = 0;i < rosrgb_msg.width * rosrgb_msg.height; ++i){
    float zero_intensity = 0;
    memcpy(&rgbtemp, &rosrgb_msg.data[i * rosrgb_msg.point_step + 16], sizeof(float));
    uint32_t rgbtemp2 = reinterpret_cast<uint32_t &>(rgbtemp);
    int b = rgbtemp2 & 0xFF;
    int g = (rgbtemp2 >> 8) & 0xFF;
    int r = (rgbtemp2 >> 16) & 0xFF;
    std::string cmpstr = std::to_string(r) + std::to_string(g) + std::to_string(b);
    if(semantic_map.find(cmpstr) != semantic_map.end()){
      memcpy(&rosrgb_msg.data[i * rosrgb_msg.point_step + 16], &semantic_map[cmpstr], sizeof(float));
    }
    else{
      memcpy(&rosrgb_msg.data[i * rosrgb_msg.point_step + 16], &semantic_map["666"], sizeof(float));
    }
  }

  // ros_msg.point_step += 2;
  // ros_msg.row_step = ros_msg.point_step * ros_msg.width;

  std::vector<uint8_t> new_data(ros_msg.row_step * ros_msg.height);

  for(size_t i = 0; i < rosrgb_msg.width * rosrgb_msg.height; ++i) {
    memcpy(&new_data[i * ros_msg.point_step], &rosrgb_msg.data[i * rosrgb_msg.point_step], rosrgb_msg.point_step);
    memcpy(&new_data[i * ros_msg.point_step + 20], &ring[i], sizeof(uint16_t));
  }
  ros_msg.data = new_data;

  pub_.publish(ros_msg);
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "ros1_demo");
  ros::NodeHandle pnh("~");
  ROS_DEMO node(&pnh);
  ros::spin();
  return 0;
}
