#ifndef MINI_R1_V1_APPLICATION__MISSION_CONTROL_PANEL_HPP_
#define MINI_R1_V1_APPLICATION__MISSION_CONTROL_PANEL_HPP_

#include <rviz_common/panel.hpp>
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <QTableWidget>
#include <QVBoxLayout>
#include <QLabel>
#include <set>

namespace mini_r1_v1_application
{

class MissionControlPanel : public rviz_common::Panel
{
  Q_OBJECT

public:
  explicit MissionControlPanel(QWidget * parent = nullptr);
  virtual ~MissionControlPanel();

  void onInitialize() override;

protected:
  // ROS Node for subscribing
  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr sub_markers_;

  // UI Elements
  QTableWidget* table_;
  std::set<std::string> logged_items_;

  // Callback
  void markerCallback(const visualization_msgs::msg::MarkerArray::SharedPtr msg);
  
private Q_SLOTS:
  // Thread safe GUI update
  void addRowToTable(const QString& type, int id, double x, double y, double z);

Q_SIGNALS:
  void newMarkerDetected(const QString& type, int id, double x, double y, double z);

};

}  // namespace mini_r1_v1_application

#endif  // MINI_R1_V1_APPLICATION__MISSION_CONTROL_PANEL_HPP_
