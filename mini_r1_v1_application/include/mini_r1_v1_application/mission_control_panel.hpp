#ifndef MINI_R1_V1_APPLICATION__MISSION_CONTROL_PANEL_HPP_
#define MINI_R1_V1_APPLICATION__MISSION_CONTROL_PANEL_HPP_

#include <rviz_common/panel.hpp>
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_msgs/msg/string.hpp>
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
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_mission_status_;

  // UI Elements
  QTableWidget* table_;
  QLabel* status_label_;
  std::set<std::string> logged_items_;
  bool mission_complete_{false};

  // Callbacks
  void markerCallback(const visualization_msgs::msg::MarkerArray::SharedPtr msg);
  void missionStatusCallback(const std_msgs::msg::String::SharedPtr msg);

private Q_SLOTS:
  // Thread safe GUI update
  void addRowToTable(const QString& type, int id, double x, double y, double z);
  void showMissionComplete();

Q_SIGNALS:
  void newMarkerDetected(const QString& type, int id, double x, double y, double z);
  void missionCompleteReceived();

};

}  // namespace mini_r1_v1_application

#endif  // MINI_R1_V1_APPLICATION__MISSION_CONTROL_PANEL_HPP_
