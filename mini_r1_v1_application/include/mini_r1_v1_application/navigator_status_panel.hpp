#ifndef MINI_R1_V1_APPLICATION__NAVIGATOR_STATUS_PANEL_HPP_
#define MINI_R1_V1_APPLICATION__NAVIGATOR_STATUS_PANEL_HPP_

#include <rviz_common/panel.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <QVBoxLayout>
#include <QLabel>
#include <QTextEdit>

namespace mini_r1_v1_application
{

class NavigatorStatusPanel : public rviz_common::Panel
{
  Q_OBJECT

public:
  explicit NavigatorStatusPanel(QWidget * parent = nullptr);
  virtual ~NavigatorStatusPanel();
  void onInitialize() override;

protected:
  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_status_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_vlm_;

  // Navigator section
  QLabel * state_label_;
  QLabel * behavior_label_;
  QLabel * detectors_label_;
  QLabel * recovery_label_;
  QLabel * stats_label_;
  QLabel * position_label_;
  QTextEdit * reasoning_text_;

  // VLM section
  QLabel * vlm_provider_label_;
  QLabel * vlm_action_label_;
  QLabel * vlm_tools_label_;
  QTextEdit * vlm_reasoning_text_;

  void statusCallback(const std_msgs::msg::String::SharedPtr msg);
  void vlmCallback(const std_msgs::msg::String::SharedPtr msg);

private Q_SLOTS:
  void updateDisplay(const QString & json_str);
  void updateVLMDisplay(const QString & json_str);

Q_SIGNALS:
  void statusReceived(const QString & json_str);
  void vlmStatusReceived(const QString & json_str);
};

}  // namespace mini_r1_v1_application

#endif
