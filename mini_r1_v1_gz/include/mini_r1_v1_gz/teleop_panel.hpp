#ifndef MINI_R1_V1_GZ_TELEOP_PANEL_HPP
#define MINI_R1_V1_GZ_TELEOP_PANEL_HPP

#ifndef Q_MOC_RUN
# include <rclcpp/rclcpp.hpp>
# include <rviz_common/panel.hpp>
# include <geometry_msgs/msg/twist.hpp>
#endif

#include <QWidget>
#include <QKeyEvent>
#include <QTimer>

namespace mini_r1_v1_gz
{

class TeleopPanel : public rviz_common::Panel
{
  Q_OBJECT
public:
  explicit TeleopPanel(QWidget * parent = 0);
  virtual ~TeleopPanel();

  virtual void onInitialize() override;

protected:
  virtual void keyPressEvent(QKeyEvent * event) override;
  virtual void keyReleaseEvent(QKeyEvent * event) override;

protected Q_SLOTS:
  void publishTwist();

private:
  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr velocity_publisher_;
  QTimer* output_timer_;

  float linear_velocity_;
  float angular_velocity_;
  bool keys_pressed_[4]; // W A S D
};

} // namespace mini_r1_v1_gz

#endif // MINI_R1_V1_GZ_TELEOP_PANEL_HPP
