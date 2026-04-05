#include "mini_r1_v1_gz/teleop_panel.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <rviz_common/display_context.hpp>

namespace mini_r1_v1_gz
{

TeleopPanel::TeleopPanel(QWidget * parent)
  : rviz_common::Panel(parent),
    linear_velocity_(0.0),
    angular_velocity_(0.0)
{
  setFocusPolicy(Qt::StrongFocus);
  for (int i = 0; i < 4; ++i) keys_pressed_[i] = false;

  output_timer_ = new QTimer(this);
  connect(output_timer_, SIGNAL(timeout()), this, SLOT(publishTwist()));
  output_timer_->start(100); // 10 Hz
}

TeleopPanel::~TeleopPanel()
{
}

void TeleopPanel::onInitialize()
{
  node_ = this->getDisplayContext()->getRosNodeAbstraction().lock()->get_raw_node();
  velocity_publisher_ = node_->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 10);
}

void TeleopPanel::keyPressEvent(QKeyEvent * event)
{
  if (event->isAutoRepeat()) {
    event->ignore();
    return;
  }
  switch(event->key()) {
    case Qt::Key_W: keys_pressed_[0] = true; break;
    case Qt::Key_A: keys_pressed_[1] = true; break;
    case Qt::Key_S: keys_pressed_[2] = true; break;
    case Qt::Key_D: keys_pressed_[3] = true; break;
    default: rviz_common::Panel::keyPressEvent(event); return;
  }
  event->accept();
}

void TeleopPanel::keyReleaseEvent(QKeyEvent * event)
{
  if (event->isAutoRepeat()) {
    event->ignore();
    return;
  }
  switch(event->key()) {
    case Qt::Key_W: keys_pressed_[0] = false; break;
    case Qt::Key_A: keys_pressed_[1] = false; break;
    case Qt::Key_S: keys_pressed_[2] = false; break;
    case Qt::Key_D: keys_pressed_[3] = false; break;
    default: rviz_common::Panel::keyReleaseEvent(event); return;
  }
  event->accept();
}

void TeleopPanel::publishTwist()
{
  if (!velocity_publisher_) return;

  linear_velocity_ = 0.0;
  angular_velocity_ = 0.0;

  if (keys_pressed_[0]) linear_velocity_ += 0.5;
  if (keys_pressed_[2]) linear_velocity_ -= 0.5;
  if (keys_pressed_[1]) angular_velocity_ += 0.5;
  if (keys_pressed_[3]) angular_velocity_ -= 0.5;

  geometry_msgs::msg::Twist twist;
  twist.linear.x = linear_velocity_;
  twist.angular.z = angular_velocity_;
  twist.linear.y = 0.0;
  twist.linear.z = 0.0;
  twist.angular.x = 0.0;
  twist.angular.y = 0.0;
  velocity_publisher_->publish(twist);
}

} // namespace mini_r1_v1_gz

PLUGINLIB_EXPORT_CLASS(mini_r1_v1_gz::TeleopPanel, rviz_common::Panel)
