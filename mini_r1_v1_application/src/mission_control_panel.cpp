#include "mini_r1_v1_application/mission_control_panel.hpp"
#include <rviz_common/display_context.hpp>
#include <QStringList>
#include <QHeaderView>
#include <QMessageBox>

namespace mini_r1_v1_application
{

MissionControlPanel::MissionControlPanel(QWidget * parent)
: rviz_common::Panel(parent)
{
  QVBoxLayout * layout = new QVBoxLayout;

  QLabel * label = new QLabel("Detected Mission Objects (ArUco / Signs)");
  layout->addWidget(label);

  table_ = new QTableWidget(0, 5, this);
  QStringList headers;
  headers << "Type" << "ID" << "X (m)" << "Y (m)" << "Z (m)";
  table_->setHorizontalHeaderLabels(headers);
  table_->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
  layout->addWidget(table_);

  // Mission status banner (hidden until mission complete)
  status_label_ = new QLabel(this);
  status_label_->setAlignment(Qt::AlignCenter);
  status_label_->setStyleSheet(
    "QLabel { background-color: #00bcd4; color: white; font-size: 16px; "
    "font-weight: bold; padding: 8px; border-radius: 4px; }");
  status_label_->setText("MISSION COMPLETE!");
  status_label_->setVisible(false);
  layout->addWidget(status_label_);

  setLayout(layout);

  connect(this, &MissionControlPanel::newMarkerDetected,
          this, &MissionControlPanel::addRowToTable, Qt::QueuedConnection);
  connect(this, &MissionControlPanel::missionCompleteReceived,
          this, &MissionControlPanel::showMissionComplete, Qt::QueuedConnection);
}

MissionControlPanel::~MissionControlPanel()
{
}

void MissionControlPanel::onInitialize()
{
  node_ = this->getDisplayContext()->getRosNodeAbstraction().lock()->get_raw_node();

  sub_markers_ = node_->create_subscription<visualization_msgs::msg::MarkerArray>(
    "/mini_r1/mission_control/detected_objects", 10,
    std::bind(&MissionControlPanel::markerCallback, this, std::placeholders::_1));

  sub_mission_status_ = node_->create_subscription<std_msgs::msg::String>(
    "/mini_r1/mission_control/mission_status", 10,
    std::bind(&MissionControlPanel::missionStatusCallback, this, std::placeholders::_1));
}

void MissionControlPanel::markerCallback(const visualization_msgs::msg::MarkerArray::SharedPtr msg)
{
  // Collect sign directions from sign_text markers
  std::map<int, std::string> sign_directions;
  for (const auto & marker : msg->markers) {
    if (marker.ns == "sign_text" && !marker.text.empty()) {
      // text is "Sign: forward" etc.
      auto pos = marker.text.find(": ");
      if (pos != std::string::npos) {
        sign_directions[marker.id] = marker.text.substr(pos + 2);
      }
    }
  }

  for (const auto & marker : msg->markers) {
    if (marker.action == visualization_msgs::msg::Marker::ADD) {
      if (marker.ns == "aruco") {
        Q_EMIT newMarkerDetected(
          QString("aruco"), marker.id,
          marker.pose.position.x,
          marker.pose.position.y,
          marker.pose.position.z);
      } else if (marker.ns == "sign") {
        // Include direction in the type column
        QString type_str = "sign";
        auto it = sign_directions.find(marker.id);
        if (it != sign_directions.end()) {
          type_str = QString("sign (%1)").arg(QString::fromStdString(it->second));
        }
        Q_EMIT newMarkerDetected(
          type_str, marker.id,
          marker.pose.position.x,
          marker.pose.position.y,
          marker.pose.position.z);
      }
    }
  }
}

void MissionControlPanel::missionStatusCallback(const std_msgs::msg::String::SharedPtr msg)
{
  if (!mission_complete_ && msg->data == "MISSION_COMPLETE") {
    mission_complete_ = true;
    Q_EMIT missionCompleteReceived();
  }
}

void MissionControlPanel::showMissionComplete()
{
  status_label_->setVisible(true);

  QMessageBox::information(
    this, "Mission Complete",
    "The robot has reached the goal zone!\n\nMission accomplished.");
}

void MissionControlPanel::addRowToTable(const QString& type, int id, double x, double y, double z)
{
  QString id_str = QString::number(id);

  for (int i = 0; i < table_->rowCount(); ++i) {
    if (table_->item(i, 0)->text() == type && table_->item(i, 1)->text() == id_str) {
      table_->item(i, 2)->setText(QString::number(x, 'f', 2));
      table_->item(i, 3)->setText(QString::number(y, 'f', 2));
      table_->item(i, 4)->setText(QString::number(z, 'f', 2));
      return;
    }
  }

  int row = table_->rowCount();
  table_->insertRow(row);

  table_->setItem(row, 0, new QTableWidgetItem(type));
  table_->setItem(row, 1, new QTableWidgetItem(id_str));
  table_->setItem(row, 2, new QTableWidgetItem(QString::number(x, 'f', 2)));
  table_->setItem(row, 3, new QTableWidgetItem(QString::number(y, 'f', 2)));
  table_->setItem(row, 4, new QTableWidgetItem(QString::number(z, 'f', 2)));
}

}  // namespace mini_r1_v1_application

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(mini_r1_v1_application::MissionControlPanel, rviz_common::Panel)
