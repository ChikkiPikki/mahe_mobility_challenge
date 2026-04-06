#include "mini_r1_v1_application/navigator_status_panel.hpp"
#include <rviz_common/display_context.hpp>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QFont>

namespace mini_r1_v1_application
{

NavigatorStatusPanel::NavigatorStatusPanel(QWidget * parent)
: rviz_common::Panel(parent)
{
  QVBoxLayout * layout = new QVBoxLayout;
  layout->setSpacing(4);

  QFont bold_font;
  bold_font.setBold(true);
  bold_font.setPointSize(10);

  QFont mono_font("Monospace", 9);
  mono_font.setStyleHint(QFont::Monospace);

  state_label_ = new QLabel("State: IDLE");
  state_label_->setFont(bold_font);
  state_label_->setStyleSheet("color: #4CAF50; padding: 4px;");
  layout->addWidget(state_label_);

  behavior_label_ = new QLabel("Behavior: none");
  behavior_label_->setFont(mono_font);
  layout->addWidget(behavior_label_);

  // Separator
  QFrame * line1 = new QFrame();
  line1->setFrameShape(QFrame::HLine);
  layout->addWidget(line1);

  detectors_label_ = new QLabel("Detectors: ...");
  detectors_label_->setFont(mono_font);
  detectors_label_->setWordWrap(true);
  layout->addWidget(detectors_label_);

  QFrame * line2 = new QFrame();
  line2->setFrameShape(QFrame::HLine);
  layout->addWidget(line2);

  recovery_label_ = new QLabel("Recovery: none");
  recovery_label_->setFont(mono_font);
  layout->addWidget(recovery_label_);

  stats_label_ = new QLabel("Signs: 0 | ArUco: 0/4");
  stats_label_->setFont(mono_font);
  layout->addWidget(stats_label_);

  position_label_ = new QLabel("Pos: (0.00, 0.00) yaw=0");
  position_label_->setFont(mono_font);
  layout->addWidget(position_label_);

  QFrame * line3 = new QFrame();
  line3->setFrameShape(QFrame::HLine);
  layout->addWidget(line3);

  QLabel * plan_header = new QLabel("Reasoning:");
  plan_header->setFont(bold_font);
  layout->addWidget(plan_header);

  reasoning_text_ = new QTextEdit();
  reasoning_text_->setReadOnly(true);
  reasoning_text_->setMaximumHeight(60);
  reasoning_text_->setFont(mono_font);
  reasoning_text_->setStyleSheet("background-color: #1a1a2e; color: #e0e0e0; padding: 4px;");
  layout->addWidget(reasoning_text_);

  setLayout(layout);

  connect(this, &NavigatorStatusPanel::statusReceived,
          this, &NavigatorStatusPanel::updateDisplay, Qt::QueuedConnection);
}

NavigatorStatusPanel::~NavigatorStatusPanel() {}

void NavigatorStatusPanel::onInitialize()
{
  node_ = this->getDisplayContext()->getRosNodeAbstraction().lock()->get_raw_node();
  sub_status_ = node_->create_subscription<std_msgs::msg::String>(
    "/mini_r1/navigator/status", 10,
    std::bind(&NavigatorStatusPanel::statusCallback, this, std::placeholders::_1));
}

void NavigatorStatusPanel::statusCallback(const std_msgs::msg::String::SharedPtr msg)
{
  Q_EMIT statusReceived(QString::fromStdString(msg->data));
}

void NavigatorStatusPanel::updateDisplay(const QString & json_str)
{
  QJsonDocument doc = QJsonDocument::fromJson(json_str.toUtf8());
  if (!doc.isObject()) return;
  QJsonObject obj = doc.object();

  QString state = obj["state"].toString("?");
  state_label_->setText("State: " + state);

  // Color code by state
  if (state == "STOPPED") {
    state_label_->setStyleSheet("color: #2196F3; font-weight: bold; padding: 4px;");
  } else if (state == "RECOVERING") {
    state_label_->setStyleSheet("color: #FF5722; font-weight: bold; padding: 4px;");
  } else if (state == "EXECUTING_SIGN") {
    state_label_->setStyleSheet("color: #FF9800; font-weight: bold; padding: 4px;");
  } else {
    state_label_->setStyleSheet("color: #4CAF50; font-weight: bold; padding: 4px;");
  }

  QString beh = obj["behavior"].toString("none");
  double elapsed = obj["behavior_elapsed_s"].toDouble(0);
  behavior_label_->setText(
    QString("Behavior: %1 (%2s)").arg(beh).arg(elapsed, 0, 'f', 1));

  // Detectors
  QJsonObject dets = obj["detectors"].toObject();
  QStringList det_lines;
  for (auto it = dets.begin(); it != dets.end(); ++it) {
    QString name = it.key();
    QString status;
    if (it.value().isObject()) {
      QJsonObject dobj = it.value().toObject();
      if (dobj["active"].toBool()) {
        status = QString("\"%1\" (%2s ago)")
          .arg(dobj["value"].toString())
          .arg(dobj["age_s"].toDouble(), 0, 'f', 1);
      } else {
        status = "clear";
      }
    } else {
      status = it.value().toBool() ? "TRIGGERED" : "clear";
    }
    QString color = (status != "clear") ? "#FF5722" : "#888";
    det_lines << QString("<span style='color:%1'>%2: %3</span>")
      .arg(color, name, status);
  }
  detectors_label_->setText(det_lines.join("<br>"));
  detectors_label_->setTextFormat(Qt::RichText);

  // Recovery
  QJsonValue rec = obj["recovery"];
  if (rec.isNull() || rec.toString().isEmpty()) {
    recovery_label_->setText("Recovery: none");
  } else {
    recovery_label_->setText("Recovery: " + rec.toString());
    recovery_label_->setStyleSheet("color: #FF5722;");
  }

  // Stats
  int signs = obj["signs_seen"].toInt(0);
  QJsonArray arucos = obj["arucos_seen"].toArray();
  stats_label_->setText(
    QString("Signs: %1 | ArUco: %2/4").arg(signs).arg(arucos.size()));

  // Position
  QJsonObject pos = obj["position"].toObject();
  position_label_->setText(
    QString("Pos: (%1, %2) yaw=%3")
      .arg(pos["x"].toDouble(), 0, 'f', 2)
      .arg(pos["y"].toDouble(), 0, 'f', 2)
      .arg(pos["yaw_deg"].toDouble(), 0, 'f', 0));

  // Reasoning
  reasoning_text_->setText(obj["reasoning"].toString("..."));
}

}  // namespace mini_r1_v1_application

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(mini_r1_v1_application::NavigatorStatusPanel, rviz_common::Panel)
