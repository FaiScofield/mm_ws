// Copyright Igor Bogoslavskyi, year 2017.
// In case of any problems with the code please contact me.
// Email: igor.bogoslavskyi@uni-bonn.de.

#include <opencv/highgui.h>
#include "./opengl_folder_player.h"

#include <QColor>
#include <QDebug>
#include <QFileDialog>
#include <QImage>
#include <QPixmap>
#include <QUuid>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>

#include <utils/folder_reader.h>
#include <utils/timer.h>
#include <utils/velodyne_utils.h>
#include <visualization/visualizer.h>

#include <vector>

#include "qt/drawables/drawable_cloud.h"
#include "qt/drawables/drawable_cube.h"
#include "qt/utils/utils.h"

#include "qt/widgets/ui_opengl_folder_player.h"

#include "image_labelers/diff_helpers/diff_factory.h"

using std::vector;
using depth_clustering::Cloud;
using depth_clustering::ProjectionParams;
using depth_clustering::FolderReader;
using depth_clustering::ReadKittiCloudTxt;
using depth_clustering::ReadKittiCloud;
using depth_clustering::MatFromDepthPng;
using depth_clustering::Radians;
using depth_clustering::DepthGroundRemover;
using depth_clustering::ImageBasedClusterer;
using depth_clustering::LinearImageLabeler;
using depth_clustering::AbstractImageLabeler;
using depth_clustering::AngleDiffPrecomputed;
using depth_clustering::DiffFactory;
using depth_clustering::time_utils::Timer;



OpenGlFolderPlayer::OpenGlFolderPlayer(QWidget *parent)
    : BaseViewerWidget(parent), ui(new Ui::OpenGlFolderPlayer) {
  ui->setupUi(this);
  ui->sldr_navigate_clouds->setEnabled(false);
  ui->spnbx_current_cloud->setEnabled(false);

  _viewer = ui->gl_widget;
  _viewer->installEventFilter(this);
  _viewer->setAutoFillBackground(true);

  connect(ui->btn_open_folder, SIGNAL(released()), this,
          SLOT(onOpenFolderToRead()));
  connect(ui->sldr_navigate_clouds, SIGNAL(valueChanged(int)), this,
          SLOT(onSliderMovedTo(int)));
  connect(ui->btn_play, SIGNAL(released()), this, SLOT(onPlayAllClouds()));

  connect(ui->spnbx_min_cluster_size, SIGNAL(valueChanged(int)), this,
          SLOT(onSegmentationParamUpdate()));
  connect(ui->spnbx_max_cluster_size, SIGNAL(valueChanged(int)), this,
          SLOT(onSegmentationParamUpdate()));
  connect(ui->spnbx_ground_angle, SIGNAL(valueChanged(double)), this,
          SLOT(onSegmentationParamUpdate()));
  connect(ui->spnbx_separation_angle, SIGNAL(valueChanged(double)), this,
          SLOT(onSegmentationParamUpdate()));
  connect(ui->spnbx_smooth_window_size, SIGNAL(valueChanged(int)), this,
          SLOT(onSegmentationParamUpdate()));
  connect(ui->radio_show_segmentation, SIGNAL(toggled(bool)), this,
          SLOT(onSegmentationParamUpdate()));
  connect(ui->radio_show_angles, SIGNAL(toggled(bool)), this,
          SLOT(onSegmentationParamUpdate()));
  connect(ui->cmb_diff_type, SIGNAL(activated(int)), this,
          SLOT(onSegmentationParamUpdate()));
  connect(ui->radio_show_segmentation, SIGNAL(toggled(bool)), this,
          SLOT(onSegmentationParamUpdate()));

  // setup viewer
  _cloud.reset(new Cloud);

  _proj_params = ProjectionParams::HDL_64();

  ui->gfx_projection_view->setViewportUpdateMode(
      QGraphicsView::BoundingRectViewportUpdate);
  ui->gfx_projection_view->setCacheMode(QGraphicsView::CacheBackground);
  ui->gfx_projection_view->setRenderHints(QPainter::Antialiasing |
                                          QPainter::SmoothPixmapTransform);

  ui->gfx_labels->setViewportUpdateMode(
      QGraphicsView::BoundingRectViewportUpdate);
  ui->gfx_labels->setCacheMode(QGraphicsView::CacheBackground);
  ui->gfx_labels->setRenderHints(QPainter::Antialiasing |
                                 QPainter::SmoothPixmapTransform);

  _painter.reset(new ObjectPainter(_viewer));
  this->onSegmentationParamUpdate();
}

void OpenGlFolderPlayer::onPlayAllClouds() {
  for (int i = ui->sldr_navigate_clouds->minimum();
       i < ui->sldr_navigate_clouds->maximum(); ++i) {
    ui->sldr_navigate_clouds->setValue(i);
    ui->gl_widget->update();
    QApplication::processEvents();
  }
  qDebug() << "All clouds shown!";
}

void OpenGlFolderPlayer::OnNewObjectReceived(const cv::Mat &image,
                                             int client_id) {
  QImage qimage;
  fprintf(stderr, "[INFO] Received Mat with type: %d\n", image.type());
  switch (image.type()) {
    case cv::DataType<float>::type: {
      // we have received a depth image
      fprintf(stderr, "[INFO] received depth.\n");
      DiffFactory::DiffType diff_type = DiffFactory::DiffType::NONE;
      switch (ui->cmb_diff_type->currentIndex()) {
        case 0: {
          fprintf(stderr, "Using DiffFactory::DiffType::ANGLES\n");
          diff_type = DiffFactory::DiffType::ANGLES;
          break;
        }
        case 1: {
          fprintf(stderr, "Using DiffFactory::DiffType::ANGLES_PRECOMPUTED\n");
          diff_type = DiffFactory::DiffType::ANGLES_PRECOMPUTED;
          break;
        }
        case 2: {
          fprintf(stderr, "Using DiffFactory::DiffType::LINE_DIST\n");
          diff_type = DiffFactory::DiffType::LINE_DIST;
          break;
        }
        case 3: {
          fprintf(stderr,
                  "Using DiffFactory::DiffType::LINE_DIST_PRECOMPUTED\n");
          diff_type = DiffFactory::DiffType::LINE_DIST_PRECOMPUTED;
          break;
        }
        default: {
          fprintf(stderr, "Using DiffFactory::DiffType::SIMPLE\n");
          diff_type = DiffFactory::DiffType::SIMPLE;
        }
      }
      auto diff_helper_ptr =
          DiffFactory::Build(diff_type, &image, _proj_params.get());
      qimage = MatToQImage(diff_helper_ptr->Visualize());
//      _label_image = diff_helper_ptr->Visualize().clone();
//      qimage = MatToQImage(_label_image);
      break;
    }
    case cv::DataType<uint16_t>::type: {
      // we have received labels
      fprintf(stderr, "[INFO] received labels.\n");
//      _label_image = AbstractImageLabeler::LabelsToColor(image).clone();
//      qimage = MatToQImage(_label_image);
      qimage = MatToQImage(AbstractImageLabeler::LabelsToColor(image));
      break;
    }
    default: {
      fprintf(stderr, "ERROR: unknown type Mat received.\n");
      return;
    }
  }
  _scene_labels.reset(new QGraphicsScene);
  _scene_labels->addPixmap(QPixmap::fromImage(qimage));
  ui->gfx_labels->setScene(_scene_labels.get());
  ui->gfx_labels->fitInView(_scene_labels->itemsBoundingRect());
}

void OpenGlFolderPlayer::onSegmentationParamUpdate() {
  // setup segmentation
  fprintf(stderr, "Info: update segmentation parameters\n");
  int smooth_window_size = ui->spnbx_smooth_window_size->value();
  Radians ground_remove_angle =
      Radians::FromDegrees(ui->spnbx_ground_angle->value());
  Radians angle_tollerance =
      Radians::FromDegrees(ui->spnbx_separation_angle->value());
  int min_cluster_size = ui->spnbx_min_cluster_size->value();
  int max_cluster_size = ui->spnbx_max_cluster_size->value();

  // create objects
  DiffFactory::DiffType diff_type = DiffFactory::DiffType::NONE;
  switch (ui->cmb_diff_type->currentIndex()) {
    case 0: {
      fprintf(stderr, "Using DiffFactory::DiffType::ANGLES\n");
      diff_type = DiffFactory::DiffType::ANGLES;
      break;
    }
    case 1: {
      fprintf(stderr, "Using DiffFactory::DiffType::ANGLES_PRECOMPUTED\n");
      diff_type = DiffFactory::DiffType::ANGLES_PRECOMPUTED;
      break;
    }
    case 2: {
      fprintf(stderr, "Using DiffFactory::DiffType::LINE_DIST\n");
      diff_type = DiffFactory::DiffType::LINE_DIST;
      break;
    }
    case 3: {
      fprintf(stderr, "Using DiffFactory::DiffType::LINE_DIST_PRECOMPUTED\n");
      diff_type = DiffFactory::DiffType::LINE_DIST_PRECOMPUTED;
      break;
    }
    default: {
      fprintf(stderr, "Using DiffFactory::DiffType::SIMPLE\n");
      diff_type = DiffFactory::DiffType::SIMPLE;
    }
  }
  _clusterer.reset(new ImageBasedClusterer<LinearImageLabeler<>>(
      angle_tollerance, min_cluster_size, max_cluster_size));
  ui->spnbx_line_dist_threshold->setValue(angle_tollerance.val());
  _clusterer->SetDiffType(diff_type);
  _ground_rem.reset(new DepthGroundRemover(*_proj_params, ground_remove_angle,
                                           smooth_window_size));
  // configure wires
  _ground_rem->AddClient(_clusterer.get());
  _clusterer->AddClient(_painter.get());
  if (ui->radio_show_segmentation->isChecked()) {
    fprintf(stderr, "Info: ready to receive labels\n");
    _clusterer->SetLabelImageClient(this);
  } else {
    _scene_labels.reset();
  }
/*
  if (ui->radio_show_projection->isChecked()) {
      this->depthImageSmooth();
//        this->OnNewObjectReceived(_current_smooth_depth_image);
//      const cv::Mat* labels_ptr = _clusterer->ImageBasedClusterer.GetLabelImage();
      depth_clustering::LinearImageLabeler<> image_labeler(_current_smooth_depth_image,
            _cloud->projection_ptr()->params(), angle_tollerance);
      image_labeler.ComputeLabels(diff_type);
      const cv::Mat* labels_ptr = image_labeler.GetLabelImage();
      _label_image = (*labels_ptr).clone();
      cv::imshow("label image", _label_image);
    }
*/
  this->onSliderMovedTo(ui->sldr_navigate_clouds->value());
}

void OpenGlFolderPlayer::onSliderMovedTo(int cloud_number) {
  if (_file_names.empty()) {
    return;
  }
  fprintf(stderr, "slider moved to: %d\n", cloud_number);
  fprintf(stderr, "loading cloud from: %s\n",
          _file_names[cloud_number].c_str());
  Timer timer;
  const auto &file_name = _file_names[cloud_number];
  _cloud = CloudFromFile(file_name, *_proj_params);
  fprintf(stderr, "[TIMER]: load cloud in %lu microsecs\n",
          timer.measure(Timer::Units::Micro));
  _current_full_depth_image = _cloud->projection_ptr()->depth_image();

  // add by @Vance
//  if (ui->checkBox_saveDepthImage->checkState() == Qt::CheckState::Checked) {
//    size_t index = _file_names[cloud_number].find('.');
//    std::string outputfile = _file_names[cloud_number].substr(0, index);
//    outputfile += ".png";
//    cv::imwrite(outputfile, _current_full_depth_image);
//  }

  ui->lbl_cloud_name->setText(QString::fromStdString(file_name));

  timer.start();
  QImage qimage = MatToQImage(_current_full_depth_image);
  // add by @Vance
//  qimage.convertToFormat(Format(), Qt::ImageConversionFlags::AutoColor);

  _scene.reset(new QGraphicsScene);
  _scene->addPixmap(QPixmap::fromImage(qimage));
  ui->gfx_projection_view->setScene(_scene.get());
  ui->gfx_projection_view->fitInView(_scene->itemsBoundingRect());
  fprintf(stderr, "[TIMER]: depth image set to gui in %lu microsecs\n",
          timer.measure(Timer::Units::Micro));
  if (ui->radio_show_angles->isChecked()) {
    this->OnNewObjectReceived(_current_full_depth_image);
  }
  fprintf(stderr, "[TIMER]: angles shown in %lu microsecs\n",
          timer.measure(Timer::Units::Micro));

  _viewer->Clear();
  _viewer->AddDrawable(DrawableCloud::FromCloud(_cloud));
  _viewer->update();

  fprintf(stderr, "[TIMER]: add cloud to gui in %lu microsecs\n",
          timer.measure(Timer::Units::Micro));

  // label cloud and show labels
  _ground_rem->OnNewObjectReceived(*_cloud, 0);

  fprintf(stderr, "[TIMER]: full segmentation took %lu milliseconds\n",
          timer.measure(Timer::Units::Milli));
/*
  // get clusters
  if (ui->radio_show_projection->isChecked()) {
    _clusters.clear();
    for (int row = 0; row < _label_image.rows; ++row) {
    for (int col = 0; col < _label_image.cols; ++col) {
      const auto& point_container = _cloud->projection_ptr()->at(row, col);
      if (point_container.IsEmpty()) {
        // this is ok, just continue, nothing interesting here, no points.
        continue;
      }
      uint16_t label = _label_image.at<uint16_t>(row, col);
      if (label < 1) {
        continue; // this is a default label, skip
      }
      for (const auto& point_idx : point_container.points()) {
        const auto& point = _cloud->points()[point_idx];
        _clusters[label].push_back(point);
      }
    }
    }
    std::vector<uint16_t> labels_to_erase;
    for (const auto& kv : _clusters) {
      const auto& cluster = kv.second;
      if (cluster.size() < 30 || cluster.size() > 5000)
        labels_to_erase.push_back(kv.first);
    }
    for (auto label : labels_to_erase) {
        _clusters.erase(label);
    }

    // 投影到深度图里并给颜色
    if (_clusters.size() < 1) {
        fprintf(stderr, "[ERROR] No clusters this frame!\n");
        return;
    }

    cvtColor(_current_smooth_depth_image, _current_projection_image, cv::COLOR_GRAY2RGB);
    for (auto& c : _clusters) {
      auto& points = c.second.points();
      for (int i = 0; i < points.size(); ++i) {
        const auto& p = points[i];
        float dist_to_sensor = p.DistToSensor2D();
        if (dist_to_sensor < 0.01f) {
          continue;
        }
        auto angle_rows = dc::Radians::FromRadians(asin(p.z() / dist_to_sensor));
        auto angle_cols = dc::Radians::FromRadians(atan2(p.y(), p.x()));
//            size_t bin_rows = p.ring();
        size_t bin_rows = _proj_params->RowFromAngle(angle_rows);
        size_t bin_cols = _proj_params->ColFromAngle(angle_cols);

        // 聚类上红色
        _current_projection_image.at<cv::Vec3b>(bin_rows, bin_cols) = cv::Vec3b(0, 0, 255);
      }
    }
    cv::imshow("projection image", _current_projection_image);
  }
*/
}

void OpenGlFolderPlayer::onOpenFolderToRead() {
  // create a dialog here
  QString folder_name = QFileDialog::getExistingDirectory(this);
  qDebug() << "Picked path:" << folder_name;

  _file_names.clear();
  FolderReader::Order order = FolderReader::Order::SORTED;
  FolderReader cloud_reader(folder_name.toStdString(),
                            ui->cmb_extension->currentText().toStdString(),
                            order);
  _file_names = cloud_reader.GetAllFilePaths();
  if (_file_names.empty()) {
    return;
  }

  // update the slider
  ui->sldr_navigate_clouds->setMaximum(_file_names.size() - 1);
  ui->spnbx_current_cloud->setMaximum(_file_names.size() - 1);

  // set current value
  ui->sldr_navigate_clouds->setValue(1);
  ui->sldr_navigate_clouds->setEnabled(true);
  ui->spnbx_current_cloud->setEnabled(true);

  // focus on the cloud
  _viewer->update();
}

void OpenGlFolderPlayer::keyPressEvent(QKeyEvent *event) {
  switch (event->key()) {
    case Qt::Key_Right:
      ui->spnbx_current_cloud->setValue(ui->spnbx_current_cloud->value() + 1);
      break;
    case Qt::Key_Left:
      ui->spnbx_current_cloud->setValue(ui->spnbx_current_cloud->value() - 1);
      break;
  }
}

OpenGlFolderPlayer::~OpenGlFolderPlayer() {}



void OpenGlFolderPlayer::interpolation(cv::Mat& image, size_t row, size_t col, size_t cellSize)
{
    if (image.ptr(row)[col] > 0.01)
        return;

    size_t validNeighbourNum = 0;
    float value = 0;
    int s, e;
    if (cellSize % 2 == 0)
        s = -cellSize/2;
    else
        s = -(cellSize-1)/2;
    e = cellSize + s - 1;

    for (int i = s; i <= e; ++i) {
        for (int j = s; j <= e; ++j) {
            if (row + i >= 0 && row + i < image.rows &&
                col + j >= 0 && col + j < image.cols &&
                image.ptr(row+i)[col+j] > 0.01 ) {
                value += image.ptr(row+i)[col+j];
                validNeighbourNum++;
            } else
                continue;
        }
    }

    if (validNeighbourNum != 0)
        image.ptr(row)[col] = value/validNeighbourNum;
}

void OpenGlFolderPlayer::depthImageSmooth() {
    _current_smooth_depth_image = _current_full_depth_image.clone();
    for (int i = 0; i < _current_smooth_depth_image.rows; ++i) {
        for (int j = 0; j < _current_smooth_depth_image.cols; ++j) {
            if (_current_smooth_depth_image.ptr(i)[j] < 0.01) {
                interpolation(_current_smooth_depth_image, i, j, 9);
            }
        }
    }
}
