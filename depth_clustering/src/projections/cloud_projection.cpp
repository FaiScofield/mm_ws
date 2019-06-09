// Copyright (C) 2017  I. Bogoslavskyi, C. Stachniss, University of Bonn

// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.

// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
// more details.

// You should have received a copy of the GNU General Public License along
// with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "projections/cloud_projection.h"
#include <string>
#include <vector>

#include "utils/mem_utils.h"

namespace depth_clustering {

using mem_utils::make_unique;

CloudProjection::PointContainer::PointContainer() {}

CloudProjection::CloudProjection(const ProjectionParams& params)
    : _params(params) {
  if (!_params.valid()) {
    throw std::runtime_error("_params not valid for projection.");
  }
  _data = PointMatrix(_params.cols(), PointColumn(_params.rows()));
  _depth_image =
      cv::Mat::zeros(_params.rows(), _params.cols(), cv::DataType<float>::type);
}

// 从深度图反投影回三维空间，获取三维坐标点
RichPoint CloudProjection::UnprojectPoint(const cv::Mat& image, const int row,
                                          const int col) const {
  float depth = image.at<float>(row, col);
  Radians angle_z = this->_params.AngleFromRow(row);
  Radians angle_xy = this->_params.AngleFromCol(col);
  RichPoint point{depth * cosf(angle_z.val()) * cosf(angle_xy.val()),   // x
                  depth * cosf(angle_z.val()) * sinf(angle_xy.val()),   // y
                  depth * sinf(angle_z.val())};                         // z
  return point;
}

void CloudProjection::CheckCloudAndStorage(const RichPoint::AlignedVector& points) {
  if (this->_data.size() < 1) {
    throw std::length_error("_data size is < 1");
  }
  if (points.empty()) {
    throw std::runtime_error("cannot fill from cloud: no points");
  }
}

void CloudProjection::CheckImageAndStorage(const cv::Mat& image) {
  if (image.type() != CV_32F) {
    throw std::runtime_error("wrong image format");
  }
  if (this->_data.size() < 1) {
    throw std::length_error("_data size is < 1");
  }
  if (this->rows() != static_cast<size_t>(image.rows) ||
      this->cols() != static_cast<size_t>(image.cols)) {
    throw std::length_error("_data dimentions do not correspond to image ones");
  }
}

void CloudProjection::FixDepthSystematicErrorIfNeeded() {
  if (_depth_image.rows < 1) {
    fprintf(stderr, "[INFO] image of wrong size, not correcting depth\n");
    return;
  }
  if (_corrections.size() != static_cast<size_t>(_depth_image.rows)) {
    fprintf(stderr, "[INFO] Not correcting depth data. for _corrections.size=%ld, depth_image.rows=%ld\n",
            _corrections.size(), _depth_image.rows);
    return;
  }
  for (int r = 0; r < _depth_image.rows; ++r) {
    auto correction = _corrections[r];
    for (int c = 0; c < _depth_image.cols; ++c) {
      if (_depth_image.at<float>(r, c) < 0.001f) {
        continue;
      }
      _depth_image.at<float>(r, c) -= correction;
    }
  }
}

const cv::Mat& CloudProjection::depth_image() const {
  return this->_depth_image;
}

cv::Mat& CloudProjection::depth_image() { return this->_depth_image; }

}  // namespace depth_clustering
