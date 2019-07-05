/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/bin.h>

#include <LightGBM/utils/common.h>
#include <LightGBM/utils/file_io.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "dense_bin.hpp"
#include "dense_nbits_bin.hpp"
#include "ordered_sparse_bin.hpp"
#include "sparse_bin.hpp"

namespace LightGBM {

  BinMapper::BinMapper() {
  }

  // deep copy function for BinMapper
  BinMapper::BinMapper(const BinMapper& other) {
    num_bin_ = other.num_bin_;
    missing_type_ = other.missing_type_;
    is_trivial_ = other.is_trivial_;
    sparse_rate_ = other.sparse_rate_;
    bin_type_ = other.bin_type_;
    if (bin_type_ == BinType::NumericalBin) {
      bin_upper_bound_ = other.bin_upper_bound_;
    } else {
      bin_2_categorical_ = other.bin_2_categorical_;
      categorical_2_bin_ = other.categorical_2_bin_;
    }
    min_val_ = other.min_val_;
    max_val_ = other.max_val_;
    default_bin_ = other.default_bin_;
  }

  BinMapper::BinMapper(const void* memory) {
    CopyFrom(reinterpret_cast<const char*>(memory));
  }

  BinMapper::~BinMapper() {
  }

  bool NeedFilter(const std::vector<int>& cnt_in_bin, int total_cnt, int filter_cnt, BinType bin_type) {
    if (bin_type == BinType::NumericalBin) {
      int sum_left = 0;
      for (size_t i = 0; i < cnt_in_bin.size() - 1; ++i) {
        sum_left += cnt_in_bin[i];
        if (sum_left >= filter_cnt && total_cnt - sum_left >= filter_cnt) {
          return false;
        }
      }
    } else {
      if (cnt_in_bin.size() <= 2) {
        for (size_t i = 0; i < cnt_in_bin.size() - 1; ++i) {
          int sum_left = cnt_in_bin[i];
          if (sum_left >= filter_cnt && total_cnt - sum_left >= filter_cnt) {
            return false;
          }
        }
      } else {
        return false;
      }
    }
    return true;
  }

  std::vector<double> GreedyFindBin(const double* distinct_values, const int* counts,
    int num_distinct_values, int max_bin, size_t total_cnt, int min_data_in_bin) {
    //distinct_values为特征的不同的取值组成的数组(从小到大排序好的)，counts为特征不同取值个数组成的数组
    //num_distinct_values为特征有多少个不同的取值，max_bin表示直方图分桶的数量，total_cnt表示所有取值的总数，min_data_in_bin表示每个桶中最少个数
    std::vector<double> bin_upper_bound; // bin_upper_bound就是记录桶分界的数组
    CHECK(max_bin > 0);
    if (num_distinct_values <= max_bin) { // 特征取值数比max_bin数量少，直接将特征两两取值的中点作为桶的分界
      bin_upper_bound.clear();
      int cur_cnt_inbin = 0;
      for (int i = 0; i < num_distinct_values - 1; ++i) {
        cur_cnt_inbin += counts[i];
        if (cur_cnt_inbin >= min_data_in_bin) { //当数量满足min_data_in_bin的要求时，求桶的边界，存入bin_upper_bound中
          auto val = Common::GetDoubleUpperBound((distinct_values[i] + distinct_values[i + 1]) / 2.0);
          if (bin_upper_bound.empty() || !Common::CheckDoubleEqualOrdered(bin_upper_bound.back(), val)) {//CheckDoubleEqualOrdered用于判断val是否小于等于上届数组中的最后一个值
            bin_upper_bound.push_back(val);
            cur_cnt_inbin = 0;
          }
        }
      }
      cur_cnt_inbin += counts[num_distinct_values - 1];
      bin_upper_bound.push_back(std::numeric_limits<double>::infinity()); //最后一个桶的上边界为正无穷
    } else {// 特征取值数比max_bin数量多，需要将一些特征值放入同一个桶中进行合并
      if (min_data_in_bin > 0) {
        max_bin = std::min(max_bin, static_cast<int>(total_cnt / min_data_in_bin)); //求出max_bin
        max_bin = std::max(max_bin, 1);
      }
      double mean_bin_size = static_cast<double>(total_cnt) / max_bin;  //求出每个桶平均容量

      // mean size for one bin
      int rest_bin_cnt = max_bin;
      int rest_sample_cnt = static_cast<int>(total_cnt);
      std::vector<bool> is_big_count_value(num_distinct_values, false); //is_big_count_value用来存储每个取值的个数是否超过mean_bin_size
      for (int i = 0; i < num_distinct_values; ++i) {
        if (counts[i] >= mean_bin_size) {
          is_big_count_value[i] = true;
          --rest_bin_cnt;
          rest_sample_cnt -= counts[i];
        }
      }
      mean_bin_size = static_cast<double>(rest_sample_cnt) / rest_bin_cnt;  //去掉个数超过桶平均容量的取值重新计算mean_bin_size
      std::vector<double> upper_bounds(max_bin, std::numeric_limits<double>::infinity()); //桶的上界数组
      std::vector<double> lower_bounds(max_bin, std::numeric_limits<double>::infinity()); //桶的下界数组

      int bin_cnt = 0;
      lower_bounds[bin_cnt] = distinct_values[0]; //第一个桶下界肯定是第一个取值
      int cur_cnt_inbin = 0;
      for (int i = 0; i < num_distinct_values - 1; ++i) { //遍历每一个取值
        if (!is_big_count_value[i]) { //如果是数量少的取值
          rest_sample_cnt -= counts[i];
        }
        cur_cnt_inbin += counts[i];
        // need a new bin
        // 当满足下面三个条件中的一个即可成一个桶，求出桶的上界和下界
        // 1. 目前取值是数量多的取值
        // 2. 当前累计的取值个数已经满足mean_bin_size
        // 3. 下一个取值是数量多的取值且当前取值已经超过mean_bin_size的一半
        if (is_big_count_value[i] || cur_cnt_inbin >= mean_bin_size ||
          (is_big_count_value[i + 1] && cur_cnt_inbin >= std::max(1.0, mean_bin_size * 0.5f))) {
          upper_bounds[bin_cnt] = distinct_values[i];
          ++bin_cnt;
          lower_bounds[bin_cnt] = distinct_values[i + 1];
          if (bin_cnt >= max_bin - 1) { break; }  //当桶的数量到达极限时，退出循环，把剩下的取值全部放在最后一个桶中
          cur_cnt_inbin = 0;
          if (!is_big_count_value[i]) { //当前面的成桶的判断满足的是2和3条件时，需要重新计算mean_bin_size
            --rest_bin_cnt;
            mean_bin_size = rest_sample_cnt / static_cast<double>(rest_bin_cnt);
          }
        }
      }
      ++bin_cnt;
      // update bin upper bound
      // 现在取值的个数和桶的个数相当，和前一种情况类似
      bin_upper_bound.clear();
      for (int i = 0; i < bin_cnt - 1; ++i) {
        auto val = Common::GetDoubleUpperBound((upper_bounds[i] + lower_bounds[i + 1]) / 2.0);  //将中点作为桶分界依据
        if (bin_upper_bound.empty() || !Common::CheckDoubleEqualOrdered(bin_upper_bound.back(), val)) {
          bin_upper_bound.push_back(val);
        }
      }
      // last bin upper bound
      bin_upper_bound.push_back(std::numeric_limits<double>::infinity());
    }
    return bin_upper_bound;
  }

  std::vector<double> FindBinWithZeroAsOneBin(const double* distinct_values, const int* counts,
    int num_distinct_values, int max_bin, size_t total_sample_cnt, int min_data_in_bin) {
    std::vector<double> bin_upper_bound;
    int left_cnt_data = 0;  // left_cnt_data记录小于0的值
    int cnt_zero = 0;
    int right_cnt_data = 0; // right_cnt_data记录大于0的值
    for (int i = 0; i < num_distinct_values; ++i) {
      // double kZeroThreshold = 1e-35f
      if (distinct_values[i] <= -kZeroThreshold) {  //kZeroThreshold是比0稍大的一个值，很接近0 
        left_cnt_data += counts[i];
      } else if (distinct_values[i] > kZeroThreshold) {
        right_cnt_data += counts[i];
      } else {
        cnt_zero += counts[i];
      }
    }
 
    int left_cnt = -1; //如果特征值里存在0和正数，则left_cnt不为-1，则left_cnt是最后一个负数的位置
    for (int i = 0; i < num_distinct_values; ++i) {
      if (distinct_values[i] > -kZeroThreshold) {
        left_cnt = i;
        break;
      }
    }
    // 如果特征值全是负值，就把取值的总数赋给left_cnt
    if (left_cnt < 0) {
      left_cnt = num_distinct_values;
    }

    if (left_cnt > 0) {
      // 负数占正数和负数之和的比例乘上(max_bin-1)，即得到负数的桶数。之所以是乘(max_bin-1)不是乘max_bin是因为要给0留一个桶
      int left_max_bin = static_cast<int>(static_cast<double>(left_cnt_data) / (total_sample_cnt - cnt_zero) * (max_bin - 1));
      left_max_bin = std::max(1, left_max_bin);
      bin_upper_bound = GreedyFindBin(distinct_values, counts, left_cnt, left_max_bin, left_cnt_data, min_data_in_bin); //调用GreedyFindBin找到每个桶的上界
      bin_upper_bound.back() = -kZeroThreshold; //负数最后一个上界是-kZeroThreshold
    }

    int right_start = -1; //如果特征值存在正数，则right_start不为-1，则right_start是第一个正数开始的位置
    for (int i = left_cnt; i < num_distinct_values; ++i) {
      if (distinct_values[i] > kZeroThreshold) {
        right_start = i;
        break;
      }
    }
    // 如果特征值里存在正数
    if (right_start >= 0) {
      int right_max_bin = max_bin - 1 - static_cast<int>(bin_upper_bound.size()); //正数的桶个数就是(max_bin-1)减去负数占的桶个数
      CHECK(right_max_bin > 0);
      auto right_bounds = GreedyFindBin(distinct_values + right_start, counts + right_start, //调用GreedyFindBin找到每个桶的上界
        num_distinct_values - right_start, right_max_bin, right_cnt_data, min_data_in_bin);
      bin_upper_bound.push_back(kZeroThreshold);  //正数第一个上界是kZeroThreshold
      bin_upper_bound.insert(bin_upper_bound.end(), right_bounds.begin(), right_bounds.end());
    } else {
      bin_upper_bound.push_back(std::numeric_limits<double>::infinity());
    }
    return bin_upper_bound;
  }

  void BinMapper::FindBin(double* values, int num_sample_values, size_t total_sample_cnt,
    int max_bin, int min_data_in_bin, int min_split_data, BinType bin_type, bool use_missing, bool zero_as_missing) {
      // values表示一个特征的所有取值（未去重未排序）
      // num_sample_values表示values的个数，不包含0对应的样本
      // total_sample_cnt表示取样的总数，包含0对应的样本
      // total_sample_cnt表示取样的个数
      // min_split_data表示
      // bin_type有两种，一种是NumericalBin，表示数值类型数据，一种是CategoricalBin，表示类别型数据
      // use_missing为真时表示需要对缺失值进行处理
      // zero_as_missing为真时表示把0作为缺失值表示，为假时用na表示缺失值
    int na_cnt = 0;
    int tmp_num_sample_values = 0;  //表示去除缺失值之后的样本数量
    for (int i = 0; i < num_sample_values; ++i) { //将缺失值na去掉
      if (!std::isnan(values[i])) {
        values[tmp_num_sample_values++] = values[i];
      }
    }
    if (!use_missing) { //当不对缺失值进行处理时
      missing_type_ = MissingType::None;
    } else if (zero_as_missing) { //当对缺失值进行处理，且把0作为缺失值
      missing_type_ = MissingType::Zero;  
    } else {  //当对缺失值进行处理，用na表示缺失值
      if (tmp_num_sample_values == num_sample_values) { //当数据中不存在na时
        missing_type_ = MissingType::None;
      } else {
        missing_type_ = MissingType::NaN;//当数据中存在na时，求出na的个数
        na_cnt = num_sample_values - tmp_num_sample_values;
      }
    }
    num_sample_values = tmp_num_sample_values; //将缺失值去掉后的取值个数

    bin_type_ = bin_type;
    default_bin_ = 0;
    int zero_cnt = static_cast<int>(total_sample_cnt - num_sample_values - na_cnt); //计算0的个数，为什么这么算？
    // find distinct_values first
    std::vector<double> distinct_values;
    std::vector<int> counts;

    std::stable_sort(values, values + num_sample_values); //对values进行排序，保证相等元素的原本相对次序在排序后保持不变

    // push zero in the front
    if (num_sample_values == 0 || (values[0] > 0.0f && zero_cnt > 0)) { //如果取值都是正数或没有0以外的取值，则将0放在distinct_values第一个
      distinct_values.push_back(0.0f);
      counts.push_back(zero_cnt);
    }

    if (num_sample_values > 0) {  //如果存在0以外的取值，则先将第一个取值放在首位
      distinct_values.push_back(values[0]);
      counts.push_back(1);
    }

    for (int i = 1; i < num_sample_values; ++i) { //对values进行合并，统计出现次数
      if (!Common::CheckDoubleEqualOrdered(values[i - 1], values[i])) { // 如果values[i - 1]小于values[i]
        if (values[i - 1] < 0.0f && values[i] > 0.0f) {// 当出现一负一正时需要将0插入在中间，因为values已经按照顺序排列好
          distinct_values.push_back(0.0f);
          counts.push_back(zero_cnt);
        }
        distinct_values.push_back(values[i]);// 把新出现的value插入
        counts.push_back(1);
      } else { // 如果values[i - 1]不小于values[i]，即只可能values[i - 1]等于values[i]，说明distinct_values已经有了这个值了，只需要把它的counts加1.
        // use the large value
        distinct_values.back() = values[i];
        ++counts.back();
      }
    }

    // push zero in the back
    if (num_sample_values > 0 && values[num_sample_values - 1] < 0.0f && zero_cnt > 0) {  //如果取值都是负数，则把0放在最后
      distinct_values.push_back(0.0f);
      counts.push_back(zero_cnt);
    }
    min_val_ = distinct_values.front(); //distinct_values的最小值
    max_val_ = distinct_values.back(); //distinct_values的最大值
    std::vector<int> cnt_in_bin;
    int num_distinct_values = static_cast<int>(distinct_values.size()); //distinct_values的数量
    if (bin_type_ == BinType::NumericalBin) { //如果是数值型特征
      if (missing_type_ == MissingType::Zero) { //把0作为缺失值
        bin_upper_bound_ = FindBinWithZeroAsOneBin(distinct_values.data(), counts.data(), num_distinct_values, max_bin, total_sample_cnt, min_data_in_bin);
        if (bin_upper_bound_.size() == 2) { //如果只有
          missing_type_ = MissingType::None;
        }
      } else if (missing_type_ == MissingType::None) {  //不对缺失值进行处理
        bin_upper_bound_ = FindBinWithZeroAsOneBin(distinct_values.data(), counts.data(), num_distinct_values, max_bin, total_sample_cnt, min_data_in_bin);
      } else { //na表示缺失值，需要专门为na留出一个桶来
        bin_upper_bound_ = FindBinWithZeroAsOneBin(distinct_values.data(), counts.data(), num_distinct_values, max_bin - 1, total_sample_cnt - na_cnt, min_data_in_bin);
        bin_upper_bound_.push_back(NaN);
      }
      num_bin_ = static_cast<int>(bin_upper_bound_.size()); //桶的数量
      {
        cnt_in_bin.resize(num_bin_, 0); //将cnt_in_bin按照桶的数量调整大小，每个元素对应一个桶
        int i_bin = 0;  
        for (int i = 0; i < num_distinct_values; ++i) {
          if (distinct_values[i] > bin_upper_bound_[i_bin]) {
            ++i_bin;  //当distinct_values[i]大于桶的上界时，表示这个取值要放到下一个桶中，对桶进行切换
          }
          cnt_in_bin[i_bin] += counts[i]; //当distinct_values[i]小于桶的上界时，表示取值属于当前桶，进行累加
        }
        if (missing_type_ == MissingType::NaN) {  //当存在na并需要处理缺失值时，把最后一个桶分配给na
          cnt_in_bin[num_bin_ - 1] = na_cnt;
        }
      }
      CHECK(num_bin_ <= max_bin);
    } else { //如果是类别型特征，需要先把类别型数据转换为int
      // convert to int type first
      std::vector<int> distinct_values_int;
      std::vector<int> counts_int;
      for (size_t i = 0; i < distinct_values.size(); ++i) { //对转换后的整数进行合并
        int val = static_cast<int>(distinct_values[i]);
        if (val < 0) {
          na_cnt += counts[i];  //负数将被视为na
          Log::Warning("Met negative value in categorical features, will convert it to NaN");
        } else {
          if (distinct_values_int.empty() || val != distinct_values_int.back()) {
            distinct_values_int.push_back(val);
            counts_int.push_back(counts[i]);
          } else {
            counts_int.back() += counts[i];
          }
        }
      }
      num_bin_ = 0;
      int rest_cnt = static_cast<int>(total_sample_cnt - na_cnt);
      if (rest_cnt > 0) {
        const int SPARSE_RATIO = 100;
        if (distinct_values_int.back() / SPARSE_RATIO > static_cast<int>(distinct_values_int.size())) {
          Log::Warning("Met categorical feature which contains sparse values. "
                       "Consider renumbering to consecutive integers started from zero");
        }
        // sort by counts
        Common::SortForPair<int, int>(counts_int, distinct_values_int, 0, true);
        // avoid first bin is zero
        if (distinct_values_int[0] == 0) {  //当value的第一个为0时，为什么要避免第一个bin为0？
          if (counts_int.size() == 1) { //当只有一种取值时
            counts_int.push_back(0);
            distinct_values_int.push_back(distinct_values_int[0] + 1);
          }
          std::swap(counts_int[0], counts_int[1]);
          std::swap(distinct_values_int[0], distinct_values_int[1]);
        }
        // will ignore the categorical of small counts
        int cut_cnt = static_cast<int>((total_sample_cnt - na_cnt) * 0.99f);  //求出取值出现次数的下界，小于cut_cnt的取值将被忽略
        size_t cur_cat = 0; //当前取值的下标
        categorical_2_bin_.clear(); //类别型取值到桶的映射
        bin_2_categorical_.clear(); //桶到类别型数据的映射
        int used_cnt = 0;
        max_bin = std::min(static_cast<int>(distinct_values_int.size()), max_bin);  //当取值数小于max_bin则取取值数为桶的数量
        cnt_in_bin.clear();
        while (cur_cat < distinct_values_int.size()
               && (used_cnt < cut_cnt || num_bin_ < max_bin)) {
          if (counts_int[cur_cat] < min_data_in_bin && cur_cat > 1) {
            break;
          }
          bin_2_categorical_.push_back(distinct_values_int[cur_cat]); //每个桶对应一个取值
          categorical_2_bin_[distinct_values_int[cur_cat]] = static_cast<unsigned int>(num_bin_); //根据取值可以获得桶的编号
          used_cnt += counts_int[cur_cat];
          cnt_in_bin.push_back(counts_int[cur_cat]);
          ++num_bin_;
          ++cur_cat;
        }
        // need an additional bin for NaN
        if (cur_cat == distinct_values_int.size() && na_cnt > 0) {
          // use -1 to represent NaN
          bin_2_categorical_.push_back(-1);
          categorical_2_bin_[-1] = num_bin_;
          cnt_in_bin.push_back(0);
          ++num_bin_;
        }
        // Use MissingType::None to represent this bin contains all categoricals
        if (cur_cat == distinct_values_int.size() && na_cnt == 0) {
          missing_type_ = MissingType::None;
        } else if (na_cnt == 0) {
          missing_type_ = MissingType::Zero;
        } else {
          missing_type_ = MissingType::NaN;
        }
        cnt_in_bin.back() += static_cast<int>(total_sample_cnt - used_cnt);
      }
    }

    // check trivial(num_bin_ == 1) feature
    if (num_bin_ <= 1) {
      is_trivial_ = true;
    } else {
      is_trivial_ = false;
    }
    // check useless bin
    if (!is_trivial_ && NeedFilter(cnt_in_bin, static_cast<int>(total_sample_cnt), min_split_data, bin_type_)) {
      is_trivial_ = true;
    }

    if (!is_trivial_) {
      default_bin_ = ValueToBin(0);
      if (bin_type_ == BinType::CategoricalBin) {
        CHECK(default_bin_ > 0);
      }
    }
    if (!is_trivial_) {
      // calculate sparse rate
      sparse_rate_ = static_cast<double>(cnt_in_bin[default_bin_]) / static_cast<double>(total_sample_cnt);
    } else {
      sparse_rate_ = 1.0f;
    }
  }


  int BinMapper::SizeForSpecificBin(int bin) {
    int size = 0;
    size += sizeof(int);
    size += sizeof(MissingType);
    size += sizeof(bool);
    size += sizeof(double);
    size += sizeof(BinType);
    size += 2 * sizeof(double);
    size += bin * sizeof(double);
    size += sizeof(uint32_t);
    return size;
  }

  void BinMapper::CopyTo(char * buffer) const {
    std::memcpy(buffer, &num_bin_, sizeof(num_bin_));
    buffer += sizeof(num_bin_);
    std::memcpy(buffer, &missing_type_, sizeof(missing_type_));
    buffer += sizeof(missing_type_);
    std::memcpy(buffer, &is_trivial_, sizeof(is_trivial_));
    buffer += sizeof(is_trivial_);
    std::memcpy(buffer, &sparse_rate_, sizeof(sparse_rate_));
    buffer += sizeof(sparse_rate_);
    std::memcpy(buffer, &bin_type_, sizeof(bin_type_));
    buffer += sizeof(bin_type_);
    std::memcpy(buffer, &min_val_, sizeof(min_val_));
    buffer += sizeof(min_val_);
    std::memcpy(buffer, &max_val_, sizeof(max_val_));
    buffer += sizeof(max_val_);
    std::memcpy(buffer, &default_bin_, sizeof(default_bin_));
    buffer += sizeof(default_bin_);
    if (bin_type_ == BinType::NumericalBin) {
      std::memcpy(buffer, bin_upper_bound_.data(), num_bin_ * sizeof(double));
    } else {
      std::memcpy(buffer, bin_2_categorical_.data(), num_bin_ * sizeof(int));
    }
  }

  void BinMapper::CopyFrom(const char * buffer) {
    std::memcpy(&num_bin_, buffer, sizeof(num_bin_));
    buffer += sizeof(num_bin_);
    std::memcpy(&missing_type_, buffer, sizeof(missing_type_));
    buffer += sizeof(missing_type_);
    std::memcpy(&is_trivial_, buffer, sizeof(is_trivial_));
    buffer += sizeof(is_trivial_);
    std::memcpy(&sparse_rate_, buffer, sizeof(sparse_rate_));
    buffer += sizeof(sparse_rate_);
    std::memcpy(&bin_type_, buffer, sizeof(bin_type_));
    buffer += sizeof(bin_type_);
    std::memcpy(&min_val_, buffer, sizeof(min_val_));
    buffer += sizeof(min_val_);
    std::memcpy(&max_val_, buffer, sizeof(max_val_));
    buffer += sizeof(max_val_);
    std::memcpy(&default_bin_, buffer, sizeof(default_bin_));
    buffer += sizeof(default_bin_);
    if (bin_type_ == BinType::NumericalBin) {
      bin_upper_bound_ = std::vector<double>(num_bin_);
      std::memcpy(bin_upper_bound_.data(), buffer, num_bin_ * sizeof(double));
    } else {
      bin_2_categorical_ = std::vector<int>(num_bin_);
      std::memcpy(bin_2_categorical_.data(), buffer, num_bin_ * sizeof(int));
      categorical_2_bin_.clear();
      for (int i = 0; i < num_bin_; ++i) {
        categorical_2_bin_[bin_2_categorical_[i]] = static_cast<unsigned int>(i);
      }
    }
  }

  void BinMapper::SaveBinaryToFile(const VirtualFileWriter* writer) const {
    writer->Write(&num_bin_, sizeof(num_bin_));
    writer->Write(&missing_type_, sizeof(missing_type_));
    writer->Write(&is_trivial_, sizeof(is_trivial_));
    writer->Write(&sparse_rate_, sizeof(sparse_rate_));
    writer->Write(&bin_type_, sizeof(bin_type_));
    writer->Write(&min_val_, sizeof(min_val_));
    writer->Write(&max_val_, sizeof(max_val_));
    writer->Write(&default_bin_, sizeof(default_bin_));
    if (bin_type_ == BinType::NumericalBin) {
      writer->Write(bin_upper_bound_.data(), sizeof(double) * num_bin_);
    } else {
      writer->Write(bin_2_categorical_.data(), sizeof(int) * num_bin_);
    }
  }

  size_t BinMapper::SizesInByte() const {
    size_t ret = sizeof(num_bin_) + sizeof(missing_type_) + sizeof(is_trivial_) + sizeof(sparse_rate_)
      + sizeof(bin_type_) + sizeof(min_val_) + sizeof(max_val_) + sizeof(default_bin_);
    if (bin_type_ == BinType::NumericalBin) {
      ret += sizeof(double) *  num_bin_;
    } else {
      ret += sizeof(int) * num_bin_;
    }
    return ret;
  }

  template class DenseBin<uint8_t>;
  template class DenseBin<uint16_t>;
  template class DenseBin<uint32_t>;

  template class SparseBin<uint8_t>;
  template class SparseBin<uint16_t>;
  template class SparseBin<uint32_t>;

  template class OrderedSparseBin<uint8_t>;
  template class OrderedSparseBin<uint16_t>;
  template class OrderedSparseBin<uint32_t>;

  Bin* Bin::CreateBin(data_size_t num_data, int num_bin, double sparse_rate,
    bool is_enable_sparse, double sparse_threshold, bool* is_sparse) {
    // sparse threshold
    if (sparse_rate >= sparse_threshold && is_enable_sparse) {
      *is_sparse = true;
      return CreateSparseBin(num_data, num_bin);
    } else {
      *is_sparse = false;
      return CreateDenseBin(num_data, num_bin);
    }
  }

  Bin* Bin::CreateDenseBin(data_size_t num_data, int num_bin) {
    if (num_bin <= 16) {
      return new Dense4bitsBin(num_data);
    } else if (num_bin <= 256) {
      return new DenseBin<uint8_t>(num_data);
    } else if (num_bin <= 65536) {
      return new DenseBin<uint16_t>(num_data);
    } else {
      return new DenseBin<uint32_t>(num_data);
    }
  }

  Bin* Bin::CreateSparseBin(data_size_t num_data, int num_bin) {
    if (num_bin <= 256) {
      return new SparseBin<uint8_t>(num_data);
    } else if (num_bin <= 65536) {
      return new SparseBin<uint16_t>(num_data);
    } else {
      return new SparseBin<uint32_t>(num_data);
    }
  }

}  // namespace LightGBM
