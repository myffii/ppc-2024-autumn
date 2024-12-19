#include <boost/serialization/vector.hpp>

namespace boost {
namespace serialization {

template <class Archive>
void serialize(Archive &ar, std::vector<std::vector<double>> &matrix, const unsigned int version) {
  for (auto &row : matrix) {
    ar & row;
  }
}

} // namespace serialization
} // namespace boost