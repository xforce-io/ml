#pragma once

#include "common.h"

namespace xforce {

class StrHelper {
  public:
    inline static void SplitStr(const std::string& str, char sep, std::vector<std::string>& vals);
    inline static void SplitStr(const char* str, char sep, std::vector<std::string>& vals);

    template <typename T>
    inline static bool GetNum(IN  const char* str, OUT T& num);

    template <typename T>
	  inline static std::string GetStr(IN T num );
};

void StrHelper::SplitStr(const std::string& str, char sep, std::vector<std::string>& vals) {
    SplitStr(str.c_str(), sep, vals);
}

void StrHelper::SplitStr(const char* str, char sep, std::vector<std::string>& vals) {
    vals.clear();

    std::string tmp_str;
    const char *ptr_0=str, *ptr_1=ptr_0;
    while (true) {
        while ('\0' != *ptr_1 && sep != *ptr_1) {
            ++ptr_1;
        }

        tmp_str.assign("");
        tmp_str.append(ptr_0, ptr_1-ptr_0);
        vals.push_back(tmp_str);

        if ('\0' == *ptr_1) break;
        ptr_0 = ++ptr_1;
    }
}

template <typename T>
bool StrHelper::GetNum(const char* str, T& num) {
    char* endptr;
    int64_t tmp_int = strtoll(str, &endptr, 10);
    if ( '\0' != *endptr 
            || SCAST<T>(tmp_int) < std::numeric_limits<T>::min()
            || SCAST<T>(tmp_int) > std::numeric_limits<T>::max() ) {
        return false;
    }
    num = SCAST<T>(tmp_int);
    return true;
}

template <typename T>
std::string StrHelper::GetStr(IN T num ) {
	  std::stringstream ss;
	  ss << num;
    return ss.str();
}

}
