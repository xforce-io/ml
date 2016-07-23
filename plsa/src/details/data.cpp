#include "../data.h"


namespace xforce { namespace ml {

bool Data::Init(const Conf &conf) {
    bool ret;
    std::vector<std::string> items;
    size_t sizeBuf = (1<<20);
    char *buf = new char [sizeBuf];

    std::string tmpStr;
    std::ifstream fin;
    fin.open(conf.GetDatapath().c_str(), std::ios::in);
    XFC_FAIL_HANDLE_FATAL(!fin.good(),
        "fail_open_datapath[" << conf.GetDatapath() << "]")

    fin.getline(buf, sizeBuf);
    XFC_FAIL_HANDLE_FATAL(fin.fail(), 
        "fail_parse_first_line_of_data[" << conf.GetDatapath() << "]")

    StrHelper::SplitStr(buf, ',', items);
    XFC_FAIL_HANDLE_FATAL(items.size() != 3, 
        "fail_parse_first_line_of_data[" << conf.GetDatapath() << "]")

    ret = StrHelper::GetNum(items[0].c_str(), numTopics_); 
    XFC_FAIL_HANDLE_FATAL(!ret, 
        "fail_parse_first_line_of_data[" << conf.GetDatapath() << "]")

    ret = StrHelper::GetNum(items[1].c_str(), numDocs_); 
    XFC_FAIL_HANDLE_FATAL(!ret, 
        "fail_parse_first_line_of_data[" << conf.GetDatapath() << "]")

    ret = StrHelper::GetNum(items[2].c_str(), numWords_); 
    XFC_FAIL_HANDLE_FATAL(!ret, 
        "fail_parse_first_line_of_data[" << conf.GetDatapath() << "]")

    XFC_NEW(accuDocWords_, size_t* [numDocs_])
    for (size_t i=0; i<numDocs_; ++i) {
      XFC_NEW(accuDocWords_[i], size_t [numWords_])
    }
    memset(accuDocWords_, 0, sizeof(accuDocWords_[0][0]) * numDocs_ * numWords_);

    XFC_NEW(accuDocs_, size_t [numDocs_])
    memset(accuDocs_, 0, sizeof(accuDocs_[0]) * numDocs_);

    for (size_t i=0; i<numDocs_; ++i) {
        fin.getline(buf, sizeBuf);
        XFC_FAIL_HANDLE_FATAL(fin.fail(), 
            "fail_parse_line_of_data[" << i << "]")

        StrHelper::SplitStr(tmpStr, '\t', items);
        XFC_FAIL_HANDLE_FATAL(items.size() != numWords_,
            "fail_parse_line_of_data[" << i << "]")

        for (size_t j=0; j<numWords_; ++j) {
            ret = StrHelper::GetNum(items[j].c_str(), accuDocWords_[i][j]);
            XFC_FAIL_HANDLE_FATAL(!ret,
                "fail_parse_line_of_data[" << i << "]")

            accuDocs_[i] += accuDocWords_[i][j];
        }
    }
    delete buf;
    return true;

    ERROR_HANDLE:
    return false;
}

}}
