#include "../data.h"

namespace xforce { namespace ml {

bool Data::Init(const Conf &conf) {
    bool ret;
    std::vector<std::string> items;
    size_t sizeBuf = (1<<20);
    char *buf = RCAST<char*>(malloc(sizeBuf));

    std::string tmpStr;
    FILE *fp = fopen(conf.GetDatapath().c_str(), "r");
    XFC_FAIL_HANDLE_FATAL(NULL==fp,
        "fail_open_datapath[" << conf.GetDatapath() << "]")

    XFC_FAIL_HANDLE_FATAL(NULL == fgets(buf, sizeBuf, fp), 
        "fail_parse_first_line_of_data[" << conf.GetDatapath() << "]")
    if (buf[strlen(buf)-1] == '\n') {
        buf[strlen(buf)-1] = '\0';
    }

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
        std::cout << i << std::endl;
        XFC_FAIL_HANDLE_FATAL(NULL == fgets(buf, sizeBuf, fp), 
            "fail_parse_line_of_data[" << i << "]")

        if (buf[strlen(buf)-1] == '\n') {
            buf[strlen(buf)-1] = '\0';
        }

        StrHelper::SplitStr(buf, '\t', items);
        XFC_FAIL_HANDLE_FATAL(items.size() != numWords_,
            "fail_parse_line_of_data[" << i << "]")

        for (size_t j=0; j<numWords_; ++j) {
            ret = StrHelper::GetNum(items[j].c_str(), accuDocWords_[i][j]);
            XFC_FAIL_HANDLE_FATAL(!ret,
                "fail_parse_line_of_data[" << i << "]")

            accuDocs_[i] += accuDocWords_[i][j];
        }
    }
    fclose(fp);
    free(buf);
    return true;

    ERROR_HANDLE:
    return false;
}

}}
