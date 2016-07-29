#include "../conf.h"

namespace xforce { namespace ml {

bool Conf::Init() {
    static const std::string kConfpath = "conf/plsa.conf";

    const JsonType *conf = JsonType::CreateConf(kConfpath);
    XFC_FAIL_HANDLE_FATAL(NULL==conf, "fail_init_conf[" << kConfpath << "]")

    XFC_FAIL_HANDLE_FATAL(
        !(*conf)["num_threads"].IsInt() || 
        (*conf)["num_threads"].AsInt() <= 0,
        "fail_init_num_threads"
    )
    numThreads_ = (*conf)["num_threads"].AsInt();

    XFC_FAIL_HANDLE_FATAL(
        !(*conf)["datapath"].IsStr(),
        "fail_init_num_threads"
    )
    datapath_ = (*conf)["datapath"].AsStr();

    XFC_FAIL_HANDLE_FATAL(
        !(*conf)["epsilon"].IsDouble() || 
        (*conf)["epsilon"].AsDouble() <= 0,
        "fail_init_epsilon"
    )
    epsilon_ = (*conf)["epsilon"].AsDouble();

    XFC_DELETE(conf)
    return true;

    ERROR_HANDLE:
    XFC_DELETE(conf)
    return false;
}

}}
