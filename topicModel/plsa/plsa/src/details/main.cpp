#include "../public.h"
#include "../conf.h"
#include "../data.h"
#include "../plsa.h"

namespace xforce {
LOGGER_IMPL(xforce_logger, "ml")
}

int main() {
    LOGGER_SYS_INIT("conf/log.conf")

    xforce::ml::Conf conf;
    if (!conf.Init()) {
        std::cout << "fail_init_conf" << std::endl;
        return 1;
    }

    xforce::ml::Data data;
    if (!data.Init(conf)) {
        std::cout << "fail_init_data" << std::endl;
        return 2;
    }

    xforce::ml::Plsa plsa(conf, data);
    plsa.Start();
    plsa.Stop();
    return 0;
}
