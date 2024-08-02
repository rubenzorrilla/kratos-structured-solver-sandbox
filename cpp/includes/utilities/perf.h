#include <chrono>

namespace Perf {
    template <class F> 
    void execute(const std::string & name, const F& f) {
        using namespace std::chrono;

        auto chron_beg = high_resolution_clock::now();
        
        f();

        auto chron_end = high_resolution_clock::now();

        std::cout << " === Executed " << name << " === in \t" << std::chrono::duration_cast<std::chrono::milliseconds>(chron_end - chron_beg) << std::endl;
    }
}