#pragma once
#include <cstddef>

#ifndef RING_SIZE
inline constexpr std::size_t RING_SIZE      = 4096;
#endif
#ifndef LOCAL_BUF_SIZE
inline constexpr std::size_t LOCAL_BUF_SIZE = 64;
#endif
#ifndef BATCH_SIZE
inline constexpr std::size_t BATCH_SIZE     = 256;
#endif

inline constexpr long        MAX_BACKOFF_NS = 100'000;
inline constexpr int         PREPOST_RECVS  = 8;
