#pragma once
#include <cstddef>
#include <cstdint>

namespace Json
{
    constexpr std::size_t kMaxStringLen = 64;

    const char* SkipWhitespace(const char* p, const char* end);

    bool MatchChar(const char*& p, const char* end, char c);
    bool MatchLiteral(const char*& p, const char* end, const char* lit);

    bool ParseString(const char*& p, const char* end, char* out, std::size_t out_capacity);
    bool ParseFloat(const char*& p, const char* end, float& out);
    bool ParseUint(const char*& p, const char* end, uint32_t& out);

    // Search object/array text for "key": value; returns pointer to start of value.
    const char* FindKeyValue(const char* begin, const char* end, const char* key);

    // Given cursor inside an array (at '[' or first element), advance to next element.
    // Returns false when the array is exhausted.
    bool NextArrayElement(const char*& cursor, const char* end, const char*& elem_begin, const char*& elem_end);

    // Skip a single JSON value starting at cursor; returns pointer after the value.
    const char* SkipValue(const char* cursor, const char* end);
}
