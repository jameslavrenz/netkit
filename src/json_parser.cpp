#include "json_parser.hpp"
#include <cstdlib>
#include <cstring>

namespace Json
{
    const char* SkipWhitespace(const char* p, const char* end)
    {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r'))
            ++p;
        return p;
    }

    bool MatchChar(const char*& p, const char* end, char c)
    {
        p = SkipWhitespace(p, end);
        if (p >= end || *p != c)
            return false;
        ++p;
        return true;
    }

    bool MatchLiteral(const char*& p, const char* end, const char* lit)
    {
        p = SkipWhitespace(p, end);
        std::size_t len = std::strlen(lit);
        if (static_cast<std::size_t>(end - p) < len || std::memcmp(p, lit, len) != 0)
            return false;
        p += len;
        return true;
    }

    bool ParseString(const char*& p, const char* end, char* out, std::size_t out_capacity)
    {
        p = SkipWhitespace(p, end);
        if (p >= end || *p != '"')
            return false;

        ++p;
        std::size_t i = 0;
        while (p < end && *p != '"')
        {
            if (i + 1 < out_capacity)
                out[i++] = *p;
            ++p;
        }

        if (p >= end || *p != '"')
            return false;

        if (out_capacity > 0)
            out[i < out_capacity ? i : out_capacity - 1] = '\0';

        ++p;
        return true;
    }

    bool ParseFloat(const char*& p, const char* end, float& out)
    {
        p = SkipWhitespace(p, end);
        if (p >= end)
            return false;

        char* after = nullptr;
        out = std::strtof(const_cast<char*>(p), &after);
        if (after == p)
            return false;

        p = after;
        return true;
    }

    bool ParseUint(const char*& p, const char* end, uint32_t& out)
    {
        p = SkipWhitespace(p, end);
        if (p >= end)
            return false;

        char* after = nullptr;
        const unsigned long value = std::strtoul(const_cast<char*>(p), &after, 10);
        if (after == p)
            return false;

        out = static_cast<uint32_t>(value);
        p = after;
        return true;
    }

    const char* FindKeyValue(const char* begin, const char* end, const char* key)
    {
        const std::size_t key_len = std::strlen(key);

        for (const char* p = begin; p + key_len + 2 < end; ++p)
        {
            if (*p != '"')
                continue;

            if (std::memcmp(p + 1, key, key_len) != 0 || p[1 + key_len] != '"')
                continue;

            const char* q = SkipWhitespace(p + 2 + key_len, end);
            if (q >= end || *q != ':')
                continue;

            return SkipWhitespace(q + 1, end);
        }

        return nullptr;
    }

    const char* SkipValue(const char* cursor, const char* end)
    {
        cursor = SkipWhitespace(cursor, end);
        if (cursor >= end)
            return end;

        if (*cursor == '"')
        {
            ++cursor;
            while (cursor < end && *cursor != '"')
            {
                if (*cursor == '\\' && cursor + 1 < end)
                    cursor += 2;
                else
                    ++cursor;
            }
            if (cursor < end)
                ++cursor;
            return cursor;
        }

        if (*cursor == '{')
        {
            int depth = 0;
            for (const char* p = cursor; p < end; ++p)
            {
                if (*p == '{')
                    ++depth;
                else if (*p == '}')
                {
                    --depth;
                    if (depth == 0)
                        return p + 1;
                }
            }
            return end;
        }

        if (*cursor == '[')
        {
            int depth = 0;
            for (const char* p = cursor; p < end; ++p)
            {
                if (*p == '[')
                    ++depth;
                else if (*p == ']')
                {
                    --depth;
                    if (depth == 0)
                        return p + 1;
                }
            }
            return end;
        }

        while (cursor < end && *cursor != ',' && *cursor != ']' && *cursor != '}')
            ++cursor;

        return cursor;
    }

    bool NextArrayElement(const char*& cursor, const char* end, const char*& elem_begin, const char*& elem_end)
    {
        cursor = SkipWhitespace(cursor, end);

        if (cursor >= end)
            return false;

        if (*cursor == '[')
            ++cursor;

        cursor = SkipWhitespace(cursor, end);
        if (cursor >= end || *cursor == ']')
            return false;

        if (*cursor == ',')
        {
            ++cursor;
            cursor = SkipWhitespace(cursor, end);
        }

        if (cursor >= end || *cursor == ']')
            return false;

        elem_begin = cursor;
        elem_end = SkipValue(cursor, end);
        cursor = SkipWhitespace(elem_end, end);
        return true;
    }
}
