#include "ternative/cache.h"
#include "ternative/tensor.h"

#include <cstring>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <sys/stat.h>
#include <iostream>
#include <unordered_map>

#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <direct.h>
#  include <io.h>
#  define MKDIR(p)   _mkdir(p)
#  define ACCESS(p)  _access(p, 0)
#else
#  include <sys/mman.h>
#  include <fcntl.h>
#  include <unistd.h>
#  include <errno.h>
#  define MKDIR(p)   mkdir(p, 0755)
#  define ACCESS(p)  access(p, F_OK)
#endif

namespace ternative {

// ---------------------------------------------------------------------------
// Portable SHA-256 (no external deps)
// Based on RFC 6234 pseudocode.
// ---------------------------------------------------------------------------

namespace sha256_impl {

static const uint32_t K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,
    0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
    0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,
    0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,
    0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
    0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,
    0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,
    0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
    0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

static inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
static inline uint32_t Ch(uint32_t x, uint32_t y, uint32_t z)  { return (x & y) ^ (~x & z); }
static inline uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
static inline uint32_t S0(uint32_t x) { return rotr(x,2)  ^ rotr(x,13) ^ rotr(x,22); }
static inline uint32_t S1(uint32_t x) { return rotr(x,6)  ^ rotr(x,11) ^ rotr(x,25); }
static inline uint32_t s0(uint32_t x) { return rotr(x,7)  ^ rotr(x,18) ^ (x>>3);     }
static inline uint32_t s1(uint32_t x) { return rotr(x,17) ^ rotr(x,19) ^ (x>>10);    }

static inline uint32_t be32(const uint8_t* p) {
    return ((uint32_t)p[0]<<24)|((uint32_t)p[1]<<16)|((uint32_t)p[2]<<8)|(uint32_t)p[3];
}

struct Ctx {
    uint32_t h[8];
    uint8_t  buf[64];
    uint64_t bits;
    uint32_t blen;
};

static void init(Ctx& c) {
    c.h[0]=0x6a09e667; c.h[1]=0xbb67ae85; c.h[2]=0x3c6ef372; c.h[3]=0xa54ff53a;
    c.h[4]=0x510e527f; c.h[5]=0x9b05688c; c.h[6]=0x1f83d9ab; c.h[7]=0x5be0cd19;
    c.bits=0; c.blen=0;
}

static void compress(Ctx& c, const uint8_t blk[64]) {
    uint32_t w[64];
    for (int i=0;i<16;++i) w[i]=be32(blk+4*i);
    for (int i=16;i<64;++i) w[i]=s1(w[i-2])+w[i-7]+s0(w[i-15])+w[i-16];
    uint32_t a=c.h[0],b=c.h[1],d=c.h[2],e=c.h[3],f=c.h[4],g=c.h[5],hh=c.h[6],t=c.h[7];
    for (int i=0;i<64;++i){
        uint32_t T1=t+S1(f)+Ch(f,g,hh)+K[i]+w[i];
        uint32_t T2=S0(a)+Maj(a,b,d);
        t=hh; hh=g; g=f; f=e+(T1); e=d; d=b; b=a; a=T1+T2;
        (void)d; // suppress warning — kept in local chain
    }
    // Note: local var 'd' above shadows loop; rewrite cleanly:
    // recompute with proper names
    uint32_t aa=c.h[0],bb=c.h[1],cc=c.h[2],dd=c.h[3],
             ee=c.h[4],ff=c.h[5],gg=c.h[6],hx=c.h[7];
    for (int i=0;i<64;++i){
        uint32_t T1=hx+S1(ee)+Ch(ee,ff,gg)+K[i]+w[i];
        uint32_t T2=S0(aa)+Maj(aa,bb,cc);
        hx=gg; gg=ff; ff=ee; ee=dd+T1; dd=cc; cc=bb; bb=aa; aa=T1+T2;
    }
    c.h[0]+=aa; c.h[1]+=bb; c.h[2]+=cc; c.h[3]+=dd;
    c.h[4]+=ee; c.h[5]+=ff; c.h[6]+=gg; c.h[7]+=hx;
}

static void update(Ctx& c, const uint8_t* data, size_t len) {
    c.bits += (uint64_t)len * 8;
    while (len > 0) {
        size_t cp = 64 - c.blen;
        if (cp > len) cp = len;
        memcpy(c.buf + c.blen, data, cp);
        c.blen += (uint32_t)cp; data += cp; len -= cp;
        if (c.blen == 64) { compress(c, c.buf); c.blen = 0; }
    }
}

static void finalize(Ctx& c, uint8_t out[32]) {
    uint64_t bits = c.bits;
    uint8_t pad = 0x80;
    update(c, &pad, 1);
    while (c.blen != 56) { pad = 0; update(c, &pad, 1); }
    uint8_t len_be[8];
    for (int i=7;i>=0;--i){ len_be[i]=(uint8_t)(bits&0xff); bits>>=8; }
    update(c, len_be, 8);
    for (int i=0;i<8;++i){
        out[4*i]  =(c.h[i]>>24)&0xff; out[4*i+1]=(c.h[i]>>16)&0xff;
        out[4*i+2]=(c.h[i]>> 8)&0xff; out[4*i+3]=(c.h[i]    )&0xff;
    }
}

} // namespace sha256_impl

// ---------------------------------------------------------------------------
// ModelCache implementation
// ---------------------------------------------------------------------------

bool ModelCache::sha256_file(const std::string& path, uint8_t out[32]) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    sha256_impl::Ctx ctx;
    sha256_impl::init(ctx);
    char buf[65536];
    while (f) {
        f.read(buf, sizeof(buf));
        size_t n = (size_t)f.gcount();
        if (n > 0) sha256_impl::update(ctx, (const uint8_t*)buf, n);
    }
    sha256_impl::finalize(ctx, out);
    return true;
}

std::string ModelCache::hex(const uint8_t bytes[32]) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (int i = 0; i < 32; ++i) oss << std::setw(2) << (int)bytes[i];
    return oss.str();
}

CacheKey ModelCache::compute_key(const std::string& base_path,
                                 const std::string& lora_path) {
    CacheKey key;
    uint8_t h[32] = {};
    if (sha256_file(base_path, h)) key.base_sha256_hex = hex(h);
    if (!lora_path.empty()) {
        if (sha256_file(lora_path, h)) key.lora_sha256_hex = hex(h);
    }
    return key;
}

std::string ModelCache::default_cache_dir() {
#ifdef _WIN32
    const char* appdata = getenv("LOCALAPPDATA");
    if (appdata) return std::string(appdata) + "\\ternative\\cache";
    return "C:\\Users\\Public\\ternative\\cache";
#else
    const char* home = getenv("HOME");
    if (home) return std::string(home) + "/.cache/ternative";
    return "/tmp/ternative_cache";
#endif
}

bool ModelCache::mkdir_p(const std::string& dir) {
    if (ACCESS(dir.c_str()) == 0) return true;

    // Create parent first
    size_t pos = dir.find_last_of("/\\");
    if (pos != std::string::npos && pos > 0) {
        std::string parent = dir.substr(0, pos);
        if (!mkdir_p(parent)) return false;
    }
    return MKDIR(dir.c_str()) == 0 || ACCESS(dir.c_str()) == 0;
}

std::string ModelCache::cache_path(const CacheKey& key,
                                   const std::string& cache_dir) {
    std::string dir = cache_dir.empty() ? default_cache_dir() : cache_dir;
    // Use first 16 hex chars of each hash for filename to keep it short
    std::string base8 = key.base_sha256_hex.substr(0, 16);
    std::string lora8 = key.lora_sha256_hex.empty()
                        ? "nolora"
                        : key.lora_sha256_hex.substr(0, 16);
    return dir + "/" + base8 + "_" + lora8 + ".tvcache";
}

// ---------------------------------------------------------------------------
// Helpers to iterate model tensors
// ---------------------------------------------------------------------------
struct TensorRef { std::string name; const Tensor* t; };

static std::vector<TensorRef> collect_tensors(const Model& model) {
    std::vector<TensorRef> refs;
    refs.push_back({"tok_embd",    &model.tok_embd});
    refs.push_back({"output_norm", &model.output_norm});
    for (int l = 0; l < (int)model.layers.size(); ++l) {
        auto pref = "blk." + std::to_string(l) + ".";
        const auto& lw = model.layers[l];
        refs.push_back({pref + "attn_norm",     &lw.attn_norm});
        refs.push_back({pref + "attn_q",        &lw.wq});
        refs.push_back({pref + "attn_k",        &lw.wk});
        refs.push_back({pref + "attn_v",        &lw.wv});
        refs.push_back({pref + "attn_output",   &lw.wo});
        if (!lw.attn_sub_norm.shape.empty())
            refs.push_back({pref + "attn_sub_norm", &lw.attn_sub_norm});
        refs.push_back({pref + "ffn_norm",      &lw.ffn_norm});
        refs.push_back({pref + "ffn_gate",      &lw.w_gate});
        refs.push_back({pref + "ffn_up",        &lw.w_up});
        refs.push_back({pref + "ffn_down",      &lw.w_down});
        if (!lw.ffn_sub_norm.shape.empty())
            refs.push_back({pref + "ffn_sub_norm",  &lw.ffn_sub_norm});
    }
    return refs;
}

// Mutable version for loading into model
struct TensorMutRef { std::string name; Tensor* t; };

static std::vector<TensorMutRef> collect_tensors_mut(Model& model) {
    std::vector<TensorMutRef> refs;
    refs.push_back({"tok_embd",    &model.tok_embd});
    refs.push_back({"output_norm", &model.output_norm});
    for (int l = 0; l < (int)model.layers.size(); ++l) {
        auto pref = "blk." + std::to_string(l) + ".";
        auto& lw = model.layers[l];
        refs.push_back({pref + "attn_norm",     &lw.attn_norm});
        refs.push_back({pref + "attn_q",        &lw.wq});
        refs.push_back({pref + "attn_k",        &lw.wk});
        refs.push_back({pref + "attn_v",        &lw.wv});
        refs.push_back({pref + "attn_output",   &lw.wo});
        refs.push_back({pref + "attn_sub_norm", &lw.attn_sub_norm});
        refs.push_back({pref + "ffn_norm",      &lw.ffn_norm});
        refs.push_back({pref + "ffn_gate",      &lw.w_gate});
        refs.push_back({pref + "ffn_up",        &lw.w_up});
        refs.push_back({pref + "ffn_down",      &lw.w_down});
        refs.push_back({pref + "ffn_sub_norm",  &lw.ffn_sub_norm});
    }
    return refs;
}

// ---------------------------------------------------------------------------
// Save
// ---------------------------------------------------------------------------
void ModelCache::save(const Model& model,
                      const std::string& path,
                      const CacheKey& key) {
    // Ensure directory exists
    size_t sep = path.find_last_of("/\\");
    if (sep != std::string::npos) mkdir_p(path.substr(0, sep));

    auto tensors = collect_tensors(model);

    // Filter out empty tensors
    std::vector<TensorRef> valid;
    for (auto& tr : tensors)
        if (!tr.t->shape.empty() && !tr.t->data.empty())
            valid.push_back(tr);

    // Build tensor table
    std::vector<TvCacheTensor> table(valid.size());
    uint64_t data_off = 0;
    for (size_t i = 0; i < valid.size(); ++i) {
        auto& te = table[i];
        memset(&te, 0, sizeof(te));
        strncpy(te.name, valid[i].name.c_str(), 127);
        te.dtype = (uint32_t)valid[i].t->type;  // preserve actual dtype (F16, F32, BF16, etc.)
        te.n_dims = (uint32_t)valid[i].t->shape.size();
        for (size_t d = 0; d < valid[i].t->shape.size() && d < 4; ++d)
            te.shape[d] = valid[i].t->shape[d];
        te.data_offset = data_off;
        te.nbytes = valid[i].t->data.size();
        data_off += te.nbytes;
        // Align to 64 bytes
        data_off = (data_off + 63) & ~uint64_t(63);
    }

    // Header
    TvCacheHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, "TVCACHE\0", 8);
    hdr.version = TVCACHE_VERSION;
    hdr.dtype   = 1;
    hdr.lora_alpha = model.lora_alpha;
    hdr.lora_rank  = (uint32_t)model.lora_rank;
    hdr.num_tensors = (uint32_t)valid.size();

    if (!key.base_sha256_hex.empty()) {
        // Convert hex back to bytes
        for (int i = 0; i < 32 && i * 2 + 1 < (int)key.base_sha256_hex.size(); ++i) {
            hdr.base_sha256[i] = (uint8_t)strtol(
                key.base_sha256_hex.substr(i * 2, 2).c_str(), nullptr, 16);
        }
    }
    if (!key.lora_sha256_hex.empty()) {
        for (int i = 0; i < 32 && i * 2 + 1 < (int)key.lora_sha256_hex.size(); ++i) {
            hdr.lora_sha256[i] = (uint8_t)strtol(
                key.lora_sha256_hex.substr(i * 2, 2).c_str(), nullptr, 16);
        }
    }

    uint64_t tensor_table_off = sizeof(TvCacheHeader);
    uint64_t data_section_off = tensor_table_off
                                + (uint64_t)valid.size() * sizeof(TvCacheTensor);
    // Align data section to 4096
    data_section_off = (data_section_off + 4095) & ~uint64_t(4095);

    hdr.tensor_table_offset = tensor_table_off;
    hdr.data_offset         = data_section_off;
    hdr.total_file_size     = data_section_off + data_off;

    // Write to temp file first, then atomically rename — prevents partial-write corruption
    std::string tmp_path = path + ".tmp";
    std::ofstream f(tmp_path, std::ios::binary | std::ios::trunc);
    if (!f) {
        std::cerr << "[Cache] Cannot write tmp: " << tmp_path << "\n";
        return;
    }

    f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    for (auto& te : table)
        f.write(reinterpret_cast<const char*>(&te), sizeof(te));

    // Pad to data_section_off
    {
        uint64_t cur = sizeof(hdr) + (uint64_t)table.size() * sizeof(TvCacheTensor);
        if (cur < data_section_off) {
            std::vector<char> pad(data_section_off - cur, 0);
            f.write(pad.data(), (std::streamsize)pad.size());
        }
    }

    // Write tensor data
    static const char zeros[64] = {};
    for (size_t i = 0; i < valid.size(); ++i) {
        const auto& d = valid[i].t->data;
        f.write(reinterpret_cast<const char*>(d.data()), (std::streamsize)d.size());
        // Align to 64 bytes
        uint64_t nb = (uint64_t)d.size();
        uint64_t aligned = (nb + 63) & ~uint64_t(63);
        if (aligned > nb)
            f.write(zeros, (std::streamsize)(aligned - nb));
    }

    f.close();
    // Atomic rename: replace any existing file only after full write
    std::remove(path.c_str());
    if (std::rename(tmp_path.c_str(), path.c_str()) != 0) {
        std::cerr << "[Cache] Rename failed: " << tmp_path << " -> " << path << "\n";
        std::remove(tmp_path.c_str());
        return;
    }
    std::cerr << "[Cache] Saved: " << path << " ("
              << (hdr.total_file_size / (1024*1024)) << " MB)\n";
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------
bool ModelCache::try_load(Model& model, const std::string& path) {
    // Check file exists
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    // Read header
    TvCacheHeader hdr;
    f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (!f || f.gcount() != sizeof(hdr)) return false;

    if (memcmp(hdr.magic, "TVCACHE\0", 8) != 0) {
        std::cerr << "[Cache] Bad magic: " << path << "\n";
        return false;
    }
    if (hdr.version != TVCACHE_VERSION) {
        std::cerr << "[Cache] Version mismatch: " << hdr.version << "\n";
        return false;
    }

    // Read tensor table
    std::vector<TvCacheTensor> table(hdr.num_tensors);
    f.seekg((std::streampos)hdr.tensor_table_offset);
    for (uint32_t i = 0; i < hdr.num_tensors; ++i) {
        f.read(reinterpret_cast<char*>(&table[i]), sizeof(TvCacheTensor));
        if (!f) { std::cerr << "[Cache] Truncated tensor table\n"; return false; }
    }

    std::cerr << "[Cache] Header OK: " << hdr.num_tensors << " tensors, "
              << "data_off=" << hdr.data_offset << "\n";

    // Build a name → TvCacheTensor map
    std::unordered_map<std::string, TvCacheTensor> by_name;
    for (auto& te : table) by_name[std::string(te.name)] = te;

    std::cerr << "[Cache] Tensor table loaded\n";

    // Get mutable tensor refs from model
    auto refs = collect_tensors_mut(model);
    std::cerr << "[Cache] Loading " << refs.size() << " tensors from file...\n";

    // Read each tensor individually from its offset — avoids 4.6GB single allocation
    int loaded = 0;
    for (auto& ref : refs) {
        auto it = by_name.find(ref.name);
        if (it == by_name.end()) continue;
        const TvCacheTensor& te = it->second;

        if (te.n_dims == 0 || te.nbytes == 0) continue;

        std::vector<int64_t> shape;
        for (uint32_t d = 0; d < te.n_dims && d < 4; ++d)
            shape.push_back((int64_t)te.shape[d]);

        try {
            *ref.t = Tensor(shape, static_cast<ggml_type>(te.dtype));
        } catch (const std::bad_alloc& e) {
            std::cerr << "[Cache] OOM allocating tensor " << ref.name
                      << " (" << te.nbytes / (1024*1024) << " MB): " << e.what() << "\n";
            return false;
        } catch (const std::exception& ex) {
            std::cerr << "[Cache] Exception allocating " << ref.name << ": " << ex.what() << "\n";
            return false;
        }
        if (ref.t->data.empty()) continue;

        uint64_t file_off = hdr.data_offset + te.data_offset;
        f.seekg(static_cast<std::ios::off_type>(file_off), std::ios::beg);
        if (!f.good()) {
            std::cerr << "[Cache] Seek failed for " << ref.name
                      << " at offset " << file_off << "\n";
            return false;
        }
        f.read(reinterpret_cast<char*>(ref.t->data.data()),
               static_cast<std::streamsize>(te.nbytes));
        if (static_cast<uint64_t>(f.gcount()) != te.nbytes) {
            std::cerr << "[Cache] Short read for " << ref.name
                      << ": " << f.gcount() << " of " << te.nbytes << "\n";
            return false;
        }
        ++loaded;
        if (loaded % 50 == 0)
            std::cerr << "[Cache] Loaded " << loaded << "/" << refs.size() << "\n";
    }
    std::cerr << "[Cache] All tensors loaded (" << loaded << ")\n";

    model.lora_alpha = hdr.lora_alpha;
    model.lora_rank  = (int)hdr.lora_rank;
    model.has_lora   = hdr.lora_rank > 0;

    std::cerr << "[Cache] Loaded: " << path << " ("
              << hdr.num_tensors << " tensors)\n";
    return true;
}

} // namespace ternative
