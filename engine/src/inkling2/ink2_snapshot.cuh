// Disk sessions for --serve, in the protocol of Strata's server (terminal/serve/server.py, _session_file: the
// llama-server style POST /slots/0?action=save|restore with --slot-save-path, and the conversation kept across an
// image job's unload).  Between requests:
//
//   SAVE <path>      ->  SAVED <tokens> <bytes> <ms>
//   RESTORE <path>   ->  RESTORED <tokens> <bytes> <ms>
//   either one       ->  SERR <invalid|storage|memory|io> 0 <reason>   (refused or failed, the engine in step)
//   RESTORE          ->  FATAL <reason>, and the engine exits            (failed after the device was written)
//   on the way       ->  SESSION <done> <total>                         (after each block of the file, 16 MiB at most)
//
// The state is the conversation cache's (ink2_session.cuh, sess_arrays): after n tokens, the full-attention rows [0, n),
// the sliding-window rows of positions [n - WINDOW, n), the conv rings, with --spec the drafter's K/V; plus the n tokens
// and the logits after the last one.  Rows are in position order, so a file does not depend on --chunk, --max-context
// or --kv-grow.  The header records --kv, --spec, the layout and a fingerprint of the weights (embedding rows and a
// slice of the unembedding): a RESTORE with any of them different is refused.
//
// A refused RESTORE leaves the live conversation as it was; the file's size is checked before the device is touched.
// With the conversation cache on, a RESTORE first parks the live conversation, as a request for another one would.
// SAVE writes <path>.part and renames it, so a SAVE that fails while writing leaves an older file at <path> as it was.
// Nothing here runs unless the controller sends one of the lines.

struct SnapHdr {
    char magic[8];   // "INK2KV01"
    uint32_t kvm, spec, ntok, vocab;
    uint32_t nfull, nring, nconv, window;
    uint64_t full_row_bytes, ring_row_bytes, conv_bytes, weights;
};

// FNV-1a over embedding rows 0..63 and the first 64 KB of the unembedding: tells checkpoints apart without reading them
uint64_t snap_weights_hash() {
    static uint64_t cached = 0;
    if (cached) return cached;
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](const void* p, size_t n) {
        const uint8_t* b = static_cast<const uint8_t*>(p);
        for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    };
    vec e(H);
    for (int t = 0; t < 64; ++t) { embed_row(t, e.data()); mix(e.data(), e.size() * 4); }
    std::vector<uint8_t> u(64 << 10);
    const void* src = E.unembed.d ? (const void*) E.unembed.d : (const void*) E.unembed.q;
    if (src) { CK(cudaMemcpy(u.data(), src, u.size(), cudaMemcpyDeviceToHost)); mix(u.data(), u.size()); }
    return cached = h;
}

struct SnapLayout {
    std::vector<SessArr> full, ring;
    std::vector<std::pair<void*, size_t>> conv;
    SnapLayout() { sess_arrays(full, ring, conv); }
    SnapHdr header(int n) const {
        SnapHdr h{};
        std::memcpy(h.magic, "INK2KV01", 8);
        h.kvm = (uint32_t) E.kvm;
        h.spec = D.on ? 1 : 0;
        h.ntok = (uint32_t) n;
        h.vocab = VOCAB;
        h.nfull = (uint32_t) full.size();
        h.nring = (uint32_t) ring.size();
        h.nconv = (uint32_t) conv.size();
        h.window = WINDOW;
        for (auto& a : full) h.full_row_bytes += a.b;
        for (auto& a : ring) h.ring_row_bytes += a.b;
        for (auto& c : conv) h.conv_bytes += c.second;
        h.weights = snap_weights_hash();
        return h;
    }
};

uint64_t snap_file_bytes(const SnapHdr& h) {
    const uint64_t n = h.ntok, nr = std::min<uint64_t>(n, h.window);
    return sizeof(SnapHdr) + n * 4 + (uint64_t) h.vocab * 4 + n * h.full_row_bytes + nr * h.ring_row_bytes + h.conv_bytes;
}

constexpr size_t SNAP_STAGE = (size_t) 16 << 20;   // the pinned block every copy goes through (one SESSION line each)

struct SnapIO {   // the file, the pinned block, the progress
    FILE* f;
    Pinned& st;
    uint64_t done, total;
    void moved(uint64_t n) {
        done += n;
        std::printf("SESSION %llu %llu\n", (unsigned long long) done, (unsigned long long) total);
        std::fflush(stdout);
    }
};

// positions [lo, hi) of each array, array after array: device -> file (d2h) or file -> device
bool snap_rows(const std::vector<SessArr>& arrs, int lo, int hi, SnapIO& io, bool d2h) {
    for (const SessArr& a : arrs) {
        const int step = (int) std::max<size_t>(1, io.st.n / a.b);
        for (int q = lo; q < hi; q += step) {
            const int e = std::min(hi, q + step);
            const size_t n = (size_t) (e - q) * a.b;
            if (d2h) {
                sess_rows(a, q, e, io.st.p, true);
                CK(cudaStreamSynchronize(cudaStreamPerThread));
                if (std::fwrite(io.st.p, 1, n, io.f) != n) return false;
            } else {
                if (std::fread(io.st.p, 1, n, io.f) != n) return false;
                sess_rows(a, q, e, io.st.p, false);
                CK(cudaStreamSynchronize(cudaStreamPerThread));
            }
            io.moved(n);
        }
    }
    return true;
}
bool snap_conv(const std::vector<std::pair<void*, size_t>>& conv, SnapIO& io, bool d2h) {
    for (auto& c : conv)
        for (size_t off = 0; off < c.second; off += io.st.n) {
            const size_t n = std::min(io.st.n, c.second - off);
            char* dv = static_cast<char*>(c.first) + off;
            if (d2h) {
                CK(cudaMemcpy(io.st.p, dv, n, cudaMemcpyDeviceToHost));
                if (std::fwrite(io.st.p, 1, n, io.f) != n) return false;
            } else {
                if (std::fread(io.st.p, 1, n, io.f) != n) return false;
                CK(cudaMemcpy(dv, io.st.p, n, cudaMemcpyHostToDevice));
            }
            io.moved(n);
        }
    return true;
}

struct SnapResult {
    std::string kind, why;   // kind empty: done
    size_t bytes = 0;
};
SnapResult snap_refuse(const char* kind, std::string why) { SnapResult r; r.kind = kind; r.why = std::move(why); return r; }

SnapResult snap_save(const std::string& path, const std::vector<int>& hist, const vec& last_logits) {
    if (hist.empty() || last_logits.size() != (size_t) VOCAB) return snap_refuse("invalid", "no conversation to save");
    const SnapLayout L;
    const int n = (int) hist.size();
    const SnapHdr h = L.header(n);
    Pinned st(SNAP_STAGE);
    if (!st.p) return snap_refuse("memory", "no pinned host memory for the copy");
    const std::string part = path + ".part";
    FILE* f = std::fopen(part.c_str(), "wb");
    if (!f) return snap_refuse("io", "cannot open " + part + " for writing");
    CK(cudaDeviceSynchronize());
    SnapIO io{f, st, 0, snap_file_bytes(h)};
    const size_t head = sizeof h + hist.size() * 4 + last_logits.size() * 4;
    bool ok = std::fwrite(&h, sizeof h, 1, f) == 1 && std::fwrite(hist.data(), 4, hist.size(), f) == hist.size() &&
              std::fwrite(last_logits.data(), 4, last_logits.size(), f) == last_logits.size();
    if (ok) io.moved(head);
    ok = ok && snap_rows(L.full, 0, n, io, true) && snap_rows(L.ring, std::max(0, n - WINDOW), n, io, true) &&
         snap_conv(L.conv, io, true);
    ok = std::fclose(f) == 0 && ok;
    if (!ok) {
        std::remove(part.c_str());
        return snap_refuse("storage", "writing " + part + " failed (disk full?)");
    }
    std::error_code ec;
    std::filesystem::rename(part, path, ec);   // replaces an older file
    if (ec) {
        std::filesystem::remove(part, ec);
        return snap_refuse("io", "renaming " + part + " to " + path + " failed");
    }
    SnapResult r;
    r.bytes = (size_t) io.total;
    return r;
}

struct SnapClose { void operator()(FILE* p) const { std::fclose(p); } };
SnapResult snap_restore(const std::string& path, std::vector<int>& hist, vec& last_logits) {
    std::unique_ptr<FILE, SnapClose> f(std::fopen(path.c_str(), "rb"));
    if (!f) return snap_refuse("io", "cannot open " + path);
    SnapHdr h{};
    if (std::fread(&h, sizeof h, 1, f.get()) != 1 || std::memcmp(h.magic, "INK2KV01", 8) != 0)
        return snap_refuse("invalid", path + " is not a strata-ink2 session file");
    if (h.kvm != (uint32_t) E.kvm || h.spec != (D.on ? 1u : 0u))
        return snap_refuse("invalid", "the session was saved with --kv mode " + std::to_string(h.kvm) + " and --spec " +
                                          std::to_string(h.spec) + "; this engine runs --kv mode " + std::to_string(E.kvm) +
                                          " and --spec " + std::to_string(D.on ? 1 : 0));
    if (h.ntok == 0 || h.ntok >= (uint32_t) E.ctx)
        return snap_refuse("invalid", "the session holds " + std::to_string(h.ntok) + " tokens; this engine's context is " +
                                          std::to_string(E.ctx));
    const SnapLayout L;
    const SnapHdr me = L.header((int) h.ntok);
    if (h.weights != me.weights) return snap_refuse("invalid", "the session was saved with other weights");
    if (std::memcmp(&h, &me, sizeof h) != 0) return snap_refuse("invalid", "the session's layout does not match this engine");
    std::error_code ec;
    const uintmax_t size = std::filesystem::file_size(path, ec);
    if (ec || size != snap_file_bytes(h))
        return snap_refuse("invalid", "the session file is truncated (" + std::to_string(ec ? 0 : size) + " of " +
                                          std::to_string(snap_file_bytes(h)) + " bytes)");
    const int n = (int) h.ntok;
    std::vector<int> ids(n);
    vec lg(VOCAB);
    if (std::fread(ids.data(), 4, ids.size(), f.get()) != ids.size() || std::fread(lg.data(), 4, lg.size(), f.get()) != lg.size())
        return snap_refuse("io", "reading " + path + " failed");
    if (!kv_room(n + 1)) return snap_refuse("memory", "the K/V of " + std::to_string(n) + " tokens does not fit beside the expert pools");
    Pinned st(SNAP_STAGE);
    if (!st.p) return snap_refuse("memory", "no pinned host memory for the copy");
    if (g_sess.on) {   // the live conversation is set aside, as for a request of another conversation
        if (!hist.empty()) sess_park(hist, last_logits);
        for (auto& c : g_sess.live) sess_drop(std::move(c.tail));
        g_sess.live.clear();
        g_sess.live_prompt = 0;
    }
    kv_fit(n + 1);
    CK(cudaDeviceSynchronize());
    hist.clear();   // from here the slots hold neither conversation until the copy ends
    last_logits.clear();
    SnapIO io{f.get(), st, 0, (uint64_t) size};
    io.moved(sizeof h + ids.size() * 4 + lg.size() * 4);
    if (!snap_rows(L.full, 0, n, io, false) || !snap_rows(L.ring, std::max(0, n - WINDOW), n, io, false) ||
        !snap_conv(L.conv, io, false)) {
        // the device state is half written: never decode from it (the server starts the engine again)
        std::fprintf(stderr, "[ink2] session restore %s: reading failed after the device state was changed\n", path.c_str());
        std::printf("FATAL restoring the session file failed after the device state was changed: reading %s failed\n", path.c_str());
        std::fflush(stdout);
        std::exit(1);
    }
    hist = std::move(ids);
    last_logits = std::move(lg);
    SnapResult r;
    r.bytes = (size_t) size;
    return r;
}

// one SAVE / RESTORE line of the serve loop
void snap_command(const std::string& line, std::vector<int>& hist, vec& last_logits) {
    const bool save = line[0] == 'S';
    const std::string path = line.substr(save ? 5 : 8);
    const double t0 = now_ms();
    SnapResult r = path.empty() ? snap_refuse("invalid", "missing path")
                 : save ? snap_save(path, hist, last_logits) : snap_restore(path, hist, last_logits);
    if (!r.kind.empty()) {
        std::string one = r.why;   // one protocol line
        for (char& c : one) if (c == '\n' || c == '\r') c = ' ';
        std::fprintf(stderr, "[ink2] session %s %s: %s\n", save ? "save" : "restore", path.c_str(), one.c_str());
        std::printf("SERR %s 0 %s\n", r.kind.c_str(), one.c_str());
    } else {
        std::fprintf(stderr, "[ink2] session %s %zu tokens, %zu bytes, %s, %.1f ms\n", save ? "saved" : "restored", hist.size(),
                     r.bytes, path.c_str(), now_ms() - t0);
        std::printf("%s %zu %zu %.1f\n", save ? "SAVED" : "RESTORED", hist.size(), r.bytes, now_ms() - t0);
    }
    std::fflush(stdout);
}
