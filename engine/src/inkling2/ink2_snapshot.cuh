// K/V snapshots for --serve: SAVE <path> writes the live conversation's state to a file, LOAD <path> reads one back, so
// that a long prompt is not read again after a restart (reading the K/V of a long prompt from disk is much faster than
// computing it again).
//
//   SAVE <path>   ->  SAVED <tokens> <bytes> <ms>    or  ERR <reason>
//   LOAD <path>   ->  LOADED <tokens> <bytes> <ms>   or  ERR <reason>
//
// The state is the conversation cache's (ink2_session.cuh, sess_arrays): after n tokens, the full-attention rows [0, n),
// the sliding-window rows of positions [n - WINDOW, n), the conv rings, with --spec the drafter's K/V; plus the n tokens
// and the logits after the last one.  Rows are in position order, so a file does not depend on --chunk, --max-context
// or --kv-grow; it does depend on --kv and --spec, which the header records (a LOAD with others is refused).  The file
// does not record which weights made it: load a snapshot into the model that saved it.
//
// A LOAD refused before the slots are written leaves the live conversation as it was; one that fails while writing them
// leaves none (the next request reads its prompt from the start).  With the conversation cache on, a LOAD first parks
// the live conversation, as a request for another conversation would.  SAVE writes <path>.part and renames it, so a
// failed SAVE leaves an older file at <path> as it was.  Nothing here runs unless the controller sends one of the lines.

struct SnapHdr {
    char magic[8];   // "INK2KV01"
    uint32_t kvm, spec, ntok, vocab;
    uint32_t nfull, nring, nconv, window;
    uint64_t full_row_bytes, ring_row_bytes, conv_bytes;
};

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
        return h;
    }
};

uint64_t snap_file_bytes(const SnapHdr& h) {
    const uint64_t n = h.ntok, nr = std::min<uint64_t>(n, h.window);
    return sizeof(SnapHdr) + n * 4 + (uint64_t) h.vocab * 4 + n * h.full_row_bytes + nr * h.ring_row_bytes + h.conv_bytes;
}

constexpr size_t SNAP_STAGE = (size_t) 64 << 20;   // the pinned buffer every copy goes through

// positions [lo, hi) of each array, array after array: device -> file (d2h) or file -> device
bool snap_rows(const std::vector<SessArr>& arrs, int lo, int hi, Pinned& st, FILE* f, bool d2h) {
    for (const SessArr& a : arrs) {
        const int step = (int) std::max<size_t>(1, st.n / a.b);
        for (int q = lo; q < hi; q += step) {
            const int e = std::min(hi, q + step);
            const size_t n = (size_t) (e - q) * a.b;
            if (d2h) {
                sess_rows(a, q, e, st.p, true);
                CK(cudaStreamSynchronize(cudaStreamPerThread));
                if (std::fwrite(st.p, 1, n, f) != n) return false;
            } else {
                if (std::fread(st.p, 1, n, f) != n) return false;
                sess_rows(a, q, e, st.p, false);
                CK(cudaStreamSynchronize(cudaStreamPerThread));
            }
        }
    }
    return true;
}
bool snap_conv(const std::vector<std::pair<void*, size_t>>& conv, Pinned& st, FILE* f, bool d2h) {
    for (auto& c : conv)
        for (size_t off = 0; off < c.second; off += st.n) {
            const size_t n = std::min(st.n, c.second - off);
            char* dv = static_cast<char*>(c.first) + off;
            if (d2h) {
                CK(cudaMemcpy(st.p, dv, n, cudaMemcpyDeviceToHost));
                if (std::fwrite(st.p, 1, n, f) != n) return false;
            } else {
                if (std::fread(st.p, 1, n, f) != n) return false;
                CK(cudaMemcpy(dv, st.p, n, cudaMemcpyHostToDevice));
            }
        }
    return true;
}

std::string snap_save(const std::string& path, const std::vector<int>& hist, const vec& last_logits, size_t& nbytes) {
    if (hist.empty() || last_logits.size() != (size_t) VOCAB) return "nothing to save: no conversation yet";
    const SnapLayout L;
    const int n = (int) hist.size();
    const SnapHdr h = L.header(n);
    Pinned st(SNAP_STAGE);
    if (!st.p) return "no pinned host memory for the copy";
    const std::string part = path + ".part";
    FILE* f = std::fopen(part.c_str(), "wb");
    if (!f) return "cannot open " + part + " for writing";
    CK(cudaDeviceSynchronize());
    bool ok = std::fwrite(&h, sizeof h, 1, f) == 1 && std::fwrite(hist.data(), 4, hist.size(), f) == hist.size() &&
              std::fwrite(last_logits.data(), 4, last_logits.size(), f) == last_logits.size() &&
              snap_rows(L.full, 0, n, st, f, true) && snap_rows(L.ring, std::max(0, n - WINDOW), n, st, f, true) &&
              snap_conv(L.conv, st, f, true);
    ok = std::fclose(f) == 0 && ok;
    std::error_code ec;
    if (ok) std::filesystem::rename(part, path, ec);   // replaces an older file
    if (!ok || ec) {
        std::filesystem::remove(part, ec);
        return "writing " + path + " failed (disk full?)";
    }
    nbytes = (size_t) snap_file_bytes(h);
    return std::string();
}

struct SnapClose { void operator()(FILE* p) const { std::fclose(p); } };
std::string snap_load(const std::string& path, std::vector<int>& hist, vec& last_logits, size_t& nbytes) {
    std::unique_ptr<FILE, SnapClose> f(std::fopen(path.c_str(), "rb"));
    if (!f) return "cannot open " + path;
    SnapHdr h{};
    if (std::fread(&h, sizeof h, 1, f.get()) != 1 || std::memcmp(h.magic, "INK2KV01", 8) != 0) return path + " is not a strata-ink2 snapshot";
    if (h.kvm != (uint32_t) E.kvm || h.spec != (D.on ? 1u : 0u))
        return "the snapshot was saved with --kv mode " + std::to_string(h.kvm) + " and --spec " + std::to_string(h.spec) +
               "; this engine runs --kv mode " + std::to_string(E.kvm) + " and --spec " + std::to_string(D.on ? 1 : 0);
    if (h.ntok == 0 || h.ntok >= (uint32_t) E.ctx)
        return "the snapshot holds " + std::to_string(h.ntok) + " tokens; this engine's context is " + std::to_string(E.ctx);
    const SnapLayout L;
    const SnapHdr me = L.header((int) h.ntok);
    if (std::memcmp(&h, &me, sizeof h) != 0) return "the snapshot's layout does not match this engine";
    std::error_code ec;
    const uintmax_t size = std::filesystem::file_size(path, ec);
    if (ec || size != snap_file_bytes(h)) return "the snapshot is truncated (" + std::to_string(ec ? 0 : size) + " of " +
                                                  std::to_string(snap_file_bytes(h)) + " bytes)";
    const int n = (int) h.ntok;
    std::vector<int> ids(n);
    vec lg(VOCAB);
    if (std::fread(ids.data(), 4, ids.size(), f.get()) != ids.size() || std::fread(lg.data(), 4, lg.size(), f.get()) != lg.size())
        return "reading " + path + " failed";
    if (!kv_room(n + 1)) return "the K/V of " + std::to_string(n) + " tokens does not fit beside the expert pools on this card";
    Pinned st(SNAP_STAGE);
    if (!st.p) return "no pinned host memory for the copy";
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
    if (!snap_rows(L.full, 0, n, st, f.get(), false) || !snap_rows(L.ring, std::max(0, n - WINDOW), n, st, f.get(), false) ||
        !snap_conv(L.conv, st, f.get(), false))
        return "reading " + path + " failed half way: no conversation kept";
    hist = std::move(ids);
    last_logits = std::move(lg);
    nbytes = (size_t) size;
    return std::string();
}
