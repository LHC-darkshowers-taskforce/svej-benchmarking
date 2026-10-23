#!/usr/bin/env python
"""Stream one HepMC3 file and write compact feature tables.

Outputs (in <outdir>, default $EJ_OUT/features/<sample>/<point>/):
    events.parquet     one row per event (see README for column glossary)
    pions.parquet      one row per dark pion (last copy in its record chain)
    jets.parquet       one row per anti-kT R=1.0 truth jet with pT > --jet-ptmin
    mediators.parquet  one row per mediator X (last copy before decay)
    meta.json          sample/point metadata, cross sections, card tau0/m0, timing

Usage:
    python extract.py --sample mediator_onshell --point mXd2000_mPiD12_ctau10
    python extract.py --input some.hepmc.gz --outdir /path/out [--max-events 50]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import common as C

# ---------------------------------------------------------------------------
# Configuration (documented in README)
# ---------------------------------------------------------------------------
JET_R = 1.0
JET_PTMIN_STORE = 50.0       # GeV, jets stored / used for HT, dphi(MET, jet)
JET_PT_HIGH = 200.0          # GeV, "signal" jet threshold
JET_ETA_MAX = 4.5            # stored jets
CENTRAL_ETA = 2.5            # tracker acceptance
TRK_PTMIN = 1.0              # GeV, "track" = charged final particle above this
TRK_MAX_LXY = 300.0          # mm, tracks with origin beyond this are not reconstructable
PROMPT_R = 1.0               # mm, 3D origin distance below which a track is "prompt" (alpha)
DISP_TRK_MIN = 2.0           # mm, displaced-track Lxy window [2, 300] mm
LXY_EDGES = (1.0, 10.0, 300.0, 3000.0)          # bins: <1,1-10,10-300,300-3000,>3000 mm
CALO_R, CALO_Z = 4000.0, 6000.0                  # mm, crude ATLAS calorimeter envelope
DET_ETA_MAX = 4.9                                # forward calorimeter edge


def _fj():
    import fastjet  # imported lazily so --help is fast and banner printed once
    return fastjet


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_graph(ev, pid_list):
    """Return prod/end vertex index per particle, vertex positions, vertex in/out
    lists, and the copy-parent index (same-PID single-parent predecessor)."""
    npart = len(pid_list)
    nv = len(ev.vertices)
    prod = np.full(npart, -1, dtype=np.int64)
    end = np.full(npart, -1, dtype=np.int64)
    copy_parent = np.full(npart, -1, dtype=np.int64)
    vpos = np.zeros((nv, 4))
    vin, vout = [None] * nv, [None] * nv
    for j, v in enumerate(ev.vertices):
        ins = [p.id - 1 for p in v.particles_in]
        outs = [p.id - 1 for p in v.particles_out]
        vin[j], vout[j] = ins, outs
        for i in ins:
            end[i] = j
        for o in outs:
            prod[o] = j
        # HepMC3 resolves unset positions recursively from the incoming particles
        ps = v.position
        vpos[j] = (ps.x, ps.y, ps.z, ps.t)
        if len(ins) == 1:
            i0 = ins[0]
            p0 = pid_list[i0]
            for o in outs:
                if pid_list[o] == p0:
                    copy_parent[o] = i0
                    break
    return prod, end, vpos, vin, vout, copy_parent


def first_copy(i, copy_parent):
    while copy_parent[i] >= 0:
        i = copy_parent[i]
    return i


# ---------------------------------------------------------------------------
# Per-event processing
# ---------------------------------------------------------------------------
def process_event(ev, ievt, jetdef, jet_ptmin):
    fj = _fj()
    P = ev.numpy.particles
    pid = np.asarray(P.pid, dtype=np.int64)
    npart = len(pid)
    ids = np.asarray(P.id)
    if not np.array_equal(ids, np.arange(1, npart + 1)):
        raise RuntimeError(f"event {ievt}: particle ids not contiguous; unsupported")
    apid = np.abs(pid)
    status = np.asarray(P.status)
    px, py, pz, E = (np.asarray(a, dtype=float) for a in (P.px, P.py, P.pz, P.e))
    mgen = np.asarray(P.generated_mass, dtype=float)
    pt, eta, phi = C.pt_eta_phi(px, py, pz)
    pabs = np.sqrt(px**2 + py**2 + pz**2)

    pid_list = pid.tolist()
    prod, end, vpos, vin, vout, copy_parent = build_graph(ev, pid_list)
    has_prod = prod >= 0
    has_end = end >= 0
    prod_pos = np.where(has_prod[:, None], vpos[np.maximum(prod, 0)], 0.0)
    end_pos = np.where(has_end[:, None], vpos[np.maximum(end, 0)], np.nan)
    origin_lxy = np.hypot(prod_pos[:, 0], prod_pos[:, 1])
    origin_r = np.sqrt(origin_lxy**2 + prod_pos[:, 2] ** 2)

    # last copy: no same-PID daughter
    is_last = np.ones(npart, dtype=bool)
    is_last[copy_parent[copy_parent >= 0]] = False

    final = status == 1
    dark = C.is_dark(apid)
    nu = np.isin(apid, C.PID_NU)
    invis = final & (dark | nu)
    vis = final & ~invis

    # ---------------- dark pions ----------------
    decayed = has_end & (status != 1)
    # Dark pions: flavour mesons (Nf = 3 or 4) and decay-eigenstate pions.  A
    # flavour meson that converts one-body into an eigenstate (--decay-basis
    # eigen) is not counted itself; the eigenstate is.
    is_pi_pid = np.isin(apid, C.PID_PI_ALL) | C.is_eig_pion(apid)
    routed = np.zeros(npart, dtype=bool)
    for ip in np.flatnonzero(is_pi_pid & decayed):
        outs = vout[end[ip]]
        if len(outs) == 1 and C.is_eig_pion(apid[outs[0]]):
            routed[ip] = True
    is_pi = is_pi_pid & is_last & ~routed
    # "decaying" = actually decays to SM particles in this event (stable species
    # never decay; decaying species always do, at any distance)
    is_pi_dec_species = decayed.copy()
    pi_idx = np.flatnonzero(is_pi)

    # label descendants of each decayed dark pion with the pion's particle index
    label = np.full(npart, -1, dtype=np.int64)
    for ip in pi_idx[decayed[pi_idx]]:
        stack = [end[ip]]
        seen = set()
        while stack:
            vj = stack.pop()
            if vj in seen:
                continue
            seen.add(vj)
            for o in vout[vj]:
                if label[o] < 0:
                    label[o] = ip
                if end[o] >= 0:
                    stack.append(end[o])
    from_pi = label >= 0

    pion_rows = []
    pi_lxy, pi_L = [], []
    for ip in pi_idx:
        i0 = first_copy(ip, copy_parent)
        x0 = prod_pos[i0]
        par = vin[prod[i0]] if prod[i0] >= 0 else []
        parent_pid = abs(pid_list[par[0]]) if par else 0
        row = dict(event=ievt, pid=int(pid[ip]), apid=int(apid[ip]), status=int(status[ip]),
                   decaying_species=bool(is_pi_dec_species[ip]), decayed=bool(decayed[ip]),
                   pt=pt[ip], eta=eta[ip], phi=phi[ip], p=pabs[ip], e=E[ip], m=mgen[ip],
                   from_rho=parent_pid in C.PID_RHO, parent_apid=parent_pid,
                   prod_x=x0[0], prod_y=x0[1], prod_z=x0[2])
        if decayed[ip]:
            x1 = end_pos[ip]
            d = x1[:3] - x0[:3]
            L = float(np.sqrt(d @ d))
            lxy = float(np.hypot(d[0], d[1]))
            m = mgen[ip]
            dt = x1[3] - x0[3]
            # visible descendants of this pion
            sel = vis & (label == ip)
            row.update(dec_x=x1[0], dec_y=x1[1], dec_z=x1[2],
                       L=L, Lxy=lxy, Lz=float(d[2]),
                       dec_r=float(np.hypot(x1[0], x1[1])), dec_absz=float(abs(x1[2])),
                       ctau=L * m / pabs[ip] if pabs[ip] > 0 else np.nan,
                       ctau_t=dt * m / E[ip] if E[ip] > 0 else np.nan,
                       betagamma=pabs[ip] / m if m > 0 else np.nan,
                       n_vis_desc=int(sel.sum()), vis_desc_pt=float(pt[sel].sum()),
                       n_bhad_desc=0)
            pi_lxy.append(lxy)
            pi_L.append(L)
        pion_rows.append(row)

    pi_lxy = np.asarray(pi_lxy)
    n_pi = len(pi_idx)
    n_pi_decaying = int(is_pi_dec_species[pi_idx].sum())
    n_pi_stable = n_pi - n_pi_decaying
    n_pi_decayed = int(decayed[pi_idx].sum())
    n_pi_diag = int(np.isin(apid[pi_idx], C.PID_PI_DIAG).sum())
    n_pi_offdiag = int(np.isin(apid[pi_idx], C.PID_PI_OFFDIAG).sum())
    n_pi_eta4 = int((apid[pi_idx] == 4900441).sum())
    edges = (0.0,) + LXY_EDGES + (np.inf,)
    lxy_counts = np.histogram(pi_lxy, bins=edges)[0] if len(pi_lxy) else np.zeros(len(edges) - 1, int)

    pi_dec_out_calo = np.zeros(npart, dtype=bool)  # per pion particle index
    for ip in pi_idx[decayed[pi_idx]]:
        x1 = end_pos[ip]
        pi_dec_out_calo[ip] = (np.hypot(x1[0], x1[1]) > CALO_R) or (abs(x1[2]) > CALO_Z)

    # ---------------- other dark-sector counts ----------------
    n_rho = int((np.isin(apid, C.PID_RHO) & is_last).sum())
    is_dq = np.isin(apid, C.PID_DARK_QUARKS)

    def parent_is_X(i):
        return prod[i] >= 0 and any(apid[q] == C.PID_X for q in vin[prod[i]])

    hard = status == 23
    dq_hard = np.flatnonzero(hard & is_dq)
    dq_fromX = [i for i in dq_hard if parent_is_X(i)]
    sm_parton = (apid <= 5) | (apid == 21)
    smp_hard = np.flatnonzero(hard & sm_parton)
    smp_hard_noX = [i for i in smp_hard if not parent_is_X(i)]

    # ---------------- mediator X ----------------
    isX = apid == C.PID_X
    X_last = np.flatnonzero(isX & is_last)
    X_rows = []
    for ix in X_last:
        row = dict(event=ievt, pid=int(pid[ix]), status=int(status[ix]), pt=pt[ix], eta=eta[ix],
                   phi=phi[ix], m=float(np.sqrt(max(E[ix]**2 - pabs[ix]**2, 0.0))),
                   first_status=int(status[first_copy(ix, copy_parent)]))
        if has_end[ix]:
            ch = vout[end[ix]]
            q = [c for c in ch if apid[c] <= 6]
            d = [c for c in ch if C.is_dark(apid[c])]
            if q:
                row.update(q_pid=int(pid[q[0]]), q_pt=pt[q[0]], q_eta=eta[q[0]], q_phi=phi[q[0]])
            if d:
                row.update(dq_pid=int(pid[d[0]]), dq_pt=pt[d[0]], dq_eta=eta[d[0]], dq_phi=phi[d[0]])
            if q and d:
                row["dR_q_dq"] = float(C.delta_r(eta[q[0]], phi[q[0]], eta[d[0]], phi[d[0]]))
        X_rows.append(row)
    X_sorted = sorted(X_rows, key=lambda r: -r["pt"])

    # ---------------- b hadrons (weakly decaying = no b-hadron daughter) ----------------
    bh = np.flatnonzero(C.is_b_hadron(apid))
    is_bh = np.zeros(npart, dtype=bool)
    is_bh[bh] = True
    n_b = n_b_from_pi = 0
    for i in bh:
        if has_end[i] and any(is_bh[c] for c in vout[end[i]]):
            continue
        n_b += 1
        if from_pi[i]:
            n_b_from_pi += 1
    # per-pion b count
    if n_b_from_pi:
        cnt = {}
        for i in bh:
            if from_pi[i] and not (has_end[i] and any(is_bh[c] for c in vout[end[i]])):
                cnt[label[i]] = cnt.get(label[i], 0) + 1
        for ip, row in zip(pi_idx, pion_rows):
            if ip in cnt:
                row["n_bhad_desc"] = cnt[ip]

    # ---------------- MET ----------------
    # "MET" = vector pT sum of invisible final particles (pointing along them)
    def inv_vec(mask):
        sx, sy = px[mask].sum(), py[mask].sum()
        return float(np.hypot(sx, sy)), float(np.arctan2(sy, sx))

    met, met_phi = inv_vec(invis)
    met_sm, _ = inv_vec(final & nu)
    met_dark, _ = inv_vec(final & dark)
    lost = invis | (vis & (np.abs(eta) > DET_ETA_MAX))
    lost |= vis & from_pi & pi_dec_out_calo[np.maximum(label, 0)]
    met_det, met_det_phi = inv_vec(lost)

    # ---------------- tracks ----------------
    q = C.charges(pid[vis])
    charged = np.zeros(npart, dtype=bool)
    charged[np.flatnonzero(vis)[q != 0]] = True
    trk = charged & (pt > TRK_PTMIN) & (np.abs(eta) < CENTRAL_ETA)
    trk_reco = trk & (origin_lxy < TRK_MAX_LXY)
    trk_disp = trk_reco & (origin_lxy > DISP_TRK_MIN)
    trk_prompt = trk_reco & (origin_r < PROMPT_R)

    vis_pt_sum = pt[vis].sum()
    vis_e_sum = E[vis].sum()

    # ---------------- jets ----------------
    vis_idx = np.flatnonzero(vis)
    pjs = []
    for i in vis_idx:
        pj = fj.PseudoJet(float(px[i]), float(py[i]), float(pz[i]), float(E[i]))
        pj.set_user_index(int(i))
        pjs.append(pj)
    cs = fj.ClusterSequence(pjs, jetdef)
    jets = [j for j in fj.sorted_by_pt(cs.inclusive_jets(jet_ptmin)) if abs(j.eta()) < JET_ETA_MAX]
    dec_pi = pi_idx[decayed[pi_idx]]
    st_pi = pi_idx[~is_pi_dec_species[pi_idx]]
    jet_rows = []
    for r, j in enumerate(jets):
        ci = np.array([c.user_index() for c in j.constituents()], dtype=np.int64)
        jpt, jeta, jphi = j.pt(), j.eta(), j.phi_std()
        cpt = pt[ci]
        spt = cpt.sum()  # scalar constituent-pT sum: denominator of the jet pT fractions
        tpt_all = pt[ci][trk_reco[ci]].sum()
        tpt_prompt = pt[ci][trk_prompt[ci]].sum()
        jet_rows.append(dict(
            event=ievt, rank=r, pt=jpt, eta=jeta, phi=jphi, m=j.m(),
            n_const=len(ci), n_charged=int(charged[ci].sum()),
            n_trk=int(trk_reco[ci].sum()), n_trk_disp=int(trk_disp[ci].sum()),
            alpha=float(tpt_prompt / tpt_all) if tpt_all > 0 else -1.0,
            frac_pt_disp1=float(cpt[origin_lxy[ci] > 1.0].sum() / spt),
            frac_pt_disp10=float(cpt[origin_lxy[ci] > 10.0].sum() / spt),
            frac_pt_darkpi=float(cpt[from_pi[ci]].sum() / spt),
            frac_pt_charged=float(cpt[charged[ci]].sum() / spt),
            n_darkpi_dR=int((C.delta_r(eta[dec_pi], phi[dec_pi], jeta, jphi) < JET_R).sum()),
            n_stablepi_dR=int((C.delta_r(eta[st_pi], phi[st_pi], jeta, jphi) < JET_R).sum()),
        ))
    jdf = jet_rows
    jpt_arr = np.array([j["pt"] for j in jdf]) if jdf else np.zeros(0)
    jeta_arr = np.array([j["eta"] for j in jdf]) if jdf else np.zeros(0)
    jphi_arr = np.array([j["phi"] for j in jdf]) if jdf else np.zeros(0)
    high = jpt_arr > JET_PT_HIGH
    high_c = high & (np.abs(jeta_arr) < CENTRAL_ETA)
    ht50 = float(jpt_arr.sum())
    ht200 = float(jpt_arr[high].sum())
    n_ej = sum(1 for j, h in zip(jdf, high_c) if h and (j["alpha"] < 0.25))

    ev_row = dict(
        event=ievt, weight=float(ev.weights[0]) if len(ev.weights) else 1.0,
        n_particles=npart,
        # dark pions
        n_pi=n_pi, n_pi_decaying=n_pi_decaying, n_pi_stable=n_pi_stable, n_pi_decayed=n_pi_decayed,
        n_pi_diag=n_pi_diag, n_pi_offdiag=n_pi_offdiag, n_pi_eta4=n_pi_eta4,
        frac_pi_stable=n_pi_stable / n_pi if n_pi else np.nan,
        n_rho=n_rho, n_dark_hadrons=n_pi + n_rho,
        pi_lxy_mean=float(pi_lxy.mean()) if len(pi_lxy) else np.nan,
        pi_lxy_max=float(pi_lxy.max()) if len(pi_lxy) else np.nan,
        pi_lxy_min=float(pi_lxy.min()) if len(pi_lxy) else np.nan,
        pi_lxy_median=float(np.median(pi_lxy)) if len(pi_lxy) else np.nan,
        n_pi_lxy_lt1=int(lxy_counts[0]), n_pi_lxy_1_10=int(lxy_counts[1]),
        n_pi_lxy_10_300=int(lxy_counts[2]), n_pi_lxy_300_3000=int(lxy_counts[3]),
        n_pi_lxy_gt3000=int(lxy_counts[4]),
        n_pi_outside_calo=int(pi_dec_out_calo.sum()),
        pi_pt_mean=float(pt[pi_idx].mean()) if n_pi else np.nan,
        pi_pt_lead=float(pt[pi_idx].max()) if n_pi else np.nan,
        # hard process
        n_dq_hard=len(dq_hard), n_dq_fromX=len(dq_fromX), n_dq_ME=len(dq_hard) - len(dq_fromX),
        n_sm_parton_hard=len(smp_hard_noX),
        # mediator
        n_X=len(X_last), n_X_records=int(isX.sum()), n_X_hard=int((isX & (status == 22)).sum()),
        X1_pt=X_sorted[0]["pt"] if X_sorted else np.nan,
        X1_eta=X_sorted[0]["eta"] if X_sorted else np.nan,
        X1_m=X_sorted[0]["m"] if X_sorted else np.nan,
        X2_pt=X_sorted[1]["pt"] if len(X_sorted) > 1 else np.nan,
        X2_eta=X_sorted[1]["eta"] if len(X_sorted) > 1 else np.nan,
        # b hadrons
        n_bhad=n_b, n_bhad_from_pi=n_b_from_pi,
        # visible energy
        f_vis_pt_darkpi=float(pt[vis & from_pi].sum() / vis_pt_sum) if vis_pt_sum > 0 else np.nan,
        f_vis_e_darkpi=float(E[vis & from_pi].sum() / vis_e_sum) if vis_e_sum > 0 else np.nan,
        vis_pt_sum=float(vis_pt_sum),
        # MET
        met=met, met_phi=met_phi, met_sm=met_sm, met_dark=met_dark,
        met_det=met_det, met_det_phi=met_det_phi,
        # tracks
        n_trk=int(trk_reco.sum()), n_trk_disp=int(trk_disp.sum()),
        f_trk_disp=float(trk_disp.sum() / trk_reco.sum()) if trk_reco.sum() else np.nan,
        # jets
        n_jet50=len(jdf), n_jet200=int(high.sum()), n_jet200_c=int(high_c.sum()),
        ht50=ht50, ht200=ht200, n_ej_proxy=int(n_ej),
        met_over_ht=met / ht50 if ht50 > 0 else np.nan,
        met_det_over_ht=met_det / ht50 if ht50 > 0 else np.nan,
    )
    for k, tag in ((0, "j1"), (1, "j2")):
        if len(jdf) > k:
            for f in ("pt", "eta", "phi", "m", "alpha", "n_trk", "n_trk_disp",
                      "frac_pt_darkpi", "frac_pt_disp1", "n_darkpi_dR"):
                ev_row[f"{tag}_{f}"] = jdf[k][f]
        else:
            for f in ("pt", "eta", "phi", "m", "alpha", "frac_pt_darkpi", "frac_pt_disp1"):
                ev_row[f"{tag}_{f}"] = np.nan
            for f in ("n_trk", "n_trk_disp", "n_darkpi_dR"):
                ev_row[f"{tag}_{f}"] = -1
    if len(jdf) > 1:
        j1, j2 = jdf[0], jdf[1]
        ev_row["dR_j1j2"] = float(C.delta_r(j1["eta"], j1["phi"], j2["eta"], j2["phi"]))
        ev_row["dphi_j1j2"] = float(abs(C.dphi(j1["phi"], j2["phi"])))
        p4 = [jets[0].px() + jets[1].px(), jets[0].py() + jets[1].py(),
              jets[0].pz() + jets[1].pz(), jets[0].E() + jets[1].E()]
        ev_row["mjj"] = float(np.sqrt(max(p4[3]**2 - p4[0]**2 - p4[1]**2 - p4[2]**2, 0.0)))
    else:
        ev_row.update(dR_j1j2=np.nan, dphi_j1j2=np.nan, mjj=np.nan)
    if len(jdf):
        dps = np.abs(C.dphi(jphi_arr, met_phi))
        ev_row["dphi_met_jclosest"] = float(dps.min())
        ev_row["dphi_met_j1"] = float(dps[0])
        dps_det = np.abs(C.dphi(jphi_arr, met_det_phi))
        ev_row["dphi_metdet_jclosest"] = float(dps_det.min())
    else:
        ev_row.update(dphi_met_jclosest=np.nan, dphi_met_j1=np.nan, dphi_metdet_jclosest=np.nan)
    return ev_row, pion_rows, jet_rows, X_rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(hepmc: Path, outdir: Path, point: C.Point | None, max_events: int | None,
        jet_ptmin: float, progress: int = 100):
    import pyhepmc
    fj = _fj()
    jetdef = fj.JetDefinition(fj.antikt_algorithm, JET_R)
    outdir.mkdir(parents=True, exist_ok=True)

    ev_rows, pi_rows, jet_rows, X_rows = [], [], [], []
    xsec = xsec_err = None
    t0 = time.time()
    n = 0
    with pyhepmc.open(str(hepmc)) as f:
        for ev in f:
            if ev is None:
                break
            if ev.length_unit.name != "MM" or ev.momentum_unit.name != "GEV":
                raise RuntimeError(f"unexpected units {ev.length_unit} {ev.momentum_unit}")
            e, p, j, x = process_event(ev, n, jetdef, jet_ptmin)
            ev_rows.append(e)
            pi_rows += p
            jet_rows += j
            X_rows += x
            cs = ev.cross_section
            if cs is not None:
                xsec, xsec_err = float(cs.xsec()), float(cs.xsec_err())
            n += 1
            if progress and n % progress == 0:
                print(f"  {n} events, {time.time() - t0:.1f} s", flush=True)
            if max_events and n >= max_events:
                break
    dt = time.time() - t0

    pd.DataFrame(ev_rows).to_parquet(outdir / "events.parquet", index=False)
    pd.DataFrame(pi_rows).to_parquet(outdir / "pions.parquet", index=False)
    pd.DataFrame(jet_rows).to_parquet(outdir / "jets.parquet", index=False)
    pd.DataFrame(X_rows).to_parquet(outdir / "mediators.parquet", index=False)

    meta = dict(input=str(hepmc), n_events=n, processing_time_s=round(dt, 1),
                xsec_pb=xsec, xsec_err_pb=xsec_err,
                sum_weights=float(sum(r["weight"] for r in ev_rows)),
                config=dict(jet_R=JET_R, jet_ptmin=jet_ptmin, jet_pt_high=JET_PT_HIGH,
                            jet_eta_max=JET_ETA_MAX, central_eta=CENTRAL_ETA, trk_ptmin=TRK_PTMIN,
                            trk_max_lxy=TRK_MAX_LXY, prompt_r=PROMPT_R, disp_trk_min=DISP_TRK_MIN,
                            lxy_edges=LXY_EDGES, calo_r=CALO_R, calo_z=CALO_Z,
                            det_eta_max=DET_ETA_MAX))
    if point is not None:
        meta.update(point.asdict())
        card = C.parse_pythia_card(point.card)
        meta["card"] = dict(header=card["header"], merging=card["merging"],
                            tau0_mm={str(k): v for k, v in card["tau0_mm"].items()},
                            m0={str(k): v for k, v in card["m0"].items()})
        meta.update(C.parse_run_logs(point.event_dir))
    with open(outdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {outdir}  ({n} events, {dt:.1f} s, {dt / max(n, 1) * 1e3:.0f} ms/event)")
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", choices=C.SAMPLES)
    ap.add_argument("--point", help="grid point name, e.g. mXd2000_mPiD12_ctau10")
    ap.add_argument("--input", type=Path, help="explicit HepMC file (overrides sample/point lookup)")
    ap.add_argument("--outdir", type=Path, help="output directory (default: $EJ_OUT/features/<sample>/<point>)")
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--jet-ptmin", type=float, default=JET_PTMIN_STORE)
    ap.add_argument("--force", action="store_true", help="re-run even if meta.json exists")
    a = ap.parse_args(argv)

    point = None
    if a.sample and a.point:
        point = C.Point(sample=a.sample, name=a.point, **C.parse_point_name(a.point))
    hepmc = a.input or (point.hepmc if point else None)
    if hepmc is None:
        ap.error("give --sample and --point, or --input")
    outdir = a.outdir or (point.outdir if point else None)
    if outdir is None:
        ap.error("--outdir required with --input when no --sample/--point")
    if (outdir / "meta.json").exists() and not a.force:
        print(f"{outdir}/meta.json exists; use --force to redo")
        return 0
    if not Path(hepmc).is_file():
        print(f"missing input {hepmc}", file=sys.stderr)
        return 1
    run(Path(hepmc), Path(outdir), point, a.max_events, a.jet_ptmin)
    return 0


if __name__ == "__main__":
    sys.exit(main())
